#!/usr/bin/env python3
"""
Fast screening approach for 100k.
Key insight: run MANY cheap perturbation+short-optimize cycles (500 iters)
instead of FEW expensive ones (5000 iters). Only do full polish on winners.

This should give 10x more trials per hour, increasing chances of finding
a perturbation that escapes the current basin.
"""
import numpy as np
import torch
import time
import sys

device = torch.device('cuda')


def gpu_score_exact(f_t):
    n = f_t.shape[0]; nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
    h = 1.0/(nc+1); z = torch.zeros(1, device=device, dtype=torch.float64)
    y = torch.cat([z, conv, z]); y0, y1 = y[:-1], y[1:]
    l2sq = (h/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv)/(nc+1); linf = torch.max(conv)
    return (l2sq/(l1*linf)).item()


def quick_dinkelbach(f_np, n_inner=500, beta=1e8):
    """Very fast single-outer Dinkelbach. ~30s at 100k."""
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    lam = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))
    best_C = lam; best_f = f_np.copy()

    w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [w_t], lr=1.0, max_iter=n_inner,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-14, tolerance_change=1e-15
    )
    best_w = [w_t.data.clone()]
    best_C_inner = [0.0]

    def closure():
        optimizer.zero_grad()
        f = w_t ** 2
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F*F, n=nfft)[:nc]
        conv = torch.maximum(conv, torch.zeros(1, device=device, dtype=torch.float64))
        hh = 1.0/(nc+1)
        zz = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([zz, conv, zz])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv)/(nc+1)
        g_max = torch.max(conv)
        linf_proxy = g_max * torch.exp(
            torch.logsumexp(beta * (conv / (g_max + 1e-18) - 1.0), dim=0) / beta
        )
        obj = l2sq - lam * l1 * linf_proxy
        C_exact = l2sq.item() / (l1.item() * g_max.item())
        if C_exact > best_C_inner[0]:
            best_C_inner[0] = C_exact
            best_w[0] = w_t.data.clone()
        loss = -obj
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        pass

    f_new = (best_w[0] ** 2).cpu().numpy()
    C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
    if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
        return f_new, C_new
    return best_f, best_C


def full_dinkelbach(f_np, n_outer=5, n_inner=5000, beta=1e8):
    """Full Dinkelbach for polishing winners."""
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    lam = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))
    best_C = lam; best_f = f_np.copy()

    for outer in range(n_outer):
        w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [w_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-15, tolerance_change=1e-16
        )
        best_w_inner = [w_t.data.clone()]
        best_C_inner = [0.0]

        def closure():
            optimizer.zero_grad()
            f = w_t ** 2
            F = torch.fft.rfft(f, n=nfft)
            conv = torch.fft.irfft(F*F, n=nfft)[:nc]
            conv = torch.maximum(conv, torch.zeros(1, device=device, dtype=torch.float64))
            hh = 1.0/(nc+1)
            zz = torch.zeros(1, device=device, dtype=torch.float64)
            y = torch.cat([zz, conv, zz])
            y0, y1 = y[:-1], y[1:]
            l2sq = (hh/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
            l1 = torch.sum(conv)/(nc+1)
            g_max = torch.max(conv)
            linf_proxy = g_max * torch.exp(
                torch.logsumexp(beta * (conv / (g_max + 1e-18) - 1.0), dim=0) / beta
            )
            obj = l2sq - current_lam * l1 * linf_proxy
            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_C_inner[0]:
                best_C_inner[0] = C_exact
                best_w_inner[0] = w_t.data.clone()
            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass

        w = best_w_inner[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new
        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)):
            best_C = C_new; best_f = f_new.copy()
        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def get_autoconv(f_np):
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    n = len(f_np); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    return torch.fft.irfft(F*F, n=nfft)[:nc].cpu().numpy()


if __name__ == "__main__":
    print("=" * 70)
    print("Fast screening optimizer for 100k")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    try:
        f_best = np.maximum(np.load('best_impevo_100k.npy').astype(np.float64), 0.0)
    except:
        f_best = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)

    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")
    sys.stdout.flush()

    save_file = 'best_impevo_100k.npy'
    rng = np.random.default_rng(777)
    n_improvements = 0
    n_screens = 0
    n_polishes = 0

    # Analyze autoconvolution structure
    conv = get_autoconv(f_best)
    g_max = np.max(conv)
    plateau = conv / (g_max + 1e-18)
    n_dips = np.sum(plateau < 0.999)
    print(f"Autoconvolution: {n_dips} dips below 99.9% of max")

    for round_num in range(10000):
        # Cross-pollinate every 50 rounds
        if round_num % 50 == 0:
            try:
                f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
                if len(f_shared) == n:
                    C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                    if C_shared > C_best and C_shared < 1.0:
                        C_best = C_shared; f_best = f_shared.copy()
                        conv = get_autoconv(f_best)
                        g_max = np.max(conv)
            except: pass

        # Generate perturbation — mix of strategies
        strategy = rng.integers(10)
        f_perturbed = f_best.copy()

        if strategy < 3:
            # Targeted: boost f at positions contributing to autoconv dips
            plateau = conv / (g_max + 1e-18)
            dip_positions = np.where(plateau < 0.999)[0]
            if len(dip_positions) > 0:
                dip_k = rng.choice(dip_positions)
                j_lo = max(0, dip_k - n + 1)
                j_hi = min(n, dip_k + 1)
                # Find contributing positions
                for _ in range(3):
                    j = rng.integers(j_lo, j_hi)
                    k_j = dip_k - j
                    if 0 <= j < n and 0 <= k_j < n:
                        boost = rng.uniform(0.001, 0.05)
                        f_perturbed[j] *= (1 + boost)
                        f_perturbed[k_j] *= (1 + boost)

        elif strategy < 5:
            # Scale a random region
            lo = rng.integers(0, n - 100)
            length = rng.integers(10, 500)
            hi = min(n, lo + length)
            scale = 1.0 + rng.normal(0, 0.01)
            f_perturbed[lo:hi] *= max(0.9, min(1.1, scale))

        elif strategy < 7:
            # Multiplicative noise on non-zero positions
            nonzero = f_best > np.max(f_best) * 0.001
            noise = np.ones(n)
            noise[nonzero] = 1.0 + rng.normal(0, 0.005, size=np.sum(nonzero))
            f_perturbed = f_best * noise

        elif strategy < 9:
            # Shift a random tooth
            # Find tooth peaks
            peaks = []
            for i in range(1, n-1):
                if f_best[i] > f_best[i-1] and f_best[i] > f_best[i+1] and f_best[i] > np.max(f_best) * 0.01:
                    peaks.append(i)
            if len(peaks) > 0:
                peak = rng.choice(peaks)
                shift = rng.integers(-3, 4)
                radius = 10
                lo = max(0, peak - radius)
                hi = min(n, peak + radius)
                tooth = f_perturbed[lo:hi].copy()
                f_perturbed[lo:hi] = 0
                new_lo = max(0, lo + shift)
                new_hi = min(n, new_lo + len(tooth))
                if new_hi > new_lo:
                    f_perturbed[new_lo:new_hi] = tooth[:new_hi - new_lo]

        else:
            # Additive noise
            sigma = rng.uniform(0.0001, 0.005) * np.max(f_best)
            f_perturbed = f_best + rng.normal(0, sigma, size=n)

        f_perturbed = np.maximum(f_perturbed, 0.0)

        # Quick screen (500 L-BFGS iters, ~30s)
        f_screened, C_screened = quick_dinkelbach(f_perturbed, n_inner=500, beta=1e8)
        n_screens += 1

        # If promising (beats current best), do full polish
        if C_screened > C_best:
            n_polishes += 1
            print(f"  R{round_num}: Promising screen C={C_screened:.13f}, polishing...", flush=True)

            # Multi-beta polish
            f_polished = f_screened
            C_polished = C_screened
            for beta in [1e8, 5e8, 1e9]:
                f_polished, C_polished = full_dinkelbach(f_polished, n_outer=3, n_inner=5000, beta=beta)

            if C_polished > C_best and C_polished < 1.0 and np.all(np.isfinite(f_polished)) and np.max(f_polished) < 1e6:
                C_best = C_polished; f_best = f_polished.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                conv = get_autoconv(f_best)
                g_max = np.max(conv)
                print(f"  R{round_num}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        if round_num % 100 == 0:
            elapsed = time.time() - t0
            rate = n_screens / max(1, elapsed) * 3600
            print(f"  R{round_num}: C_best={C_best:.13f} [{elapsed:.0f}s, {n_screens} screens, "
                  f"{n_polishes} polishes, {n_improvements} impr, {rate:.0f} screens/hr]", flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_screens} screens, {n_improvements} improvements")

#!/usr/bin/env python3
"""
Fresh comb construction + Dinkelbach for 100k.
Instead of perturbing the existing solution, build multiple fresh combs
with different numbers of teeth, spacings, and profiles, then optimize
each one with Dinkelbach + high beta.

Key idea: the existing solution may be in a suboptimal basin determined
by its comb structure (number/spacing of teeth). A different number of
teeth might reach a higher C value.

Also tries: tooth-count interpolation between the current best and
nearby tooth counts.
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


def dinkelbach_improve(f_np, n_outer=5, n_inner=5000, beta=1e8):
    """Standard Dinkelbach + L-BFGS."""
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
        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()
        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def make_comb(n, n_teeth, width_frac=0.3, profile='triangle', rng=None):
    """
    Construct a comb function with n_teeth teeth.
    width_frac: tooth width as fraction of spacing
    profile: 'triangle', 'gaussian', 'rect', 'cosine'
    """
    if rng is None:
        rng = np.random.default_rng()

    f = np.zeros(n, dtype=np.float64)
    spacing = n / n_teeth

    for i in range(n_teeth):
        center = int((i + 0.5) * spacing)
        half_width = max(1, int(spacing * width_frac / 2))

        lo = max(0, center - half_width)
        hi = min(n, center + half_width + 1)
        w = hi - lo

        if profile == 'triangle':
            x = np.linspace(-1, 1, w)
            tooth = np.maximum(1 - np.abs(x), 0)
        elif profile == 'gaussian':
            x = np.linspace(-2, 2, w)
            tooth = np.exp(-x**2)
        elif profile == 'rect':
            tooth = np.ones(w)
        elif profile == 'cosine':
            x = np.linspace(-np.pi/2, np.pi/2, w)
            tooth = np.cos(x)
        else:
            tooth = np.ones(w)

        f[lo:hi] = tooth

    return f


def make_comb_irregular(n, n_teeth, jitter=0.1, height_var=0.1, rng=None):
    """Comb with slightly irregular spacing and heights."""
    if rng is None:
        rng = np.random.default_rng()

    spacing = n / n_teeth
    f = np.zeros(n, dtype=np.float64)

    for i in range(n_teeth):
        center = int((i + 0.5) * spacing + rng.normal(0, spacing * jitter))
        center = max(0, min(n-1, center))
        height = 1.0 + rng.normal(0, height_var)
        height = max(0.1, height)
        half_width = max(1, int(spacing * 0.15))

        lo = max(0, center - half_width)
        hi = min(n, center + half_width + 1)
        w = hi - lo

        x = np.linspace(-1, 1, w)
        tooth = height * np.maximum(1 - np.abs(x), 0)
        f[lo:hi] = np.maximum(f[lo:hi], tooth)

    return f


def make_comb_from_primes(n, p):
    """
    Construct comb using quadratic residues mod p.
    These give near-optimal autocorrelation properties.
    """
    # Quadratic residues mod p
    qr = set()
    for x in range(1, p):
        qr.add((x * x) % p)

    f = np.zeros(n, dtype=np.float64)
    # Place teeth at positions corresponding to QR
    scale = n / p
    half_width = max(1, int(scale * 0.15))

    for r in sorted(qr):
        center = int(r * scale)
        if center >= n:
            continue
        lo = max(0, center - half_width)
        hi = min(n, center + half_width + 1)
        w = hi - lo
        x = np.linspace(-1, 1, w)
        tooth = np.maximum(1 - np.abs(x), 0)
        f[lo:hi] = np.maximum(f[lo:hi], tooth)

    return f


if __name__ == "__main__":
    print("=" * 70)
    print("Fresh comb construction + Dinkelbach for 100k")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Load current best
    try:
        f_best = np.maximum(np.load('best_impevo_100k.npy').astype(np.float64), 0.0)
    except:
        f_best = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)

    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Current best: n={n}, C = {C_best:.13f}")

    # Count teeth in current solution
    threshold = np.max(f_best) * 0.005
    above = f_best > threshold
    tooth_count = 0
    in_tooth = False
    for i in range(n):
        if above[i] and not in_tooth:
            in_tooth = True
            tooth_count += 1
        elif not above[i]:
            in_tooth = False
    print(f"Current solution has ~{tooth_count} teeth")

    save_file = 'best_impevo_100k.npy'
    rng = np.random.default_rng(2024)
    n_improvements = 0

    # Phase 1: Try different tooth counts near the current count
    # The optimal tooth count might not be exactly what we have
    print("\n--- Phase 1: Tooth count sweep ---")

    tooth_counts = list(range(max(100, tooth_count - 200), tooth_count + 201, 20))
    # Add some wider range too
    tooth_counts += [500, 750, 1000, 1500, 2000, 2500, 3000]
    tooth_counts = sorted(set(tooth_counts))

    for nt in tooth_counts:
        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best and C_shared < 1.0:
                    C_best = C_shared; f_best = f_shared.copy()
        except: pass

        for profile in ['triangle', 'cosine']:
            for width_frac in [0.2, 0.3, 0.4]:
                f_comb = make_comb(n, nt, width_frac=width_frac, profile=profile)
                C_init = gpu_score_exact(torch.tensor(f_comb, dtype=torch.float64, device=device))

                # Quick Dinkelbach at moderate beta
                f_opt, C_opt = dinkelbach_improve(f_comb, n_outer=3, n_inner=2000, beta=1e6)

                if C_opt > 0.95:  # Only full optimize promising ones
                    # Full Dinkelbach at high beta
                    f_opt2, C_opt2 = dinkelbach_improve(f_opt, n_outer=3, n_inner=5000, beta=1e8)

                    if C_opt2 > C_best and C_opt2 < 1.0 and np.all(np.isfinite(f_opt2)):
                        C_best = C_opt2; f_best = f_opt2.copy()
                        np.save(save_file, f_best)
                        n_improvements += 1
                        print(f"  nt={nt}, {profile}, w={width_frac}: NEW BEST C = {C_best:.13f} !!!")
                        sys.stdout.flush()

                    if C_opt2 > 0.96:
                        print(f"  nt={nt}, {profile}, w={width_frac}: init={C_init:.6f} → "
                              f"quick={C_opt:.10f} → full={C_opt2:.13f} [{time.time()-t0:.0f}s]", flush=True)
                elif nt == tooth_counts[0] or nt % 200 == 0:
                    print(f"  nt={nt}, {profile}, w={width_frac}: init={C_init:.6f} → "
                          f"quick={C_opt:.10f} [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nAfter tooth count sweep: C = {C_best:.13f}")

    # Phase 2: Quadratic residue combs (number-theoretic construction)
    print("\n--- Phase 2: Number-theoretic combs ---")

    # Primes near various tooth counts
    from sympy import nextprime, prevprime
    target_teeth = [500, 750, 1000, 1200, 1300, 1314, 1400, 1500, 2000, 2500, 3000]

    for target in target_teeth:
        p = nextprime(2 * target)  # QR gives ~p/2 teeth
        f_qr = make_comb_from_primes(n, p)
        C_init = gpu_score_exact(torch.tensor(f_qr, dtype=torch.float64, device=device))

        f_opt, C_opt = dinkelbach_improve(f_qr, n_outer=3, n_inner=2000, beta=1e6)

        if C_opt > 0.95:
            f_opt2, C_opt2 = dinkelbach_improve(f_opt, n_outer=3, n_inner=5000, beta=1e8)

            if C_opt2 > C_best and C_opt2 < 1.0 and np.all(np.isfinite(f_opt2)):
                C_best = C_opt2; f_best = f_opt2.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  QR p={p} (~{len([x for x in range(1,p) if pow(x,2,p) != 0])} teeth): "
                      f"NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        print(f"  QR p={p}: init={C_init:.6f} → opt={C_opt:.10f} [{time.time()-t0:.0f}s]", flush=True)

    # Phase 3: Irregular combs with random perturbation
    print("\n--- Phase 3: Irregular combs ---")

    best_tooth_counts = [tooth_count - 50, tooth_count - 20, tooth_count,
                          tooth_count + 20, tooth_count + 50]

    for trial in range(50):
        nt = rng.choice(best_tooth_counts)
        jitter = rng.uniform(0.01, 0.2)
        height_var = rng.uniform(0.01, 0.3)

        f_irreg = make_comb_irregular(n, nt, jitter=jitter, height_var=height_var, rng=rng)
        f_opt, C_opt = dinkelbach_improve(f_irreg, n_outer=3, n_inner=2000, beta=1e6)

        if C_opt > 0.95:
            f_opt2, C_opt2 = dinkelbach_improve(f_opt, n_outer=3, n_inner=5000, beta=1e8)

            if C_opt2 > C_best and C_opt2 < 1.0 and np.all(np.isfinite(f_opt2)):
                C_best = C_opt2; f_best = f_opt2.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  Trial {trial}, nt={nt}, j={jitter:.2f}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        if trial % 10 == 0:
            print(f"  Trial {trial}: nt={nt}, opt={C_opt:.10f}, best={C_best:.13f} [{time.time()-t0:.0f}s]",
                  flush=True)

        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best and C_shared < 1.0:
                    C_best = C_shared; f_best = f_shared.copy()
        except: pass

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

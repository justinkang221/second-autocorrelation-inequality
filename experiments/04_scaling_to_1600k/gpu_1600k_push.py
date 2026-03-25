#!/usr/bin/env python3
"""
Push 1.6M SOTA further. Start from best_1600k.npy at C≈0.96272.
Try ultra-high betas (100M+) and surgery-style perturbation + Dinkelbach.
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


if __name__ == "__main__":
    print("=" * 70)
    print("Push 1.6M SOTA")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f_best = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")
    sys.stdout.flush()

    save_file = 'best_1600k.npy'
    rng = np.random.default_rng(77)
    n_improvements = 0

    # Phase 1: Continue beta sweep from where we left off (100M+)
    print("\n--- Phase 1: Ultra-high beta sweep ---")
    for beta in [2e8, 5e8, 1e9, 2e9, 5e9]:
        for n_inner in [5000, 10000]:
            f_opt, C_opt = dinkelbach_improve(f_best, n_outer=5, n_inner=n_inner, beta=beta)
            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                C_best = C_opt; f_best = f_opt.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  beta={beta:.0e}, iters={n_inner}: NEW BEST C = {C_best:.13f} !!!")
            print(f"  beta={beta:.0e}, iters={n_inner}: C={C_opt:.13f} [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nAfter beta sweep: C = {C_best:.13f}")

    # Phase 2: Surgery perturbation + Dinkelbach at highest effective beta
    print("\n--- Phase 2: Perturbation + Dinkelbach ---")
    best_beta = 1e9

    for round_num in range(100):
        improved = False

        for intensity in [0.1, 0.01, 0.001, 0.0001]:
            # Simple noise perturbation
            noise = rng.normal(0, np.max(f_best) * intensity * 0.001, size=n)
            f_perturbed = np.maximum(f_best + noise, 0.0)

            f_opt, C_opt = dinkelbach_improve(f_perturbed, n_outer=3, n_inner=5000, beta=best_beta)
            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                C_best = C_opt; f_best = f_opt.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                improved = True
                print(f"  R{round_num} noise int={intensity}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()
                break

        if not improved:
            # Try spectral perturbation
            F = np.fft.rfft(f_best)
            band = max(1, int(len(F) * 0.001))
            start = rng.integers(0, max(1, len(F) - band))
            noise_f = rng.normal(0, np.abs(F[start:start+band]).mean() * 0.001, size=band)
            F[start:start+band] += noise_f
            f_perturbed = np.maximum(np.fft.irfft(F, n=n), 0.0)

            f_opt, C_opt = dinkelbach_improve(f_perturbed, n_outer=3, n_inner=5000, beta=best_beta)
            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                C_best = C_opt; f_best = f_opt.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  R{round_num} spectral: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        if round_num % 5 == 0:
            print(f"  R{round_num}: C_best={C_best:.13f} [{time.time()-t0:.0f}s, {n_improvements} impr]", flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

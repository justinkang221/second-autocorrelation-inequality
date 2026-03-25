#!/usr/bin/env python3
"""
Dinkelbach with ultra-high beta sweep.
Goes: 1M, 2M, 5M, 10M, 20M, 50M, 100M, 500M, 1B
For both 1.6M and 100k solutions.
"""
import numpy as np
import torch
import time
import sys
import os

device = torch.device('cuda')

# Determine which resolution to run based on command line arg
# Usage: python3 gpu_dink_ultrabeta.py [100k|1600k]
mode = sys.argv[1] if len(sys.argv) > 1 else "1600k"

def gpu_score_exact(f_t):
    n = f_t.shape[0]; nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
    h = 1.0/(nc+1); z = torch.zeros(1, device=device, dtype=torch.float64)
    y = torch.cat([z, conv, z]); y0, y1 = y[:-1], y[1:]
    l2sq = (h/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv)/(nc+1); linf = torch.max(conv)
    return (l2sq/(l1*linf)).item()


def dinkelbach_improve(f_np, n_outer=20, n_inner=5000, beta=1e6):
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
        eval_count = [0]

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
            # Stabilized LogSumExp: safe for any beta since exponents are <= 0
            linf_proxy = g_max * torch.exp(
                torch.logsumexp(beta * (conv / (g_max + 1e-18) - 1.0), dim=0) / beta
            )

            obj = l2sq - current_lam * l1 * linf_proxy
            eval_count[0] += 1

            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_C_inner[0]:
                best_C_inner[0] = C_exact
                best_w_inner[0] = w_t.data.clone()

            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    Outer {outer} exception: {e}", flush=True)

        w = best_w_inner[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new

        if C_new > best_C:
            best_C = C_new; best_f = f_new.copy()

        print(f"    Dink outer={outer}: C={C_new:.13f} (evals={eval_count[0]})", flush=True)

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            print(f"    Converged", flush=True)
            break

    return best_f, best_C


def perturb(f_np, intensity, rng):
    f = f_np.copy()
    n = len(f)
    if rng.uniform() > 0.5:
        f = f[::-1].copy()
    noise = rng.uniform(0, intensity * 0.1, size=n)
    f = np.maximum(f + noise, 0.0)
    return f


if __name__ == "__main__":
    print("=" * 70)
    print(f"Dinkelbach ultra-beta optimizer — mode={mode}")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    if mode == "100k":
        # Load best 100k
        try:
            f_best = np.maximum(np.load('best_impevo_100k.npy').astype(np.float64), 0.0)
        except:
            f_best = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)
        save_file = 'best_impevo_100k.npy'
    else:
        f_best = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
        save_file = 'best_1600k.npy'

    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")
    sys.stdout.flush()

    rng = np.random.default_rng(55)
    n_improvements = 0

    # Ultra-high beta sweep
    betas = [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 5e8, 1e9]
    print("\n--- Beta sweep ---")
    for beta in betas:
        for n_inner in [5000, 10000]:
            # Cross-pollinate
            try:
                f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
                if len(f_shared) == n:
                    C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                    if C_shared > C_best:
                        C_best = C_shared; f_best = f_shared.copy()
                        print(f"  Cross-pollinated: C = {C_best:.13f}")
            except: pass

            f_opt, C_opt = dinkelbach_improve(f_best, n_outer=20, n_inner=n_inner, beta=beta)
            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                C_best = C_opt; f_best = f_opt.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  beta={beta:.0e}, iters={n_inner}: NEW BEST C = {C_best:.13f} !!!")
            print(f"  beta={beta:.0e}, iters={n_inner}: C={C_opt:.13f} [{time.time()-t0:.0f}s]", flush=True)

    # Basin-hopping with best beta found
    print("\n--- Basin-hopping with high beta ---")
    best_beta = betas[-1]  # Use highest beta
    for round_num in range(20):
        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best:
                    C_best = C_shared; f_best = f_shared.copy()
                    print(f"  Cross-pollinated: C = {C_best:.13f}")
        except: pass

        for sigma in [0.1, 0.01, 0.001]:
            f_perturbed = perturb(f_best, sigma, rng)
            f_improved, C_improved = dinkelbach_improve(
                f_perturbed, n_outer=10, n_inner=5000, beta=best_beta
            )
            if C_improved > C_best and C_improved < 1.0 and np.all(np.isfinite(f_improved)) and np.max(f_improved) < 1e6:
                C_best = C_improved; f_best = f_improved.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  Round {round_num}, sigma={sigma}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()
            print(f"  Round {round_num}, sigma={sigma:.3f}: C={C_improved:.13f}, "
                  f"best={C_best:.13f} [{time.time()-t0:.0f}s]", flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

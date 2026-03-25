#!/usr/bin/env python3
"""
Dinkelbach + high-beta L-BFGS optimizer for ACI 2.
Combines the Dinkelbach fractional programming iteration with
ImprovEvolve's high-beta LogSumExp and long L-BFGS runs.
"""
import numpy as np
import torch
import time
import sys

device = torch.device('cuda')

def gpu_score_exact(f_t):
    """Exact C(f) score."""
    n = f_t.shape[0]; nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
    h = 1.0/(nc+1); z = torch.zeros(1, device=device, dtype=torch.float64)
    y = torch.cat([z, conv, z]); y0, y1 = y[:-1], y[1:]
    l2sq = (h/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv)/(nc+1); linf = torch.max(conv)
    return (l2sq/(l1*linf)).item()


def dinkelbach_improve(f_np, n_outer=20, n_inner=2000, beta=20000.0):
    """
    Dinkelbach iteration with L-BFGS inner loop.
    Each outer iteration: fix lambda, optimize l2sq - lambda * l1 * linf_proxy
    Then update lambda = l2sq / (l1 * linf).
    """
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w)
    nc = 2*n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    # Initial lambda from current solution
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    lam = gpu_score_exact(f_t)

    best_C = lam
    best_f = f_np.copy()

    for outer in range(n_outer):
        w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)

        current_lam = lam  # capture for closure

        optimizer = torch.optim.LBFGS(
            [w_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-15, tolerance_change=1e-16
        )

        eval_count = [0]
        best_w_inner = [w_t.data.clone()]
        best_obj_inner = [-1e30]

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

            # Dinkelbach objective: maximize l2sq - lambda * l1 * linf
            obj = l2sq - current_lam * l1 * linf_proxy
            eval_count[0] += 1

            # Track best by exact C
            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_obj_inner[0]:
                best_obj_inner[0] = C_exact
                best_w_inner[0] = w_t.data.clone()

            loss = -obj  # minimize negative
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    Outer {outer} exception: {e}", flush=True)

        # Update w from best inner result
        w = best_w_inner[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))

        # Update lambda
        lam = C_new

        if C_new > best_C:
            best_C = C_new
            best_f = f_new.copy()

        print(f"    Dinkelbach outer={outer}: C={C_new:.13f}, lambda={lam:.13f} "
              f"(evals={eval_count[0]})", flush=True)

        # Convergence check: if lambda barely changed
        if outer > 0 and abs(C_new - lam) < 1e-14:
            print(f"    Converged at outer={outer}", flush=True)
            break

    return best_f, best_C


def perturb(f_np, intensity, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    f = f_np.copy()
    n = len(f)
    if rng.uniform() > 0.5:
        f = f[::-1].copy()
    noise = rng.uniform(0, intensity * 0.1, size=n)
    f = np.maximum(f + noise, 0.0)
    return f


if __name__ == "__main__":
    print("=" * 70)
    print("Dinkelbach + high-beta L-BFGS optimizer for ACI 2")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Try to load the best available solution
    f_best = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")
    sys.stdout.flush()

    # Strategy 1: Dinkelbach at full resolution
    print("\n--- Strategy 1: Dinkelbach at full res ---")
    f_opt, C_opt = dinkelbach_improve(f_best, n_outer=20, n_inner=2000, beta=20000.0)
    if C_opt > C_best:
        C_best = C_opt; f_best = f_opt.copy()
        np.save('best_1600k.npy', f_best)
        print(f"  NEW BEST: C = {C_best:.13f} !!!")
    print(f"  After Dinkelbach: C = {C_best:.13f} [{time.time()-t0:.0f}s]")
    sys.stdout.flush()

    # Strategy 2: Dinkelbach with higher beta
    print("\n--- Strategy 2: Dinkelbach with beta=50000 ---")
    f_opt2, C_opt2 = dinkelbach_improve(f_best, n_outer=20, n_inner=2000, beta=50000.0)
    if C_opt2 > C_best:
        C_best = C_opt2; f_best = f_opt2.copy()
        np.save('best_1600k.npy', f_best)
        print(f"  NEW BEST: C = {C_best:.13f} !!!")
    print(f"  After Dinkelbach50k: C = {C_best:.13f} [{time.time()-t0:.0f}s]")
    sys.stdout.flush()

    # Strategy 3: Basin-hopping with Dinkelbach
    print("\n--- Strategy 3: Basin-hopping with Dinkelbach ---")
    rng = np.random.default_rng(123)
    n_improvements = 0

    for round_num in range(10):
        for sigma in [0.1, 0.01, 0.001]:
            # Cross-pollinate
            try:
                f_shared = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
                if len(f_shared) == n:
                    C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                    if C_shared > C_best:
                        C_best = C_shared; f_best = f_shared.copy()
                        print(f"  Cross-pollinated: C = {C_best:.13f}")
            except: pass

            f_perturbed = perturb(f_best, sigma, rng)
            f_improved, C_improved = dinkelbach_improve(
                f_perturbed, n_outer=10, n_inner=2000, beta=20000.0
            )

            if C_improved > C_best:
                C_best = C_improved; f_best = f_improved.copy()
                np.save('best_1600k.npy', f_best)
                n_improvements += 1
                print(f"  Round {round_num}, sigma={sigma}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

            print(f"  Round {round_num}, sigma={sigma:.3f}: C_trial={C_improved:.13f}, "
                  f"C_best={C_best:.13f} [{time.time()-t0:.0f}s]", flush=True)

    np.save('best_1600k.npy', f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

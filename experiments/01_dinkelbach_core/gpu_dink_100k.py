#!/usr/bin/env python3
"""
Dinkelbach + high-beta optimizer for 100k (EinsteinArena submission).
Tries:
1. Dinkelbach with beta sweep on existing 100k solution
2. Downsample 1.6M to 100k via various methods, then Dinkelbach
3. Basin-hopping with Dinkelbach at 100k
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


def dinkelbach_improve(f_np, n_outer=20, n_inner=5000, beta=200000.0):
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
    print("Dinkelbach optimizer for 100k (EinsteinArena)")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Load existing 100k solution
    f_100k = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)
    C_100k = gpu_score_exact(torch.tensor(f_100k, dtype=torch.float64, device=device))
    print(f"Existing 100k: n={len(f_100k)}, C = {C_100k:.13f}")

    # Try downsampling 1.6M solution
    print("\n--- Downsampling 1.6M to 100k ---")
    f_1600k = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n_1600k = len(f_1600k)
    C_1600k = gpu_score_exact(torch.tensor(f_1600k, dtype=torch.float64, device=device))
    print(f"1.6M solution: n={n_1600k}, C = {C_1600k:.13f}")

    # Method 1: Take every 16th point (if 1.6M)
    ratio = n_1600k // 100000
    if ratio * 100000 == n_1600k:
        f_subsample = f_1600k[::ratio].copy()
        C_sub = gpu_score_exact(torch.tensor(f_subsample, dtype=torch.float64, device=device))
        print(f"  Subsample (every {ratio}th): n={len(f_subsample)}, C = {C_sub:.13f}")
    else:
        C_sub = 0.0

    # Method 2: np.interp
    x_old = np.linspace(0, 1, n_1600k)
    x_new = np.linspace(0, 1, 100000)
    f_interp = np.maximum(np.interp(x_new, x_old, f_1600k), 0.0)
    C_interp = gpu_score_exact(torch.tensor(f_interp, dtype=torch.float64, device=device))
    print(f"  Interp: n={len(f_interp)}, C = {C_interp:.13f}")

    # Method 3: Block average (average each block of 16)
    if ratio * 100000 == n_1600k:
        f_block = f_1600k.reshape(100000, ratio).mean(axis=1)
        f_block = np.maximum(f_block, 0.0)
        C_block = gpu_score_exact(torch.tensor(f_block, dtype=torch.float64, device=device))
        print(f"  Block avg ({ratio}): n={len(f_block)}, C = {C_block:.13f}")
    else:
        C_block = 0.0

    # Method 4: Block max
    if ratio * 100000 == n_1600k:
        f_blockmax = f_1600k.reshape(100000, ratio).max(axis=1)
        f_blockmax = np.maximum(f_blockmax, 0.0)
        C_blockmax = gpu_score_exact(torch.tensor(f_blockmax, dtype=torch.float64, device=device))
        print(f"  Block max ({ratio}): n={len(f_blockmax)}, C = {C_blockmax:.13f}")
    else:
        C_blockmax = 0.0

    # Pick best starting point
    candidates = [
        (C_100k, f_100k, "existing 100k"),
        (C_sub, f_subsample if C_sub > 0 else f_100k, "subsample"),
        (C_interp, f_interp, "interp"),
        (C_block, f_block if C_block > 0 else f_100k, "block avg"),
        (C_blockmax, f_blockmax if C_blockmax > 0 else f_100k, "block max"),
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    C_best = candidates[0][0]
    f_best = candidates[0][1].copy()
    print(f"\nBest starting point: {candidates[0][2]}, C = {C_best:.13f}")
    sys.stdout.flush()

    n_improvements = 0
    rng = np.random.default_rng(42)

    # Strategy 1: Dinkelbach beta sweep on best starting point
    print("\n--- Strategy 1: Dinkelbach beta sweep ---")
    for beta in [50000, 100000, 200000, 500000, 1000000]:
        for n_inner in [5000, 10000]:
            f_opt, C_opt = dinkelbach_improve(f_best, n_outer=20, n_inner=n_inner, beta=beta)
            if C_opt > C_best:
                C_best = C_opt; f_best = f_opt.copy()
                np.save('best_impevo_100k.npy', f_best)
                n_improvements += 1
                print(f"  beta={beta}, iters={n_inner}: NEW BEST C = {C_best:.13f} !!!")
            print(f"  beta={beta}, iters={n_inner}: C={C_opt:.13f} [{time.time()-t0:.0f}s]", flush=True)

    # Strategy 2: Dinkelbach on ALL downsampled versions
    print("\n--- Strategy 2: Dinkelbach on each downsample method ---")
    for C_cand, f_cand, name in candidates:
        if C_cand < 0.5:
            continue
        f_opt, C_opt = dinkelbach_improve(f_cand, n_outer=20, n_inner=5000, beta=200000.0)
        if C_opt > C_best:
            C_best = C_opt; f_best = f_opt.copy()
            np.save('best_impevo_100k.npy', f_best)
            n_improvements += 1
            print(f"  {name}: NEW BEST C = {C_best:.13f} !!!")
        print(f"  {name}: start={C_cand:.13f} -> opt={C_opt:.13f} [{time.time()-t0:.0f}s]", flush=True)

    # Strategy 3: Basin-hopping with Dinkelbach
    print("\n--- Strategy 3: Basin-hopping ---")
    for round_num in range(30):
        for sigma in [0.1, 0.05, 0.01, 0.005, 0.001]:
            f_perturbed = perturb(f_best, sigma, rng)
            f_improved, C_improved = dinkelbach_improve(
                f_perturbed, n_outer=10, n_inner=5000, beta=200000.0
            )
            if C_improved > C_best:
                C_best = C_improved; f_best = f_improved.copy()
                np.save('best_impevo_100k.npy', f_best)
                n_improvements += 1
                print(f"  Round {round_num}, sigma={sigma}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()
            print(f"  Round {round_num}, sigma={sigma:.3f}: C={C_improved:.13f}, "
                  f"best={C_best:.13f} [{time.time()-t0:.0f}s]", flush=True)

    np.save('best_impevo_100k.npy', f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

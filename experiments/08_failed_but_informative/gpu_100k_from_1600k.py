#!/usr/bin/env python3
"""
Try to create a better 100k solution from the 1.6M solution.
Multiple approaches:
1. Extract comb structure from 1.6M, rebuild at 100k
2. Fourier downsample (keep low frequencies)
3. Smart block operations
4. Then Dinkelbach optimize each
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


def dinkelbach_improve(f_np, n_outer=10, n_inner=5000, beta=1e8):
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
        except Exception as e:
            print(f"      Exception: {e}", flush=True)

        w = best_w_inner[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new

        if C_new > best_C:
            best_C = C_new; best_f = f_new.copy()

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


if __name__ == "__main__":
    print("=" * 70)
    print("Creating better 100k from 1.6M solution")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Load solutions
    f_1600k = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n_big = len(f_1600k)
    C_1600k = gpu_score_exact(torch.tensor(f_1600k, dtype=torch.float64, device=device))
    print(f"1.6M: n={n_big}, C={C_1600k:.13f}")

    try:
        f_100k_old = np.maximum(np.load('best_impevo_100k.npy').astype(np.float64), 0.0)
    except:
        f_100k_old = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)
    C_100k_old = gpu_score_exact(torch.tensor(f_100k_old, dtype=torch.float64, device=device))
    print(f"Existing 100k: C={C_100k_old:.13f}")

    C_best = C_100k_old
    f_best = f_100k_old.copy()

    ratio = n_big // 100000

    # Method 1: Identify tooth positions in 1.6M, rebuild at 100k
    print("\n--- Method 1: Comb reconstruction ---")
    # Find teeth: connected components where f > threshold
    threshold = np.max(f_1600k) * 0.01
    teeth = []
    in_tooth = False
    start = 0
    for i in range(n_big):
        if f_1600k[i] > threshold and not in_tooth:
            in_tooth = True
            start = i
        elif f_1600k[i] <= threshold and in_tooth:
            in_tooth = False
            teeth.append((start, i))
    if in_tooth:
        teeth.append((start, n_big))

    print(f"  Found {len(teeth)} teeth in 1.6M solution")

    # Rebuild: place teeth at corresponding positions in 100k
    f_comb = np.zeros(100000)
    for (s, e) in teeth:
        # Map position to 100k scale
        s100 = int(s * 100000 / n_big)
        e100 = int(e * 100000 / n_big)
        if e100 <= s100:
            e100 = s100 + 1
        # Copy tooth shape (subsample)
        tooth = f_1600k[s:e]
        tooth_100k = np.interp(
            np.linspace(0, 1, e100 - s100),
            np.linspace(0, 1, len(tooth)),
            tooth
        )
        f_comb[s100:e100] = tooth_100k

    C_comb = gpu_score_exact(torch.tensor(f_comb, dtype=torch.float64, device=device))
    print(f"  Comb reconstruction: C={C_comb:.13f}")

    # Optimize it
    f_opt, C_opt = dinkelbach_improve(f_comb, n_outer=10, n_inner=10000, beta=1e8)
    print(f"  After Dinkelbach: C={C_opt:.13f} [{time.time()-t0:.0f}s]")
    if C_opt > C_best:
        C_best = C_opt; f_best = f_opt.copy()
        np.save('best_impevo_100k.npy', f_best)
        print(f"  NEW BEST 100k: C={C_best:.13f} !!!")

    # Method 2: Fourier downsample
    print("\n--- Method 2: Fourier downsample ---")
    F_big = np.fft.rfft(f_1600k)
    # Keep first 50001 coefficients (for 100k signal)
    F_small = np.zeros(50001, dtype=complex)
    F_small[:50001] = F_big[:50001]
    f_fourier = np.fft.irfft(F_small, n=100000)
    f_fourier = np.maximum(f_fourier, 0.0)
    # Scale to match
    f_fourier = f_fourier * (np.sum(f_1600k) / (np.sum(f_fourier) + 1e-18))
    C_fourier = gpu_score_exact(torch.tensor(f_fourier, dtype=torch.float64, device=device))
    print(f"  Fourier downsample: C={C_fourier:.13f}")

    f_opt2, C_opt2 = dinkelbach_improve(f_fourier, n_outer=10, n_inner=10000, beta=1e8)
    print(f"  After Dinkelbach: C={C_opt2:.13f} [{time.time()-t0:.0f}s]")
    if C_opt2 > C_best:
        C_best = C_opt2; f_best = f_opt2.copy()
        np.save('best_impevo_100k.npy', f_best)
        print(f"  NEW BEST 100k: C={C_best:.13f} !!!")

    # Method 3: Block sum (preserve mass)
    print("\n--- Method 3: Block sum ---")
    f_blocksum = f_1600k.reshape(100000, ratio).sum(axis=1)
    f_blocksum = np.maximum(f_blocksum, 0.0)
    C_blocksum = gpu_score_exact(torch.tensor(f_blocksum, dtype=torch.float64, device=device))
    print(f"  Block sum: C={C_blocksum:.13f}")

    f_opt3, C_opt3 = dinkelbach_improve(f_blocksum, n_outer=10, n_inner=10000, beta=1e8)
    print(f"  After Dinkelbach: C={C_opt3:.13f} [{time.time()-t0:.0f}s]")
    if C_opt3 > C_best:
        C_best = C_opt3; f_best = f_opt3.copy()
        np.save('best_impevo_100k.npy', f_best)
        print(f"  NEW BEST 100k: C={C_best:.13f} !!!")

    # Method 4: Median filter downsample
    print("\n--- Method 4: Block median ---")
    f_blockmed = np.median(f_1600k.reshape(100000, ratio), axis=1)
    f_blockmed = np.maximum(f_blockmed, 0.0)
    C_blockmed = gpu_score_exact(torch.tensor(f_blockmed, dtype=torch.float64, device=device))
    print(f"  Block median: C={C_blockmed:.13f}")

    f_opt4, C_opt4 = dinkelbach_improve(f_blockmed, n_outer=10, n_inner=10000, beta=1e8)
    print(f"  After Dinkelbach: C={C_opt4:.13f} [{time.time()-t0:.0f}s]")
    if C_opt4 > C_best:
        C_best = C_opt4; f_best = f_opt4.copy()
        np.save('best_impevo_100k.npy', f_best)
        print(f"  NEW BEST 100k: C={C_best:.13f} !!!")

    # Method 5: Start from existing best and do progressive with Dinkelbach
    print("\n--- Method 5: Progressive downsample+optimize ---")
    # Go 1.6M -> 800k -> 400k -> 200k -> 100k, optimizing at each step
    f_prog = f_1600k.copy()
    resolutions = [800000, 400000, 200000, 100000]
    for target_n in resolutions:
        # Downsample by 2x (block avg of pairs)
        current_n = len(f_prog)
        if current_n != target_n:
            r = current_n // target_n
            f_prog = f_prog.reshape(target_n, r).mean(axis=1)
            f_prog = np.maximum(f_prog, 0.0)

        C_pre = gpu_score_exact(torch.tensor(f_prog, dtype=torch.float64, device=device))
        f_prog, C_post = dinkelbach_improve(f_prog, n_outer=10, n_inner=5000, beta=1e8)
        print(f"  n={target_n}: C_pre={C_pre:.13f}, C_post={C_post:.13f} [{time.time()-t0:.0f}s]", flush=True)

    if C_post > C_best:
        C_best = C_post; f_best = f_prog.copy()
        np.save('best_impevo_100k.npy', f_best)
        print(f"  NEW BEST 100k: C={C_best:.13f} !!!")

    np.save('best_impevo_100k.npy', f_best)
    print(f"\nFINAL 100k: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s")

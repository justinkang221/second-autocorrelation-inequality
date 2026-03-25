#!/usr/bin/env python3
"""
Surgery perturbation + Dinkelbach high-beta polish for 100k.
Uses our surgery v2b approach for targeted structural perturbation,
then Dinkelbach + high beta for optimization.
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
            pass

        w = best_w_inner[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new

        if C_new > best_C:
            best_C = C_new; best_f = f_new.copy()

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def surgery_perturb(f_np, rng, intensity=1.0):
    """Surgery v2b: targeted perturbation based on autoconvolution structure."""
    f = f_np.copy()
    n = len(f)

    # Compute autoconvolution to find dips
    f_t = torch.tensor(f, dtype=torch.float64, device=device)
    nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F*F, n=nfft)[:nc].cpu().numpy()

    g_max = np.max(conv)
    plateau = conv / (g_max + 1e-18)

    # Find dips in the plateau (positions where conv is below threshold)
    threshold = 1.0 - 0.001 * intensity
    dip_mask = plateau < threshold
    dip_positions = np.where(dip_mask)[0]

    if len(dip_positions) == 0:
        # No dips, just add noise
        noise = rng.uniform(0, intensity * 0.001, size=n)
        f = np.maximum(f + noise, 0.0)
        return f

    # Pick a random action
    action = rng.choice(['expand', 'shift', 'scale', 'add_tooth', 'noise'])

    if action == 'expand':
        # Expand teeth near a dip
        dip_idx = rng.choice(dip_positions)
        # Map dip in conv space back to f space
        f_idx = dip_idx // 2  # approximate
        f_idx = min(max(f_idx, 0), n-1)
        # Find nearest tooth
        radius = int(n * 0.01 * intensity)
        lo = max(0, f_idx - radius)
        hi = min(n, f_idx + radius)
        # Scale up this region
        scale = 1.0 + rng.uniform(0.001, 0.01) * intensity
        f[lo:hi] *= scale

    elif action == 'shift':
        # Shift a tooth slightly
        # Find tooth centers (local maxima)
        from scipy.signal import argrelmax
        peaks = argrelmax(f, order=max(1, n//10000))[0]
        if len(peaks) > 0:
            peak = rng.choice(peaks)
            shift = rng.integers(-max(1, int(n*0.0001*intensity)),
                                  max(2, int(n*0.0001*intensity)+1))
            # Extract tooth region
            radius = max(1, n // 10000)
            lo = max(0, peak - radius)
            hi = min(n, peak + radius)
            tooth = f[lo:hi].copy()
            f[lo:hi] = 0
            new_lo = max(0, lo + shift)
            new_hi = min(n, new_lo + len(tooth))
            if new_hi > new_lo:
                f[new_lo:new_hi] = tooth[:new_hi-new_lo]

    elif action == 'scale':
        # Scale a random tooth
        from scipy.signal import argrelmax
        peaks = argrelmax(f, order=max(1, n//10000))[0]
        if len(peaks) > 0:
            peak = rng.choice(peaks)
            radius = max(1, n // 10000)
            lo = max(0, peak - radius)
            hi = min(n, peak + radius)
            scale = 1.0 + rng.normal(0, 0.01 * intensity)
            f[lo:hi] *= max(0.5, scale)

    elif action == 'add_tooth':
        # Add a small tooth at a gap
        # Find positions with f near zero
        zero_mask = f < np.max(f) * 0.001
        zero_pos = np.where(zero_mask)[0]
        if len(zero_pos) > 0:
            pos = rng.choice(zero_pos)
            width = max(1, int(n * 0.0001))
            height = np.max(f) * rng.uniform(0.01, 0.1) * intensity
            lo = max(0, pos - width//2)
            hi = min(n, pos + width//2 + 1)
            f[lo:hi] = height

    else:  # noise
        noise = rng.normal(0, np.max(f) * 0.001 * intensity, size=n)
        f = np.maximum(f + noise, 0.0)

    f = np.maximum(f, 0.0)
    return f


if __name__ == "__main__":
    print("=" * 70)
    print("Surgery + Dinkelbach optimizer for 100k")
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

    rng = np.random.default_rng(42)
    n_improvements = 0

    # Main loop: surgery perturbation + Dinkelbach polish
    for round_num in range(200):
        for intensity in [5.0, 2.0, 1.0, 0.5, 0.1]:
            f_perturbed = surgery_perturb(f_best, rng, intensity)
            f_improved, C_improved = dinkelbach_improve(
                f_perturbed, n_outer=10, n_inner=5000, beta=1e8
            )

            if C_improved > C_best and C_improved < 1.0 and np.all(np.isfinite(f_improved)) and np.max(f_improved) < 1e6:
                C_best = C_improved; f_best = f_improved.copy()
                np.save('best_impevo_100k.npy', f_best)
                n_improvements += 1
                print(f"  Round {round_num}, int={intensity}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

            if round_num % 10 == 0 and intensity == 5.0:
                elapsed = time.time() - t0
                print(f"  Round {round_num}: C_best={C_best:.13f} [{elapsed:.0f}s, {n_improvements} impr]",
                      flush=True)

    np.save('best_impevo_100k.npy', f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

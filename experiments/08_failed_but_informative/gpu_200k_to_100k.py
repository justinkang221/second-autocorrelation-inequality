#!/usr/bin/env python3
"""
200k→100k transfer with optimization-based decimation.

Strategy:
1. Continue optimizing the 200k solution with surgery+dk_iter (it was at 0.96210)
2. Try multiple decimation methods to get to 100k
3. Run heavy dk_iter from each decimated start
4. The 200k-informed starting point may reach a different (higher) basin than
   our current 100k solution

Key decimation methods:
A. Every-other-point: f_100k[i] = f_200k[2*i]
B. Averaged: f_100k[i] = (f_200k[2*i] + f_200k[2*i+1]) / sqrt(2)
C. Max-pooled: f_100k[i] = max(f_200k[2*i], f_200k[2*i+1])
D. Structure-aware: identify comb teeth in 200k, reconstruct at 100k
E. Optimization-based: minimize ||conv_100k - downsample(conv_200k)||

Then: heavy surgery+dk_iter from the best decimated solution.
"""
import numpy as np
import torch
import time
import sys

device = torch.device('cuda')

def score_exact_np(f):
    conv = np.convolve(f, f)
    nc = len(conv)
    h = 1.0 / (nc + 1)
    y = np.concatenate([[0], conv, [0]])
    l2sq = (h / 3.0) * np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
    l1 = np.sum(conv) / (nc + 1)
    linf = np.max(conv)
    return l2sq / (l1 * linf)

def gpu_score_exact(f_t):
    n = f_t.shape[0]
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F * F, n=nfft)[:nc]
    h = 1.0 / (nc + 1)
    z = torch.zeros(1, device=device, dtype=torch.float64)
    y = torch.cat([z, conv, z])
    y0, y1 = y[:-1], y[1:]
    l2sq = (h / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv) / (nc + 1)
    linf = torch.max(conv)
    return (l2sq / (l1 * linf)).item()

def gpu_softplus_lbfgs(f_np, maxiter=150, expand_radius=500, soft_temp=0.001):
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    support = torch.nonzero(f_t > 1e-12).squeeze()
    opt_mask = torch.zeros(n, dtype=torch.bool, device=device)
    opt_mask[support] = True
    for r in range(1, expand_radius + 1):
        opt_mask[torch.clamp(support + r, 0, n-1)] = True
        opt_mask[torch.clamp(support - r, 0, n-1)] = True
    indices = torch.nonzero(opt_mask).squeeze()
    f_vals = f_t[indices].clone()
    f_clamped = torch.clamp(f_vals, min=1e-10, max=50.0)
    h_init = torch.log(torch.exp(f_clamped) - 1.0 + 1e-30)
    h_init[f_vals < 1e-10] = -20.0
    h_param = h_init.detach().requires_grad_(True)
    best_C = [0.0]
    best_h = [h_param.data.clone()]
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
    def closure():
        optimizer.zero_grad()
        f = torch.zeros(n, device=device, dtype=torch.float64)
        f[indices] = torch.nn.functional.softplus(h_param)
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        hh = 1.0 / (nc + 1)
        z = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([z, conv, z])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv) / (nc + 1)
        linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item() / (l1.item() * linf_exact)
        if C_exact > best_C[0]:
            best_C[0] = C_exact
            best_h[0] = h_param.data.clone()
        loss = -l2sq / (l1 * linf_soft)
        loss.backward()
        return loss
    try:
        optimizer.step(closure)
    except:
        pass
    f_out = np.zeros(n, dtype=np.float64)
    f_out[indices.cpu().numpy()] = torch.nn.functional.softplus(best_h[0]).cpu().numpy()
    return f_out, best_C[0]

def gpu_lbfgsb(f_np, maxiter=500, soft_temp=0.001):
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    support = f_t > 1e-12
    indices = torch.nonzero(support).squeeze()
    if indices.numel() == 0:
        return f_np, 0.0
    h_param = torch.sqrt(f_t[indices]).detach().requires_grad_(True)
    best_C = [0.0]
    best_h = [h_param.data.clone()]
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
    def closure():
        optimizer.zero_grad()
        f = torch.zeros(n, device=device, dtype=torch.float64)
        f[indices] = h_param ** 2
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        hh = 1.0 / (nc + 1)
        z = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([z, conv, z])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv) / (nc + 1)
        linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item() / (l1.item() * linf_exact)
        if C_exact > best_C[0]:
            best_C[0] = C_exact
            best_h[0] = h_param.data.clone()
        loss = -l2sq / (l1 * linf_soft)
        loss.backward()
        return loss
    try:
        optimizer.step(closure)
    except:
        pass
    f_out = np.zeros(n, dtype=np.float64)
    f_out[indices.cpu().numpy()] = (best_h[0] ** 2).cpu().numpy()
    return f_out, best_C[0]

def gpu_dinkelbach(f_np, n_iters=10, maxiter_inner=150, soft_temp=0.001):
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    C_current = gpu_score_exact(f_t)
    best_C = C_current
    best_f = f_np.copy()
    for dk_iter in range(n_iters):
        lam = C_current
        support = f_t > 1e-12
        indices = torch.nonzero(support).squeeze()
        if indices.numel() == 0: break
        h_param = torch.sqrt(f_t[indices]).detach().requires_grad_(True)
        best_h_dk = [h_param.data.clone()]
        best_C_dk = [0.0]
        optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter_inner,
            line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
        def closure():
            optimizer.zero_grad()
            f = torch.zeros(n, device=device, dtype=torch.float64)
            f[indices] = h_param ** 2
            F = torch.fft.rfft(f, n=nfft)
            conv = torch.fft.irfft(F * F, n=nfft)[:nc]
            hh = 1.0 / (nc + 1)
            z = torch.zeros(1, device=device, dtype=torch.float64)
            y = torch.cat([z, conv, z])
            y0, y1 = y[:-1], y[1:]
            l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
            l1 = torch.sum(conv) / (nc + 1)
            linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
            linf_exact = torch.max(conv).item()
            C_exact = l2sq.item() / (l1.item() * linf_exact)
            if C_exact > best_C_dk[0]:
                best_C_dk[0] = C_exact
                best_h_dk[0] = h_param.data.clone()
            loss = -(l2sq - lam * l1 * linf_soft)
            loss.backward()
            return loss
        try:
            optimizer.step(closure)
        except:
            pass
        f_new = np.zeros(n, dtype=np.float64)
        f_new[indices.cpu().numpy()] = (best_h_dk[0] ** 2).cpu().numpy()
        f_t = torch.tensor(f_new, dtype=torch.float64, device=device)
        C_new = gpu_score_exact(f_t)
        if C_new > best_C:
            best_C = C_new
            best_f = f_new.copy()
        C_current = C_new
    return best_f, best_C

def full_dk_cycle(f, expand_radius, soft_temp):
    f_sp, C_sp = gpu_softplus_lbfgs(f, maxiter=150, expand_radius=expand_radius, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C_sp > C_cur:
        f = f_sp
    f1, C1 = gpu_lbfgsb(f, maxiter=500, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C1 > C_cur:
        f = f1
    f2, C2 = gpu_dinkelbach(f, n_iters=10, maxiter_inner=150, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C2 > C_cur:
        f = f2
    C_final = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    return f, C_final

def surgery_round(f_best, n_dips=50, scale_factor=0.001):
    n = len(f_best)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_best, dtype=torch.float64, device=device)
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F * F, n=nfft)[:nc]
    conv_np = conv.cpu().numpy()
    max_conv = np.max(conv_np)
    plateau = conv_np > 0.5 * max_conv
    plateau_indices = np.where(plateau)[0]
    if len(plateau_indices) == 0:
        return f_best
    plateau_vals = conv_np[plateau_indices]
    deviation = max_conv - plateau_vals
    worst_indices = plateau_indices[np.argsort(-deviation)[:min(200, len(deviation))][:n_dips]]
    f_new = f_best.copy()
    for conv_idx in worst_indices:
        best_contribution = 0
        best_pos = -1
        for i in range(max(0, conv_idx - n + 1), min(n, conv_idx + 1)):
            j = conv_idx - i
            if 0 <= j < n:
                contrib = f_best[i] * f_best[j]
                if contrib > best_contribution:
                    best_contribution = contrib
                    best_pos = i
        if best_pos >= 0:
            j = conv_idx - best_pos
            dev_ratio = deviation[np.where(plateau_indices == conv_idx)[0][0]] / max_conv
            scale = 1.0 + scale_factor * dev_ratio
            f_new[best_pos] *= scale
            if best_pos != j and 0 <= j < n:
                f_new[j] *= scale
    return np.maximum(f_new, 0.0)


# =========================================================================
# Decimation methods
# =========================================================================
def decimate_everyother(f_200k):
    """f_100k[i] = f_200k[2*i]"""
    return f_200k[::2].copy()

def decimate_averaged(f_200k):
    """f_100k[i] = (f_200k[2*i] + f_200k[2*i+1]) / sqrt(2)"""
    n = len(f_200k) // 2
    return (f_200k[::2][:n] + f_200k[1::2][:n]) / np.sqrt(2)

def decimate_maxpool(f_200k):
    """f_100k[i] = max(f_200k[2*i], f_200k[2*i+1])"""
    n = len(f_200k) // 2
    return np.maximum(f_200k[::2][:n], f_200k[1::2][:n])

def decimate_energy(f_200k):
    """f_100k[i] = sqrt(f_200k[2*i]^2 + f_200k[2*i+1]^2)"""
    n = len(f_200k) // 2
    return np.sqrt(f_200k[::2][:n]**2 + f_200k[1::2][:n]**2)

def decimate_interp(f_200k, n_target=100000):
    """Linear interpolation to n_target points."""
    x_old = np.linspace(0, 1, len(f_200k))
    x_new = np.linspace(0, 1, n_target)
    return np.maximum(np.interp(x_new, x_old, f_200k), 0.0)

def decimate_spectral(f_200k, n_target=100000):
    """Keep low-frequency Fourier components."""
    F = np.fft.rfft(f_200k)
    n_keep = n_target // 2 + 1
    F_new = np.zeros(n_keep, dtype=complex)
    F_new[:min(n_keep, len(F))] = F[:min(n_keep, len(F))]
    f_new = np.fft.irfft(F_new, n=n_target) * (n_target / len(f_200k))
    return np.maximum(f_new, 0.0)


if __name__ == "__main__":
    print("=" * 70)
    print("200k → 100k Transfer with Heavy Optimization")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Load 200k solution
    f_200k = np.maximum(np.load('best_200k.npy').astype(np.float64), 0.0)
    C_200k = gpu_score_exact(torch.tensor(f_200k, dtype=torch.float64, device=device))
    print(f"200k solution: C = {C_200k:.12f}")

    # Load current 100k best
    f_100k_best = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
    C_100k_best = gpu_score_exact(torch.tensor(f_100k_best, dtype=torch.float64, device=device))
    print(f"Current 100k best: C = {C_100k_best:.12f}")
    sys.stdout.flush()

    # Try all decimation methods
    methods = {
        'everyother': decimate_everyother,
        'averaged': decimate_averaged,
        'maxpool': decimate_maxpool,
        'energy': decimate_energy,
        'interp': decimate_interp,
        'spectral': decimate_spectral,
    }

    best_method = None
    best_C_dec = 0.0
    best_f_dec = None

    for name, method in methods.items():
        f_dec = method(f_200k)
        if len(f_dec) != 100000:
            f_dec = f_dec[:100000]
        C_dec = gpu_score_exact(torch.tensor(f_dec, dtype=torch.float64, device=device))
        print(f"  {name}: C = {C_dec:.8f}")
        if C_dec > best_C_dec:
            best_C_dec = C_dec
            best_f_dec = f_dec.copy()
            best_method = name
    print(f"\nBest decimation: {best_method} (C = {best_C_dec:.8f})")
    sys.stdout.flush()

    # Also try blending: α * best_100k + (1-α) * best_decimation
    print(f"\nBlending current 100k best with decimated 200k...")
    for alpha in [0.9, 0.95, 0.99, 0.5, 0.7, 0.3]:
        f_blend = alpha * f_100k_best + (1 - alpha) * best_f_dec
        f_blend = np.maximum(f_blend, 0.0)
        C_blend = gpu_score_exact(torch.tensor(f_blend, dtype=torch.float64, device=device))
        print(f"  alpha={alpha:.2f}: C = {C_blend:.8f}")
        if C_blend > best_C_dec:
            best_C_dec = C_blend
            best_f_dec = f_blend.copy()
            best_method = f"blend_{alpha}"
    print(f"\nBest start: {best_method} (C = {best_C_dec:.8f})")
    sys.stdout.flush()

    # Now heavy optimization from the best starting point
    print(f"\n{'='*60}")
    print(f"Phase 2: Heavy optimization from {best_method}")
    print(f"{'='*60}")

    f_work = best_f_dec.copy()
    C_work_best = best_C_dec
    expand_schedule = [200, 500, 1000, 1500, 2000, 300, 750, 1200]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    scale_schedule = [0.001, 0.002, 0.005, 0.0005, 0.003]

    # Run many rounds of surgery + dk_iter from the transferred solution
    for round_num in range(100):
        # Cross-pollination
        try:
            f_shared = np.load('best_dinkelbach_iter.npy')
            C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
            if C_shared > C_work_best:
                C_work_best = C_shared
                f_work = f_shared.copy()
                print(f"  Loaded better shared solution: C = {C_work_best:.12f}")
        except:
            pass

        # Surgery
        scale = scale_schedule[round_num % len(scale_schedule)]
        n_dips = [30, 50, 80, 100, 150][round_num % 5]
        f_new = surgery_round(f_work, n_dips=n_dips, scale_factor=scale)

        # dk_iter cycles
        improved = False
        for cycle in range(10):
            temp = temps[cycle % len(temps)]
            expand = expand_schedule[cycle % len(expand_schedule)]
            f_new, C_cycle = full_dk_cycle(f_new, expand, temp)
            if C_cycle > C_work_best:
                C_work_best = C_cycle
                f_work = f_new.copy()
                # Only save to shared if we beat the real best
                if C_work_best > C_100k_best:
                    np.save('best_dinkelbach_iter.npy', f_work)
                    C_100k_best = C_work_best
                np.save('best_transfer_100k.npy', f_work)
                improved = True
                print(f"  Round {round_num}, cycle {cycle}: "
                      f"NEW BEST C = {C_work_best:.12f} !!!")

        if round_num % 5 == 4 or not improved:
            elapsed = time.time() - t0
            print(f"  Round {round_num}: C_best = {C_work_best:.12f} "
                  f"(improved={improved}) [{elapsed:.0f}s]")
        sys.stdout.flush()

    np.save('best_transfer_100k.npy', f_work)
    print(f"\nFINAL: C = {C_work_best:.12f}")
    print(f"Total time: {time.time() - t0:.0f}s")

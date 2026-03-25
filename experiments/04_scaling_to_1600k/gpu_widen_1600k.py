#!/usr/bin/env python3
"""
Tooth-widening experiment for n=1600k.
Strategy: take current comb solution, widen each tooth by convolving
with a small kernel, then re-optimize.
Wider teeth -> smoother autoconvolution -> potentially higher C.
Also tries Gaussian smoothing of the function.
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

def gpu_lbfgsb(f_np, maxiter=500, soft_temp=0.001):
    n = len(f_np); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    indices = torch.nonzero(f_t > 1e-12).squeeze()
    if indices.numel() == 0: return f_np, 0.0
    h_param = torch.sqrt(f_t[indices]).detach().requires_grad_(True)
    best_C = [0.0]; best_h = [h_param.data.clone()]
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
    def closure():
        optimizer.zero_grad()
        f = torch.zeros(n, device=device, dtype=torch.float64)
        f[indices] = h_param**2
        F = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
        hh = 1.0/(nc+1); zz = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([zz, conv, zz]); y0, y1 = y[:-1], y[1:]
        l2sq = (hh/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv)/(nc+1)
        linf_soft = soft_temp*torch.logsumexp(conv/soft_temp, dim=0)
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item()/(l1.item()*linf_exact)
        if C_exact > best_C[0]: best_C[0] = C_exact; best_h[0] = h_param.data.clone()
        loss = -l2sq/(l1*linf_soft); loss.backward(); return loss
    try: optimizer.step(closure)
    except: pass
    f_out = np.zeros(n, dtype=np.float64)
    f_out[indices.cpu().numpy()] = (best_h[0]**2).cpu().numpy()
    return f_out, best_C[0]

def gpu_softplus_lbfgs(f_np, maxiter=150, expand_radius=500, soft_temp=0.001):
    n = len(f_np); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    support = torch.nonzero(f_t > 1e-12).squeeze()
    if support.numel() == 0: return f_np, 0.0
    opt_mask = torch.zeros(n, dtype=torch.bool, device=device)
    opt_mask[support] = True
    for r in range(1, expand_radius+1):
        opt_mask[torch.clamp(support+r, 0, n-1)] = True
        opt_mask[torch.clamp(support-r, 0, n-1)] = True
    indices = torch.nonzero(opt_mask).squeeze()
    f_vals = f_t[indices].clone()
    f_clamped = torch.clamp(f_vals, min=1e-10, max=50.0)
    h_init = torch.log(torch.exp(f_clamped) - 1.0 + 1e-30)
    h_init[f_vals < 1e-10] = -20.0
    h_param = h_init.detach().requires_grad_(True)
    best_C = [0.0]; best_h = [h_param.data.clone()]
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
    def closure():
        optimizer.zero_grad()
        f = torch.zeros(n, device=device, dtype=torch.float64)
        f[indices] = torch.nn.functional.softplus(h_param)
        F = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
        hh = 1.0/(nc+1); zz = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([zz, conv, zz]); y0, y1 = y[:-1], y[1:]
        l2sq = (hh/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv)/(nc+1)
        linf_soft = soft_temp*torch.logsumexp(conv/soft_temp, dim=0)
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item()/(l1.item()*linf_exact)
        if C_exact > best_C[0]: best_C[0] = C_exact; best_h[0] = h_param.data.clone()
        loss = -l2sq/(l1*linf_soft); loss.backward(); return loss
    try: optimizer.step(closure)
    except: pass
    f_out = np.zeros(n, dtype=np.float64)
    f_out[indices.cpu().numpy()] = torch.nn.functional.softplus(best_h[0]).cpu().numpy()
    return f_out, best_C[0]

def gpu_dinkelbach(f_np, n_iters=5, maxiter_inner=150, soft_temp=0.001):
    n = len(f_np); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    C_current = gpu_score_exact(f_t); best_C = C_current; best_f = f_np.copy()
    for dk in range(n_iters):
        lam = C_current
        indices = torch.nonzero(f_t > 1e-12).squeeze()
        if indices.numel() == 0: break
        h_param = torch.sqrt(f_t[indices]).detach().requires_grad_(True)
        best_h_dk = [h_param.data.clone()]; best_C_dk = [0.0]
        optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter_inner,
            line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
        def closure():
            optimizer.zero_grad()
            f = torch.zeros(n, device=device, dtype=torch.float64)
            f[indices] = h_param**2
            F = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
            hh = 1.0/(nc+1); zz = torch.zeros(1, device=device, dtype=torch.float64)
            y = torch.cat([zz, conv, zz]); y0, y1 = y[:-1], y[1:]
            l2sq = (hh/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
            l1 = torch.sum(conv)/(nc+1)
            linf_soft = soft_temp*torch.logsumexp(conv/soft_temp, dim=0)
            linf_exact = torch.max(conv).item()
            C_exact = l2sq.item()/(l1.item()*linf_exact)
            if C_exact > best_C_dk[0]: best_C_dk[0] = C_exact; best_h_dk[0] = h_param.data.clone()
            loss = -(l2sq - lam*l1*linf_soft); loss.backward(); return loss
        try: optimizer.step(closure)
        except: pass
        f_new = np.zeros(n, dtype=np.float64)
        f_new[indices.cpu().numpy()] = (best_h_dk[0]**2).cpu().numpy()
        f_t = torch.tensor(f_new, dtype=torch.float64, device=device)
        C_new = gpu_score_exact(f_t)
        if C_new > best_C: best_C = C_new; best_f = f_new.copy()
        C_current = C_new
    return best_f, best_C

def full_dk_cycle(f, expand_radius, soft_temp):
    f_sp, C_sp = gpu_softplus_lbfgs(f, maxiter=150, expand_radius=expand_radius, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C_sp > C_cur: f = f_sp
    f1, C1 = gpu_lbfgsb(f, maxiter=500, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C1 > C_cur: f = f1
    f2, C2 = gpu_dinkelbach(f, n_iters=5, maxiter_inner=150, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C2 > C_cur: f = f2
    return f, gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))

def surgery_round(f_best, n_dips=50, scale_factor=0.001):
    n = len(f_best); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_best, dtype=torch.float64, device=device)
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F*F, n=nfft)[:nc]
    conv_np = conv.cpu().numpy()
    max_conv = np.max(conv_np)
    plateau = conv_np > 0.5 * max_conv
    plateau_indices = np.where(plateau)[0]
    if len(plateau_indices) == 0: return f_best
    plateau_vals = conv_np[plateau_indices]
    deviation = max_conv - plateau_vals
    worst_k = min(300, len(deviation))
    worst_indices = plateau_indices[np.argsort(-deviation)[:worst_k]]
    f_new = f_best.copy()
    for conv_idx in worst_indices[:n_dips]:
        i_min = max(0, conv_idx - n + 1)
        i_max = min(n, conv_idx + 1)
        i_arr = np.arange(i_min, i_max)
        j_arr = conv_idx - i_arr
        valid = (j_arr >= 0) & (j_arr < n)
        i_arr, j_arr = i_arr[valid], j_arr[valid]
        if len(i_arr) == 0: continue
        contribs = f_best[i_arr] * f_best[j_arr]
        best_idx = np.argmax(contribs)
        best_pos = i_arr[best_idx]; j = j_arr[best_idx]
        dev_ratio = deviation[np.searchsorted(plateau_indices, conv_idx)] / max_conv
        scale = 1.0 + scale_factor * dev_ratio
        f_new[best_pos] *= scale
        if best_pos != j: f_new[j] *= scale
    return np.maximum(f_new, 0.0)


def widen_teeth_fft(f_np, sigma):
    """Widen teeth by convolving with a Gaussian kernel using FFT."""
    n = len(f_np)
    # Create Gaussian kernel
    k_size = int(6 * sigma) + 1
    k_half = k_size // 2
    x = np.arange(-k_half, k_half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # FFT convolution
    nc = n + k_size - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    F_f = np.fft.rfft(f_np, n=nfft)
    F_k = np.fft.rfft(kernel, n=nfft)
    result = np.fft.irfft(F_f * F_k, n=nfft)[:n]
    return np.maximum(result, 0.0)


def interpolate_widen(f_np, factor):
    """Widen by interpolation: stretch the function by factor, then truncate.
    This widens each feature while maintaining relative positions."""
    n = len(f_np)
    # Find center of mass
    nonzero = np.nonzero(f_np > 1e-10)[0]
    if len(nonzero) == 0:
        return f_np
    com = int(np.average(nonzero, weights=f_np[nonzero]))

    # Create widened version by interpolation
    f_new = np.zeros(n, dtype=np.float64)
    for idx in nonzero:
        # Map position relative to center of mass
        new_pos = com + (idx - com) * factor
        # Distribute to nearest grid points
        lo = int(np.floor(new_pos))
        hi = lo + 1
        frac = new_pos - lo
        if 0 <= lo < n:
            f_new[lo] += f_np[idx] * (1 - frac)
        if 0 <= hi < n:
            f_new[hi] += f_np[idx] * frac

    # Scale to preserve L1 norm
    if np.sum(f_new) > 0:
        f_new *= np.sum(f_np) / np.sum(f_new)

    return np.maximum(f_new, 0.0)


if __name__ == "__main__":
    print("=" * 70)
    print("Tooth-widening experiments for n=1600k")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f_orig = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n = len(f_orig)
    C_orig = gpu_score_exact(torch.tensor(f_orig, dtype=torch.float64, device=device))
    print(f"Original: n={n}, C = {C_orig:.13f}")
    sys.stdout.flush()

    C_best = C_orig
    f_best = f_orig.copy()

    expand_schedule = [r*16 for r in [200, 500, 1000, 300, 750]]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001]

    # Experiment 1: Gaussian widening at different scales
    print("\n--- Exp 1: Gaussian widening ---")
    for sigma in [2, 5, 10, 20, 50, 100, 200]:
        f_wide = widen_teeth_fft(f_orig, sigma)
        C_wide = gpu_score_exact(torch.tensor(f_wide, dtype=torch.float64, device=device))
        delta = C_wide - C_orig
        print(f"  sigma={sigma:4d}: C = {C_wide:.10f} (delta={delta:.2e})")

        if C_wide > C_orig * 0.999:  # Don't bother if it drops too much
            f_opt, C_opt = gpu_lbfgsb(f_wide, maxiter=500, soft_temp=0.001)
            f_opt, C_opt2 = gpu_dinkelbach(f_opt, n_iters=5, maxiter_inner=150, soft_temp=0.001)
            C_opt = max(C_opt, C_opt2)
            print(f"    After opt: C = {C_opt:.10f} (vs orig {C_orig:.10f})")

            if C_opt > C_best:
                C_best = C_opt; f_best = f_opt.copy()
                np.save('best_1600k.npy', f_best)
                print(f"    *** NEW BEST: {C_best:.13f} ***")
    sys.stdout.flush()

    # Experiment 2: Interpolation-based widening/narrowing
    print("\n--- Exp 2: Interpolation widening ---")
    for factor in [0.95, 0.9, 0.85, 1.05, 1.1, 1.15]:
        f_interp = interpolate_widen(f_orig, factor)
        C_interp = gpu_score_exact(torch.tensor(f_interp, dtype=torch.float64, device=device))
        delta = C_interp - C_orig
        print(f"  factor={factor:.2f}: C = {C_interp:.10f} (delta={delta:.2e})")

        if C_interp > C_orig * 0.999:
            f_opt, C_opt = gpu_lbfgsb(f_interp, maxiter=500, soft_temp=0.001)
            f_opt, C_opt2 = gpu_dinkelbach(f_opt, n_iters=5, maxiter_inner=150, soft_temp=0.001)
            C_opt = max(C_opt, C_opt2)
            print(f"    After opt: C = {C_opt:.10f}")

            if C_opt > C_best:
                C_best = C_opt; f_best = f_opt.copy()
                np.save('best_1600k.npy', f_best)
                print(f"    *** NEW BEST: {C_best:.13f} ***")
    sys.stdout.flush()

    # Experiment 3: Partial widening (only on teeth that have narrower width)
    # Identify narrow teeth and widen only those
    print("\n--- Exp 3: Selective widening ---")
    nonzero = np.nonzero(f_orig > 1e-10)[0]
    gaps = np.diff(nonzero)
    cluster_breaks = np.where(gaps > 50)[0]
    clusters = []
    start = 0
    for b in cluster_breaks:
        clusters.append(nonzero[start:b+1])
        start = b+1
    clusters.append(nonzero[start:])

    widths = np.array([c[-1]-c[0]+1 for c in clusters])
    median_width = np.median(widths)
    print(f"  Tooth widths: median={median_width:.0f}, min={min(widths)}, max={max(widths)}")

    # Try widening only narrow teeth
    for target_width_mult in [1.5, 2.0, 3.0]:
        target_width = int(median_width * target_width_mult)
        f_sel = f_orig.copy()
        n_widened = 0
        for c in clusters:
            w = c[-1] - c[0] + 1
            if w < target_width:
                # Widen this tooth
                center = int(np.mean(c))
                hw = target_width // 2
                lo = max(0, center - hw)
                hi = min(n, center + hw + 1)
                # Spread the mass of this tooth over the wider range
                tooth_mass = np.sum(f_orig[c])
                f_sel[c] = 0  # Clear old
                f_sel[lo:hi] = tooth_mass / (hi - lo)
                n_widened += 1

        C_sel = gpu_score_exact(torch.tensor(f_sel, dtype=torch.float64, device=device))
        print(f"  target_width={target_width} (x{target_width_mult:.1f}): widened {n_widened} teeth, C = {C_sel:.10f}")

        if C_sel > C_orig * 0.99:
            f_opt, C_opt = gpu_lbfgsb(f_sel, maxiter=500, soft_temp=0.001)
            f_opt, C_opt2 = gpu_dinkelbach(f_opt, n_iters=5, maxiter_inner=150, soft_temp=0.001)
            C_opt = max(C_opt, C_opt2)
            print(f"    After opt: C = {C_opt:.10f}")

            if C_opt > C_best:
                C_best = C_opt; f_best = f_opt.copy()
                np.save('best_1600k.npy', f_best)
                print(f"    *** NEW BEST: {C_best:.13f} ***")
    sys.stdout.flush()

    # Phase 2: Run surgery on the best found
    print(f"\n--- Phase 2: Surgery on best ({C_best:.13f}) ---")
    n_improvements = 0
    for round_num in range(200):
        try:
            f_shared = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best:
                    C_best = C_shared; f_best = f_shared.copy()
                    print(f"  Cross-pollinated: C = {C_best:.13f}")
        except: pass

        sf = [0.001, 0.002, 0.005, 0.0005, 0.003][round_num % 5]
        n_dips = [30, 50, 80, 100, 150][round_num % 5]
        f_work = surgery_round(f_best, n_dips=n_dips, scale_factor=sf)

        for cycle in range(6):
            temp = temps[cycle % len(temps)]
            expand = expand_schedule[cycle % len(expand_schedule)]
            f_work, C_work = full_dk_cycle(f_work, expand, temp)
            if C_work > C_best:
                C_best = C_work; f_best = f_work.copy()
                np.save('best_1600k.npy', f_best)
                n_improvements += 1
                print(f"  Round {round_num}, cycle {cycle}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        if round_num % 10 == 9:
            elapsed = time.time() - t0
            print(f"  Round {round_num}: C = {C_best:.13f} [{elapsed:.0f}s, {n_improvements} impr]")
            sys.stdout.flush()

    np.save('best_1600k.npy', f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s")

#!/usr/bin/env python3
"""
Low-dimensional comb optimization.

The solution has ~595 comb teeth. Instead of optimizing 100k variables,
parameterize f as: f[i] = Σ_k a_k * template_k[i]

where template_k is the shape of tooth k (frozen from current best) and
a_k is the amplitude (optimized). This reduces the problem to ~595 dimensions.

In this low-dimensional space:
1. L-BFGS converges much better (not 100k ill-conditioned variables)
2. CMA-ES or Nelder-Mead become feasible
3. The landscape is much smoother
4. We can also optimize tooth positions (shift each template by ±1)

After finding better amplitudes in comb-space, convert back to full f
and run dk_iter to refine the fine structure.
"""
import numpy as np
import torch
import time
import sys
from scipy.signal import find_peaks

device = torch.device('cuda')

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
    if C_sp > C_cur: f = f_sp
    f1, C1 = gpu_lbfgsb(f, maxiter=500, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C1 > C_cur: f = f1
    f2, C2 = gpu_dinkelbach(f, n_iters=10, maxiter_inner=150, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C2 > C_cur: f = f2
    return f, gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))

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
    if len(plateau_indices) == 0: return f_best
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


def extract_comb_structure(f):
    """Extract comb teeth: find peaks, define neighborhoods, create templates."""
    peaks, props = find_peaks(f, height=np.max(f) * 0.02, distance=20)
    print(f"  Found {len(peaks)} comb teeth")

    # Define tooth neighborhoods: half the distance to nearest neighbor
    tooth_radius = 15  # Each tooth covers ±15 positions
    n = len(f)
    templates = []
    masks = []

    for i, peak in enumerate(peaks):
        lo = max(0, peak - tooth_radius)
        hi = min(n, peak + tooth_radius + 1)
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True
        template = np.zeros(n, dtype=np.float64)
        template[lo:hi] = f[lo:hi]
        # Normalize template so amplitude = peak value
        peak_val = f[peak]
        if peak_val > 1e-12:
            template /= peak_val
        templates.append(template)
        masks.append(mask)

    return peaks, templates, masks


def comb_lbfgs_optimize(f_orig, peaks, templates, maxiter=500, soft_temp=0.001):
    """
    Optimize tooth amplitudes using L-BFGS in the low-dimensional comb space.
    """
    n = len(f_orig)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    K = len(peaks)
    # Initial amplitudes = current peak values
    init_amplitudes = np.array([f_orig[p] for p in peaks])

    # Convert templates to torch (on GPU)
    template_matrix = torch.zeros(K, n, device=device, dtype=torch.float64)
    for k, template in enumerate(templates):
        template_matrix[k] = torch.tensor(template, dtype=torch.float64, device=device)

    # Also keep the residual (parts of f not covered by any tooth)
    covered = np.zeros(n, dtype=bool)
    for template in templates:
        covered |= (template > 1e-12)
    residual = torch.tensor(f_orig * (~covered).astype(float),
                           dtype=torch.float64, device=device)

    # Optimize log-amplitudes (ensures positivity)
    log_a = torch.tensor(np.log(np.maximum(init_amplitudes, 1e-10)),
                        dtype=torch.float64, device=device, requires_grad=True)

    best_C = [0.0]
    best_log_a = [log_a.data.clone()]

    optimizer = torch.optim.LBFGS([log_a], lr=1.0, max_iter=maxiter,
        history_size=50,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)

    def closure():
        optimizer.zero_grad()
        a = torch.exp(log_a)
        # Reconstruct f = Σ a_k * template_k + residual
        f = torch.mv(template_matrix.T, a) + residual
        f = torch.clamp(f, min=0.0)

        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        h = 1.0 / (nc + 1)
        z = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([z, conv, z])
        y0, y1 = y[:-1], y[1:]
        l2sq = (h / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv) / (nc + 1)
        linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item() / (l1.item() * linf_exact)
        if C_exact > best_C[0]:
            best_C[0] = C_exact
            best_log_a[0] = log_a.data.clone()
        loss = -l2sq / (l1 * linf_soft)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except:
        pass

    # Reconstruct f from best amplitudes
    a_best = torch.exp(best_log_a[0])
    f_out = (torch.mv(template_matrix.T, a_best) + residual).cpu().numpy()
    f_out = np.maximum(f_out, 0.0)

    return f_out, best_C[0]


def comb_dinkelbach_optimize(f_orig, peaks, templates, n_outer=5, maxiter=200, soft_temp=0.001):
    """
    Dinkelbach in comb space: linearize the fractional objective,
    optimize tooth amplitudes.
    """
    n = len(f_orig)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    K = len(peaks)
    init_amplitudes = np.array([f_orig[p] for p in peaks])

    template_matrix = torch.zeros(K, n, device=device, dtype=torch.float64)
    for k, template in enumerate(templates):
        template_matrix[k] = torch.tensor(template, dtype=torch.float64, device=device)

    covered = np.zeros(n, dtype=bool)
    for template in templates:
        covered |= (template > 1e-12)
    residual = torch.tensor(f_orig * (~covered).astype(float),
                           dtype=torch.float64, device=device)

    f_t = torch.tensor(f_orig, dtype=torch.float64, device=device)
    C_best = gpu_score_exact(f_t)
    f_best = f_orig.copy()
    best_amplitudes = init_amplitudes.copy()

    for outer in range(n_outer):
        lam = C_best

        log_a = torch.tensor(np.log(np.maximum(best_amplitudes, 1e-10)),
                            dtype=torch.float64, device=device, requires_grad=True)

        best_C_inner = [0.0]
        best_log_a = [log_a.data.clone()]

        optimizer = torch.optim.LBFGS([log_a], lr=1.0, max_iter=maxiter,
            history_size=50,
            line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)

        def closure():
            optimizer.zero_grad()
            a = torch.exp(log_a)
            f = torch.clamp(torch.mv(template_matrix.T, a) + residual, min=0.0)
            F = torch.fft.rfft(f, n=nfft)
            conv = torch.fft.irfft(F * F, n=nfft)[:nc]
            h = 1.0 / (nc + 1)
            z = torch.zeros(1, device=device, dtype=torch.float64)
            y = torch.cat([z, conv, z])
            y0, y1 = y[:-1], y[1:]
            l2sq = (h / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
            l1 = torch.sum(conv) / (nc + 1)
            linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
            linf_exact = torch.max(conv).item()
            C_exact = l2sq.item() / (l1.item() * linf_exact)
            if C_exact > best_C_inner[0]:
                best_C_inner[0] = C_exact
                best_log_a[0] = log_a.data.clone()
            loss = -(l2sq - lam * l1 * linf_soft)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except:
            pass

        a_new = torch.exp(best_log_a[0])
        f_new = torch.clamp(torch.mv(template_matrix.T, a_new) + residual, min=0.0).cpu().numpy()
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))

        if C_new > C_best:
            C_best = C_new
            f_best = f_new.copy()
            best_amplitudes = a_new.cpu().numpy()

    return f_best, C_best


if __name__ == "__main__":
    print("=" * 70)
    print("Low-Dimensional Comb Optimization")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
    n = len(f)
    f_t = torch.tensor(f, dtype=torch.float64, device=device)
    C_best = gpu_score_exact(f_t)
    f_best = f.copy()
    print(f"Starting: C = {C_best:.12f}")
    sys.stdout.flush()

    # Extract comb structure
    print("\nExtracting comb structure...")
    peaks, templates, masks = extract_comb_structure(f)
    print(f"  {len(peaks)} teeth, mean spacing {np.mean(np.diff(peaks)):.1f}")

    # Phase 1: Comb-space L-BFGS
    print(f"\n{'='*60}")
    print("Phase 1: L-BFGS in comb space (amplitude optimization)")
    print(f"{'='*60}")
    sys.stdout.flush()

    for temp in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
        f_comb, C_comb = comb_lbfgs_optimize(f_best, peaks, templates, maxiter=500, soft_temp=temp)
        if C_comb > C_best:
            C_best = C_comb
            f_best = f_comb.copy()
            print(f"  temp={temp}: NEW BEST C = {C_best:.12f} !!!")
        else:
            print(f"  temp={temp}: C = {C_comb:.12f} (no improvement)")
    sys.stdout.flush()

    # Phase 2: Comb-space Dinkelbach
    print(f"\n{'='*60}")
    print("Phase 2: Dinkelbach in comb space")
    print(f"{'='*60}")
    sys.stdout.flush()

    for temp in [0.01, 0.001, 0.0001]:
        f_dk, C_dk = comb_dinkelbach_optimize(f_best, peaks, templates,
                                               n_outer=5, maxiter=300, soft_temp=temp)
        if C_dk > C_best:
            C_best = C_dk
            f_best = f_dk.copy()
            print(f"  temp={temp}: NEW BEST C = {C_best:.12f} !!!")
        else:
            print(f"  temp={temp}: C = {C_dk:.12f} (no improvement)")
    sys.stdout.flush()

    # Phase 3: Full dk_iter to refine fine structure
    print(f"\n{'='*60}")
    print("Phase 3: Full dk_iter refinement")
    print(f"{'='*60}")
    sys.stdout.flush()

    expand_schedule = [200, 500, 1000, 1500, 2000]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    for cycle in range(5):
        f_best, C_cycle = full_dk_cycle(f_best, expand_schedule[cycle], temps[cycle])
        if C_cycle > C_best:
            C_best = C_cycle
        print(f"  Cycle {cycle}: C = {C_best:.12f}")
    sys.stdout.flush()

    # Phase 4: Interleave comb optimization and surgery
    print(f"\n{'='*60}")
    print("Phase 4: Alternating comb-opt + surgery + dk_iter")
    print(f"{'='*60}")
    sys.stdout.flush()

    expand_schedule = [200, 500, 1000, 1500, 2000, 300, 750, 1200]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    scale_schedule = [0.001, 0.002, 0.005, 0.0005, 0.003]

    for round_num in range(100):
        # Cross-pollination
        try:
            f_shared = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
            C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
            if C_shared > C_best:
                C_best = C_shared
                f_best = f_shared.copy()
                # Re-extract comb from updated solution
                peaks, templates, masks = extract_comb_structure(f_best)
                print(f"  Loaded better shared solution: C = {C_best:.12f}")
        except:
            pass

        improved = False

        if round_num % 3 == 0:
            # Comb-space Dinkelbach
            temp = temps[round_num % len(temps)]
            f_new, C_new = comb_dinkelbach_optimize(f_best, peaks, templates,
                                                     n_outer=3, maxiter=200, soft_temp=temp)
            if C_new > C_best:
                C_best = C_new
                f_best = f_new.copy()
                np.save('best_dinkelbach_iter.npy', f_best)
                np.save('best_comb_opt.npy', f_best)
                improved = True
                print(f"  Round {round_num} CombDK: NEW BEST C = {C_best:.12f} !!!")
        else:
            # Surgery + dk_iter (proven approach)
            scale = scale_schedule[round_num % len(scale_schedule)]
            n_dips = [30, 50, 80, 100, 150][round_num % 5]
            f_new = surgery_round(f_best, n_dips=n_dips, scale_factor=scale)

            for cycle in range(6):
                temp = temps[cycle % len(temps)]
                expand = expand_schedule[cycle % len(expand_schedule)]
                f_new, C_work = full_dk_cycle(f_new, expand, temp)
                if C_work > C_best:
                    C_best = C_work
                    f_best = f_new.copy()
                    np.save('best_dinkelbach_iter.npy', f_best)
                    np.save('best_comb_opt.npy', f_best)
                    improved = True
                    print(f"  Round {round_num}, cycle {cycle}: "
                          f"NEW BEST C = {C_best:.12f} !!!")

        # Re-extract comb structure periodically
        if round_num % 10 == 9:
            peaks, templates, masks = extract_comb_structure(f_best)

        if round_num % 5 == 4 or not improved:
            elapsed = time.time() - t0
            print(f"  Round {round_num}: C_best = {C_best:.12f} "
                  f"(improved={improved}) [{elapsed:.0f}s]")
        sys.stdout.flush()

    np.save('best_comb_opt.npy', f_best)
    np.save('best_dinkelbach_iter.npy', f_best)
    print(f"\nFINAL: C = {C_best:.12f}")
    print(f"Total time: {time.time() - t0:.0f}s")

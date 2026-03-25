#!/usr/bin/env python3
"""
Comb Factory: Construct combs from scratch with different spacings and envelopes.

Key insight: the current solution is a comb with spacing ~344 and a graded envelope.
But we've never explored whether different spacings/envelopes could give better C.

This script:
1. Scans over spacings d=100 to d=500
2. For each spacing, tries multiple envelope shapes (Gaussian, flat, cosine, optimized)
3. Evaluates C for each comb
4. Refines the top candidates with dk_iter
5. Cross-pollinates with best_dinkelbach_iter.npy

The fundamental difference from previous approaches: we're constructing solutions
from scratch rather than perturbing the existing one. This lets us explore
entirely different basins.
"""
import numpy as np
import torch
import time
import sys
from itertools import product

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
    if support.numel() == 0:
        return f_np, 0.0
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


def build_comb(n, spacing, envelope_type, envelope_params, tooth_width=15):
    """Construct a comb signal with given spacing and envelope."""
    K = (n - 1) // spacing + 1  # number of teeth
    positions = np.arange(K) * spacing
    # Trim to ensure last tooth fits
    positions = positions[positions < n - tooth_width]
    K = len(positions)

    # Compute amplitudes from envelope
    center = positions[K // 2] if K > 0 else n // 2

    if envelope_type == 'gaussian':
        sigma = envelope_params.get('sigma', K * spacing * 0.25)
        amplitudes = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    elif envelope_type == 'flat':
        amplitudes = np.ones(K)
    elif envelope_type == 'cosine':
        # Raised cosine
        amplitudes = 0.5 * (1 + np.cos(np.pi * (positions - center) / (K * spacing * 0.5 + 1)))
        amplitudes = np.maximum(amplitudes, 0.01)
    elif envelope_type == 'tukey':
        # Tukey window (flat middle, tapered edges)
        alpha = envelope_params.get('alpha', 0.3)
        x = np.linspace(0, 1, K)
        amplitudes = np.ones(K)
        lo = alpha / 2
        hi = 1 - alpha / 2
        mask_lo = x < lo
        mask_hi = x > hi
        amplitudes[mask_lo] = 0.5 * (1 + np.cos(np.pi * (x[mask_lo] / lo - 1)))
        amplitudes[mask_hi] = 0.5 * (1 + np.cos(np.pi * ((x[mask_hi] - hi) / lo)))
    elif envelope_type == 'linear_rise':
        # Linear rise then flat (asymmetric)
        amplitudes = np.minimum(np.linspace(0, 2, K), 1.0)
    elif envelope_type == 'optimal_flat':
        # Try to make self-convolution flat by using sqrt of triangle inverse
        # Triangle conv: b[m] = K - |m| for |m| < K
        # To flatten: a[k] should have DFT |A(ω)|^2 ≈ constant
        # One heuristic: a = IFFT(1/|RECT_DFT|) clipped to non-negative
        # Simpler: use raised cosine that compensates for triangle taper
        k = np.arange(K, dtype=float)
        # Weight inversely proportional to expected overlap count
        overlap = np.minimum(k + 1, K - k)  # overlap count at each position
        # Weight to make self-conv flat: a[k] ~ 1/sqrt(overlap_weight)
        # But this is approximate. Let's try:
        amplitudes = 1.0 / np.sqrt(overlap / K)
        amplitudes /= amplitudes.max()
    elif envelope_type == 'custom':
        amplitudes = envelope_params.get('amplitudes', np.ones(K))[:K]
    else:
        amplitudes = np.ones(K)

    # Scale amplitudes to have reasonable magnitude
    amplitudes *= 30.0 / np.max(amplitudes) if np.max(amplitudes) > 0 else 1.0

    # Build the signal: Gaussian tooth shape at each position
    f = np.zeros(n, dtype=np.float64)
    tooth_sigma = tooth_width / 3.0  # width of each tooth
    for i, pos in enumerate(positions):
        lo = max(0, pos - tooth_width)
        hi = min(n, pos + tooth_width + 1)
        x = np.arange(lo, hi) - pos
        tooth = amplitudes[i] * np.exp(-0.5 * (x / tooth_sigma) ** 2)
        f[lo:hi] += tooth

    return np.maximum(f, 0.0)


def quick_score(f):
    """Score a vector on GPU."""
    f_t = torch.tensor(f, dtype=torch.float64, device=device)
    return gpu_score_exact(f_t)


if __name__ == "__main__":
    print("=" * 70)
    print("Comb Factory: Exploring different comb structures")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Load current best
    f_current = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
    n = len(f_current)
    C_current = quick_score(f_current)
    print(f"Current best: C = {C_current:.12f}")
    sys.stdout.flush()

    # Extract tooth width from current solution
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(f_current, height=np.max(f_current)*0.1, distance=100)
    if len(peaks) > 2:
        current_spacing = int(np.median(np.diff(peaks)))
    else:
        current_spacing = 344
    print(f"Current spacing: {current_spacing}")

    # Phase 1: Quick scan over spacings and envelopes
    print(f"\n{'='*60}")
    print("Phase 1: Quick scan over comb configurations")
    print(f"{'='*60}")
    sys.stdout.flush()

    spacings = [100, 150, 172, 200, 250, 300, 340, 344, 350, 400, 450, 500]
    envelopes = ['gaussian', 'flat', 'cosine', 'tukey', 'optimal_flat']
    tooth_widths = [10, 15, 20]

    results = []
    for d in spacings:
        for env in envelopes:
            for tw in tooth_widths:
                params = {'sigma': n * 0.25} if env == 'gaussian' else {'alpha': 0.3}
                f_test = build_comb(n, d, env, params, tooth_width=tw)
                C_test = quick_score(f_test)
                results.append((C_test, d, env, tw))

    # Sort by C, descending
    results.sort(key=lambda x: -x[0])
    print(f"\nTop 20 raw comb configurations:")
    for i, (C, d, env, tw) in enumerate(results[:20]):
        K = (n - 1) // d + 1
        print(f"  {i+1}. d={d:4d} K~{K:4d} env={env:15s} tw={tw:2d}: C = {C:.10f}")
    sys.stdout.flush()

    # Phase 2: Refine top candidates with dk_iter
    print(f"\n{'='*60}")
    print("Phase 2: Refining top candidates with dk_iter")
    print(f"{'='*60}")
    sys.stdout.flush()

    C_best = C_current
    f_best = f_current.copy()
    top_n = 10  # refine top 10 candidates

    for rank, (C_raw, d, env, tw) in enumerate(results[:top_n]):
        params = {'sigma': n * 0.25} if env == 'gaussian' else {'alpha': 0.3}
        f_cand = build_comb(n, d, env, params, tooth_width=tw)

        print(f"\n  Refining #{rank+1}: d={d} env={env} tw={tw} (raw C={C_raw:.10f})")
        sys.stdout.flush()

        # Quick dk_iter refinement (2 cycles)
        expand_schedule = [500, 1000]
        temps = [0.005, 0.001]
        for cycle in range(2):
            f_cand, C_ref = full_dk_cycle(f_cand, expand_schedule[cycle], temps[cycle])
            print(f"    cycle {cycle}: C = {C_ref:.10f}")

        if C_ref > C_best:
            C_best = C_ref
            f_best = f_cand.copy()
            np.save('best_dinkelbach_iter.npy', f_best)
            np.save('best_comb_factory.npy', f_best)
            print(f"    *** NEW GLOBAL BEST: C = {C_best:.12f} ***")
        sys.stdout.flush()

    print(f"\nAfter Phase 2: C_best = {C_best:.12f}")
    print(f"Time so far: {time.time() - t0:.0f}s")
    sys.stdout.flush()

    # Phase 3: Fine-grained spacing scan around best spacing
    best_spacing = results[0][1]
    print(f"\n{'='*60}")
    print(f"Phase 3: Fine-grained scan around d={best_spacing}")
    print(f"{'='*60}")
    sys.stdout.flush()

    fine_spacings = range(max(50, best_spacing - 30), best_spacing + 31)
    best_env = results[0][2]
    best_tw = results[0][3]

    fine_results = []
    for d in fine_spacings:
        params = {'sigma': n * 0.25} if best_env == 'gaussian' else {'alpha': 0.3}
        f_test = build_comb(n, d, best_env, params, tooth_width=best_tw)
        C_test = quick_score(f_test)
        fine_results.append((C_test, d))

    fine_results.sort(key=lambda x: -x[0])
    print(f"  Top 5 fine spacings:")
    for C, d in fine_results[:5]:
        print(f"    d={d}: C = {C:.10f}")

    # Refine best fine spacing
    best_fine_d = fine_results[0][1]
    params = {'sigma': n * 0.25} if best_env == 'gaussian' else {'alpha': 0.3}
    f_fine = build_comb(n, best_fine_d, best_env, params, tooth_width=best_tw)
    print(f"\n  Full refinement of d={best_fine_d}...")
    sys.stdout.flush()

    expand_schedule = [200, 500, 1000, 1500, 2000]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    for cycle in range(5):
        f_fine, C_fine = full_dk_cycle(f_fine, expand_schedule[cycle], temps[cycle])
        print(f"    cycle {cycle}: C = {C_fine:.10f}")
        if C_fine > C_best:
            C_best = C_fine
            f_best = f_fine.copy()
            np.save('best_dinkelbach_iter.npy', f_best)
            np.save('best_comb_factory.npy', f_best)
            print(f"    *** NEW GLOBAL BEST: C = {C_best:.12f} ***")
    sys.stdout.flush()

    # Phase 4: Blend best factory comb with current solution
    print(f"\n{'='*60}")
    print("Phase 4: Blending factory combs with current best")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Reload current best (may have been updated by other jobs)
    try:
        f_current = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
        C_current = quick_score(f_current)
        if C_current > C_best:
            C_best = C_current
            f_best = f_current.copy()
    except:
        pass

    for rank, (C_raw, d, env, tw) in enumerate(results[:5]):
        params = {'sigma': n * 0.25} if env == 'gaussian' else {'alpha': 0.3}
        f_cand = build_comb(n, d, env, params, tooth_width=tw)

        for alpha in [0.01, 0.05, 0.1, 0.2]:
            f_blend = (1 - alpha) * f_best + alpha * f_cand
            f_blend = np.maximum(f_blend, 0.0)
            C_blend = quick_score(f_blend)
            if C_blend > C_best * 0.95:  # Only refine if within 5% of best
                f_ref, C_ref = full_dk_cycle(f_blend, 500, 0.001)
                if C_ref > C_best:
                    C_best = C_ref
                    f_best = f_ref.copy()
                    np.save('best_dinkelbach_iter.npy', f_best)
                    np.save('best_comb_factory.npy', f_best)
                    print(f"  Blend d={d} env={env} α={alpha}: "
                          f"NEW BEST C = {C_best:.12f} !!!")
                else:
                    print(f"  Blend d={d} env={env} α={alpha}: C = {C_ref:.10f}")
    sys.stdout.flush()

    # Phase 5: Extended refinement of best with surgery + dk_iter
    print(f"\n{'='*60}")
    print("Phase 5: Extended refinement with surgery + dk_iter")
    print(f"{'='*60}")
    sys.stdout.flush()

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

    expand_schedule = [200, 500, 1000, 1500, 2000, 300, 750, 1200]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    scale_schedule = [0.001, 0.002, 0.005, 0.0005, 0.003]

    for round_num in range(50):
        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
            C_shared = quick_score(f_shared)
            if C_shared > C_best:
                C_best = C_shared
                f_best = f_shared.copy()
                print(f"  Loaded better shared: C = {C_best:.12f}")
        except:
            pass

        improved = False
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
                np.save('best_comb_factory.npy', f_best)
                improved = True
                print(f"  Round {round_num}, cycle {cycle}: "
                      f"NEW BEST C = {C_best:.12f} !!!")

        if round_num % 5 == 4 or not improved:
            elapsed = time.time() - t0
            print(f"  Round {round_num}: C_best = {C_best:.12f} "
                  f"(improved={improved}) [{elapsed:.0f}s]")
        sys.stdout.flush()

    np.save('best_comb_factory.npy', f_best)
    print(f"\nFINAL: C = {C_best:.12f}")
    print(f"Total time: {time.time() - t0:.0f}s")

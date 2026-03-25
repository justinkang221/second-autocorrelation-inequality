#!/usr/bin/env python3
"""
Comb parametric optimization for n=1600k.
Parameterize f as a comb: f[i] = A(i) if i mod d in {0,...,w-1}, else 0
where A is a smooth envelope parameterized by a low-dim B-spline.
Optimize d (spacing), w (tooth width), and envelope coefficients.
This is a much lower-dimensional search that can explore structural changes.
"""
import numpy as np
import torch
import torch.nn.functional as F
import time
import sys

device = torch.device('cuda')

def gpu_score_exact(f_t):
    n = f_t.shape[0]; nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    Fq = torch.fft.rfft(f_t, n=nfft); conv = torch.fft.irfft(Fq*Fq, n=nfft)[:nc]
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
        Fq = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(Fq*Fq, n=nfft)[:nc]
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
            Fq = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(Fq*Fq, n=nfft)[:nc]
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


def analyze_comb(f_np):
    """Extract comb parameters from a solution."""
    nonzero = np.nonzero(f_np > 1e-10)[0]
    if len(nonzero) == 0:
        return None
    gaps = np.diff(nonzero)
    cluster_breaks = np.where(gaps > 50)[0]
    clusters = []
    start = 0
    for b in cluster_breaks:
        clusters.append(nonzero[start:b+1])
        start = b+1
    clusters.append(nonzero[start:])
    centers = np.array([np.mean(c) for c in clusters])
    masses = np.array([np.sum(f_np[c]) for c in clusters])
    widths = np.array([c[-1]-c[0]+1 for c in clusters])
    return {'centers': centers, 'masses': masses, 'widths': widths,
            'n_teeth': len(clusters), 'clusters': clusters}


def generate_comb(n, spacing, n_teeth, tooth_width, center_offset,
                  envelope_coeffs, satellite_frac=0.0, satellite_offset=0.0):
    """Generate a comb function with given parameters.

    envelope_coeffs: coefficients for polynomial envelope on [0, 1]
    """
    f = np.zeros(n, dtype=np.float64)

    # Main cluster teeth positions
    teeth_centers = center_offset + np.arange(n_teeth) * spacing
    teeth_centers = teeth_centers.astype(int)
    teeth_centers = teeth_centers[(teeth_centers >= 0) & (teeth_centers < n)]

    # Envelope: polynomial in normalized position
    if len(teeth_centers) == 0:
        return f
    positions_norm = (teeth_centers - teeth_centers[0]) / max(teeth_centers[-1] - teeth_centers[0], 1)
    envelope = np.zeros(len(teeth_centers))
    for k, c in enumerate(envelope_coeffs):
        envelope += c * positions_norm**k
    envelope = np.maximum(envelope, 0.0)

    # Place teeth
    hw = tooth_width // 2
    for i, (tc, amp) in enumerate(zip(teeth_centers, envelope)):
        lo = max(0, tc - hw)
        hi = min(n, tc + hw + 1)
        f[lo:hi] = amp

    # Satellite cluster
    if satellite_frac > 0 and satellite_offset > 0:
        sat_center = int(center_offset + satellite_offset)
        sat_n_teeth = max(1, int(n_teeth * 0.1))
        sat_spacing = spacing
        for t in range(sat_n_teeth):
            tc = sat_center + t * sat_spacing
            if 0 <= tc < n:
                lo = max(0, tc - hw)
                hi = min(n, tc + hw + 1)
                f[lo:hi] = satellite_frac * np.mean(envelope)

    return f


def random_comb_search(n, n_trials=50, best_C_ref=0.0):
    """Random search over comb parameters."""
    best_C = best_C_ref
    best_f = None

    for trial in range(n_trials):
        # Random comb parameters
        spacing = np.random.randint(1500, 5000)
        n_teeth = np.random.randint(100, 400)
        tooth_width = np.random.randint(5, 50)
        center_offset = np.random.randint(0, n // 10)

        # Random polynomial envelope (degree 2-4)
        degree = np.random.randint(2, 5)
        coeffs = np.random.uniform(-1, 5, degree + 1)
        coeffs[0] = np.random.uniform(1, 10)  # Ensure positive start

        # Satellite parameters
        satellite_frac = np.random.uniform(0, 0.3)
        satellite_offset = np.random.uniform(n * 0.4, n * 0.7)

        f = generate_comb(n, spacing, n_teeth, tooth_width, center_offset,
                         coeffs, satellite_frac, satellite_offset)

        if np.sum(f) < 1e-10:
            continue

        C = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))

        if C > best_C:
            best_C = C
            best_f = f.copy()
            print(f"  Trial {trial}: C = {C:.10f} (spacing={spacing}, teeth={n_teeth}, "
                  f"width={tooth_width}, deg={degree})")
            sys.stdout.flush()

    return best_f, best_C


def perturb_comb_structure(f_np, n, perturbation_type='shift'):
    """Make structural perturbations to the comb."""
    info = analyze_comb(f_np)
    if info is None or info['n_teeth'] < 5:
        return f_np

    f_new = f_np.copy()

    if perturbation_type == 'shift':
        # Shift entire function by a small amount
        shift = np.random.randint(-50, 51)
        if shift > 0:
            f_new[shift:] = f_np[:-shift]
            f_new[:shift] = 0
        elif shift < 0:
            f_new[:shift] = f_np[-shift:]
            f_new[shift:] = 0

    elif perturbation_type == 'compress':
        # Compress support: reduce spacing between teeth by interpolation
        factor = np.random.uniform(0.95, 0.99)
        nonzero = np.nonzero(f_np > 1e-10)[0]
        center = int(np.mean(nonzero))
        f_new = np.zeros(n, dtype=np.float64)
        for idx in nonzero:
            new_idx = int(center + (idx - center) * factor)
            if 0 <= new_idx < n:
                f_new[new_idx] += f_np[idx]

    elif perturbation_type == 'expand':
        # Expand support
        factor = np.random.uniform(1.01, 1.05)
        nonzero = np.nonzero(f_np > 1e-10)[0]
        center = int(np.mean(nonzero))
        f_new = np.zeros(n, dtype=np.float64)
        for idx in nonzero:
            new_idx = int(center + (idx - center) * factor)
            if 0 <= new_idx < n:
                f_new[new_idx] += f_np[idx]

    elif perturbation_type == 'envelope_scale':
        # Scale envelope: make early teeth larger, later teeth smaller (or vice versa)
        nonzero = np.nonzero(f_np > 1e-10)[0]
        if len(nonzero) > 0:
            pos_norm = (nonzero - nonzero[0]) / max(nonzero[-1] - nonzero[0], 1)
            power = np.random.uniform(-0.3, 0.3)
            scale = (1 - pos_norm) ** power if power > 0 else (pos_norm + 0.01) ** (-power)
            scale = scale / np.mean(scale)  # Normalize to preserve total mass
            f_new[nonzero] = f_np[nonzero] * scale

    elif perturbation_type == 'satellite_adjust':
        # Adjust satellite cluster amplitude
        info = analyze_comb(f_np)
        centers = info['centers']
        masses = info['masses']
        # Find the gap that separates main from satellite
        spacings = np.diff(centers)
        if len(spacings) > 0:
            big_gap_idx = np.argmax(spacings)
            # Scale satellite (everything after the big gap)
            sat_clusters = info['clusters'][big_gap_idx+1:]
            sat_scale = np.random.uniform(0.8, 1.2)
            for c in sat_clusters:
                f_new[c] *= sat_scale

    return np.maximum(f_new, 0.0)


if __name__ == "__main__":
    print("=" * 70)
    print("Comb parametric optimization for n=1600k")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f_best = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")
    sys.stdout.flush()

    # Analyze current comb structure
    info = analyze_comb(f_best)
    print(f"Current structure: {info['n_teeth']} teeth")
    sys.stdout.flush()

    # Phase 1: Try random comb constructions (might find a better basin)
    print("\n--- Phase 1: Random comb search ---")
    sys.stdout.flush()
    f_comb, C_comb = random_comb_search(n, n_trials=100, best_C_ref=0.0)
    if f_comb is not None:
        print(f"Best random comb: C = {C_comb:.10f}")
        # Polish it
        f_comb, C_comb = gpu_lbfgsb(f_comb, maxiter=500, soft_temp=0.001)
        print(f"After LBFGS polish: C = {C_comb:.10f}")
        f_comb, C_comb = gpu_dinkelbach(f_comb, n_iters=5, maxiter_inner=150, soft_temp=0.001)
        print(f"After Dinkelbach: C = {C_comb:.10f}")
        if C_comb > C_best:
            C_best = C_comb; f_best = f_comb.copy()
            np.save('best_1600k.npy', f_best)
            print(f"*** Random comb beats current best! C = {C_best:.13f} ***")
    sys.stdout.flush()

    # Phase 2: Structural perturbations on current best
    print("\n--- Phase 2: Structural perturbations ---")
    sys.stdout.flush()
    perturbation_types = ['shift', 'compress', 'expand', 'envelope_scale', 'satellite_adjust']
    n_improvements = 0

    for round_num in range(500):
        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best:
                    C_best = C_shared; f_best = f_shared.copy()
                    print(f"  Cross-pollinated: C = {C_best:.13f}")
                    sys.stdout.flush()
        except: pass

        # Structural perturbation
        ptype = perturbation_types[round_num % len(perturbation_types)]
        f_work = perturb_comb_structure(f_best, n, perturbation_type=ptype)

        # Polish with LBFGS + Dinkelbach
        temp = [0.01, 0.005, 0.001, 0.0005, 0.0001][round_num % 5]
        f_work, C_work = gpu_lbfgsb(f_work, maxiter=500, soft_temp=temp)
        f_work, C_dk = gpu_dinkelbach(f_work, n_iters=5, maxiter_inner=150, soft_temp=temp)
        C_work = max(C_work, C_dk)

        if C_dk > C_best:
            C_best = C_dk; f_best = f_work.copy()
            np.save('best_1600k.npy', f_best)
            n_improvements += 1
            print(f"  Round {round_num} ({ptype}): NEW BEST C = {C_best:.13f} !!!")
            sys.stdout.flush()

        if round_num % 10 == 9:
            elapsed = time.time() - t0
            print(f"  Round {round_num}: C = {C_best:.13f} [{elapsed:.0f}s, {n_improvements} impr]")
            sys.stdout.flush()

    np.save('best_1600k.npy', f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

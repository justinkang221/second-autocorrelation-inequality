#!/usr/bin/env python3
"""
Comb parameterization for 1.6M optimization.
Instead of optimizing all 1.6M values, parameterize the comb structure:
  - positions (continuous relaxation)
  - heights
  - widths
This gives ~8500 parameters instead of 1.6M → much faster L-BFGS.
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
    Ft = torch.fft.rfft(f_t, n=nfft); conv = torch.fft.irfft(Ft*Ft, n=nfft)[:nc]
    h = 1.0/(nc+1); z = torch.zeros(1, device=device, dtype=torch.float64)
    y = torch.cat([z, conv, z]); y0, y1 = y[:-1], y[1:]
    l2sq = (h/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv)/(nc+1); linf = torch.max(conv)
    return (l2sq/(l1*linf)).item()


def extract_teeth(f_np, min_height_frac=0.005):
    """Extract tooth positions, heights, and widths from comb function."""
    f_max = np.max(f_np)
    threshold = f_max * min_height_frac
    above = f_np > threshold
    teeth = []
    in_tooth = False
    start = 0
    for i in range(len(f_np)):
        if above[i] and not in_tooth:
            start = i
            in_tooth = True
        elif not above[i] and in_tooth:
            peak_idx = start + np.argmax(f_np[start:i])
            teeth.append({
                'pos': float(peak_idx),
                'height': float(f_np[peak_idx]),
                'width': float(i - start),
                'start': start, 'end': i
            })
            in_tooth = False
    if in_tooth:
        peak_idx = start + np.argmax(f_np[start:])
        teeth.append({
            'pos': float(peak_idx),
            'height': float(f_np[peak_idx]),
            'width': float(len(f_np) - start),
            'start': start, 'end': len(f_np)
        })
    return teeth


def generate_comb_soft(positions, log_heights, log_widths, n, grid):
    """
    Generate a comb function from parameters using soft/differentiable placement.
    Uses Gaussian-shaped teeth for differentiability w.r.t. position.

    positions: (n_teeth,) — center positions (continuous)
    log_heights: (n_teeth,) — log of heights (ensures positivity)
    log_widths: (n_teeth,) — log of half-widths (ensures positivity)
    grid: (n,) — precomputed grid [0, 1, ..., n-1]
    """
    heights = torch.exp(log_heights)  # (n_teeth,)
    half_widths = torch.exp(log_widths)  # (n_teeth,)

    # Compute distance from each grid point to each tooth center
    # grid: (n,), positions: (n_teeth,)
    # Use broadcasting: (n, 1) - (1, n_teeth) = (n, n_teeth)
    # But n=1.6M × n_teeth=2849 is too big for memory!
    # Instead, use sparse computation: each tooth only affects nearby points

    f = torch.zeros(n, device=device, dtype=torch.float64)

    for i in range(len(positions)):
        pos = positions[i]
        h = heights[i]
        hw = half_widths[i]

        # Only compute within ±4*half_width of center
        radius = int(4 * hw.item()) + 2
        lo = max(0, int(pos.item()) - radius)
        hi = min(n, int(pos.item()) + radius + 1)

        local_grid = grid[lo:hi]
        # Gaussian tooth shape (differentiable w.r.t. position)
        dist = (local_grid - pos) / (hw + 1e-8)
        tooth = h * torch.exp(-0.5 * dist ** 2)
        f[lo:hi] = f[lo:hi] + tooth

    return f


def generate_comb_batch(positions, log_heights, log_widths, n, grid):
    """
    Batch version: groups teeth by similar width for efficiency.
    Falls back to loop but with vectorized local computation.
    """
    heights = torch.exp(log_heights)
    half_widths = torch.exp(log_widths)

    f = torch.zeros(n, device=device, dtype=torch.float64)

    # Process teeth in chunks to keep memory manageable
    n_teeth = len(positions)

    for i in range(n_teeth):
        pos = positions[i]
        h = heights[i]
        hw = half_widths[i]

        radius = int(4 * hw.item()) + 2
        center = int(pos.item())
        lo = max(0, center - radius)
        hi = min(n, center + radius + 1)

        if hi <= lo:
            continue

        local_grid = grid[lo:hi]
        dist = (local_grid - pos) / (hw + 1e-8)
        tooth = h * torch.exp(-0.5 * dist ** 2)
        f[lo:hi] = f[lo:hi] + tooth

    return f


def comb_dinkelbach(positions, log_heights, log_widths, n, grid,
                    n_outer=5, n_inner=2000, beta=1e8):
    """Dinkelbach optimization over comb parameters."""
    nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    # Initial score
    with torch.no_grad():
        f_init = generate_comb_batch(positions, log_heights, log_widths, n, grid)
    C_init = gpu_score_exact(f_init)
    lam = C_init
    best_C = C_init

    # Detach and clone for optimization
    pos_opt = positions.detach().clone().requires_grad_(True)
    lh_opt = log_heights.detach().clone().requires_grad_(True)
    lw_opt = log_widths.detach().clone().requires_grad_(True)

    best_params = (pos_opt.data.clone(), lh_opt.data.clone(), lw_opt.data.clone())

    for outer in range(n_outer):
        pos_opt = best_params[0].clone().requires_grad_(True)
        lh_opt = best_params[1].clone().requires_grad_(True)
        lw_opt = best_params[2].clone().requires_grad_(True)

        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [pos_opt, lh_opt, lw_opt], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-12, tolerance_change=1e-14
        )
        best_C_inner = [0.0]
        best_params_inner = [best_params]
        eval_count = [0]

        def closure():
            optimizer.zero_grad()
            f = generate_comb_batch(pos_opt, lh_opt, lw_opt, n, grid)

            Ft = torch.fft.rfft(f, n=nfft)
            conv = torch.fft.irfft(Ft*Ft, n=nfft)[:nc]
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
            if C_exact > best_C_inner[0] and C_exact < 1.0:
                best_C_inner[0] = C_exact
                best_params_inner[0] = (pos_opt.data.clone(), lh_opt.data.clone(), lw_opt.data.clone())

            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    Outer {outer} exception: {e}", flush=True)

        best_params = best_params_inner[0]

        # Compute exact score
        with torch.no_grad():
            f_new = generate_comb_batch(best_params[0], best_params[1], best_params[2], n, grid)
        C_new = gpu_score_exact(f_new)
        lam = C_new

        if C_new > best_C and C_new < 1.0:
            best_C = C_new

        print(f"    Comb Dink outer={outer}: C={C_new:.13f} (evals={eval_count[0]})", flush=True)

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            print(f"    Converged", flush=True)
            break

    # Generate final f
    with torch.no_grad():
        f_final = generate_comb_batch(best_params[0], best_params[1], best_params[2], n, grid)

    return f_final.cpu().numpy(), best_C, best_params


def full_dinkelbach_polish(f_np, n_outer=3, n_inner=5000, beta=1e9):
    """Standard full-vector Dinkelbach polish."""
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
            Ft = torch.fft.rfft(f, n=nfft)
            conv = torch.fft.irfft(Ft*Ft, n=nfft)[:nc]
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
        except Exception:
            pass

        w = best_w_inner[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new

        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)):
            best_C = C_new; best_f = f_new.copy()

        print(f"    Full Dink outer={outer}: C={C_new:.13f}", flush=True)

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


if __name__ == "__main__":
    print("=" * 70)
    print("Comb Parameterization Optimizer for 1.6M")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f_best = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")

    # Extract tooth structure
    teeth = extract_teeth(f_best)
    n_teeth = len(teeth)
    print(f"Extracted {n_teeth} teeth")

    # Initialize comb parameters
    positions = torch.tensor([t['pos'] for t in teeth], dtype=torch.float64, device=device)
    log_heights = torch.tensor([np.log(max(t['height'], 1e-10)) for t in teeth], dtype=torch.float64, device=device)
    log_widths = torch.tensor([np.log(max(t['width']/4.0, 0.5)) for t in teeth], dtype=torch.float64, device=device)

    grid = torch.arange(n, dtype=torch.float64, device=device)

    # Verify reconstruction quality
    with torch.no_grad():
        f_recon = generate_comb_batch(positions, log_heights, log_widths, n, grid)
    C_recon = gpu_score_exact(f_recon)
    print(f"Reconstructed comb: C = {C_recon:.13f}")
    print(f"Parameters: {n_teeth * 3} (vs {n} full)")
    sys.stdout.flush()

    save_file = 'best_1600k.npy'
    rng = np.random.default_rng(99)
    n_improvements = 0

    # Phase 1: Optimize comb parameters with Dinkelbach at increasing beta
    print("\n--- Phase 1: Comb parameter optimization ---")
    for beta in [1e6, 1e7, 1e8, 1e9]:
        print(f"\n  Beta = {beta:.0e}", flush=True)
        f_comb, C_comb, best_params = comb_dinkelbach(
            positions, log_heights, log_widths, n, grid,
            n_outer=5, n_inner=2000, beta=beta
        )
        positions, log_heights, log_widths = best_params

        print(f"  Comb C = {C_comb:.13f} [{time.time()-t0:.0f}s]", flush=True)

        if C_comb > C_best and C_comb < 1.0 and np.all(np.isfinite(f_comb)) and np.max(f_comb) < 1e6:
            # Polish with full Dinkelbach
            print(f"  Polishing with full Dinkelbach...", flush=True)
            f_polished, C_polished = full_dinkelbach_polish(f_comb, n_outer=2, n_inner=5000, beta=beta)
            print(f"  Polished C = {C_polished:.13f}", flush=True)

            if C_polished > C_best and C_polished < 1.0:
                C_best = C_polished; f_best = f_polished.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  NEW BEST C = {C_best:.13f} !!!", flush=True)

    # Phase 2: Perturb comb parameters and re-optimize
    print("\n--- Phase 2: Comb perturbation + optimization ---")
    for round_num in range(200):
        # Choose perturbation type
        perturb_type = rng.choice(['pos_shift', 'height_scale', 'width_scale',
                                    'add_tooth', 'remove_tooth', 'swap_teeth'])

        pos_p = positions.clone()
        lh_p = log_heights.clone()
        lw_p = log_widths.clone()

        if perturb_type == 'pos_shift':
            # Shift random teeth positions
            n_shift = max(1, int(n_teeth * rng.uniform(0.01, 0.1)))
            idxs = rng.choice(n_teeth, size=n_shift, replace=False)
            shifts = torch.tensor(rng.normal(0, 5.0, size=n_shift), dtype=torch.float64, device=device)
            pos_p[idxs] += shifts

        elif perturb_type == 'height_scale':
            n_scale = max(1, int(n_teeth * rng.uniform(0.01, 0.1)))
            idxs = rng.choice(n_teeth, size=n_scale, replace=False)
            scales = torch.tensor(rng.normal(0, 0.05, size=n_scale), dtype=torch.float64, device=device)
            lh_p[idxs] += scales

        elif perturb_type == 'width_scale':
            n_scale = max(1, int(n_teeth * rng.uniform(0.01, 0.1)))
            idxs = rng.choice(n_teeth, size=n_scale, replace=False)
            scales = torch.tensor(rng.normal(0, 0.1, size=n_scale), dtype=torch.float64, device=device)
            lw_p[idxs] += scales

        elif perturb_type == 'add_tooth':
            # Add a tooth in the largest gap
            sorted_pos = torch.sort(pos_p)[0]
            gaps = sorted_pos[1:] - sorted_pos[:-1]
            gap_idx = torch.argmax(gaps).item()
            new_pos = (sorted_pos[gap_idx] + sorted_pos[gap_idx + 1]) / 2
            # Average height and width of neighbors
            new_lh = (lh_p[gap_idx] + lh_p[min(gap_idx+1, n_teeth-1)]) / 2
            new_lw = (lw_p[gap_idx] + lw_p[min(gap_idx+1, n_teeth-1)]) / 2
            pos_p = torch.cat([pos_p, new_pos.unsqueeze(0)])
            lh_p = torch.cat([lh_p, new_lh.unsqueeze(0)])
            lw_p = torch.cat([lw_p, new_lw.unsqueeze(0)])

        elif perturb_type == 'remove_tooth':
            if n_teeth > 10:
                # Remove the smallest tooth
                idx = torch.argmin(torch.exp(lh_p)).item()
                mask = torch.ones(len(pos_p), dtype=torch.bool, device=device)
                mask[idx] = False
                pos_p = pos_p[mask]
                lh_p = lh_p[mask]
                lw_p = lw_p[mask]

        elif perturb_type == 'swap_teeth':
            # Swap positions of two random teeth
            if n_teeth >= 2:
                i, j = rng.choice(n_teeth, size=2, replace=False)
                pos_p[i], pos_p[j] = pos_p[j].clone(), pos_p[i].clone()

        # Quick optimize with comb parameters
        try:
            f_comb, C_comb, new_params = comb_dinkelbach(
                pos_p, lh_p, lw_p, n, grid,
                n_outer=3, n_inner=1000, beta=1e9
            )
        except Exception as e:
            print(f"  R{round_num} {perturb_type}: error {e}", flush=True)
            continue

        if C_comb > C_best and C_comb < 1.0 and np.all(np.isfinite(f_comb)) and np.max(f_comb) < 1e6:
            # Polish with full Dinkelbach
            f_polished, C_polished = full_dinkelbach_polish(f_comb, n_outer=2, n_inner=5000, beta=1e9)

            if C_polished > C_best and C_polished < 1.0 and np.all(np.isfinite(f_polished)):
                C_best = C_polished; f_best = f_polished.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                positions, log_heights, log_widths = new_params
                print(f"  R{round_num} {perturb_type}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        if round_num % 10 == 0:
            print(f"  R{round_num}: C_best={C_best:.13f} [{time.time()-t0:.0f}s, {n_improvements} impr, "
                  f"n_teeth={len(pos_p)}]", flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

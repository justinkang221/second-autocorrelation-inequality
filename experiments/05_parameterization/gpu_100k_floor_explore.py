#!/usr/bin/env python3
"""
Explore the zero regions of the 100k solution.

Key hypothesis: the w² parameterization forces positions to stay at zero
because gradient vanishes there. But maybe nonzero values in those regions
would improve C(f). This script:

1. Adds various "floors" (small positive values) to the zero regions
2. Optimizes with exp(v) parameterization (which CAN adjust zeros freely)
3. Tests whether the optimizer wants to grow mass in zero regions

Also tries:
- Interpolation with constant function
- Adding teeth systematically in largest gaps
- exp(v) optimization from perturbed starting points with nonzero floors
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


def dinkelbach_exp(f_np, n_outer=5, n_inner=5000, beta=1e8):
    """Dinkelbach with exp(v) parameterization."""
    n = len(f_np); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    eps = 1e-20
    v_init = np.log(np.maximum(f_np, eps))
    v_init[f_np < 1e-15] = -40.0

    lam = gpu_score_exact(torch.tensor(np.maximum(f_np, 0.0), dtype=torch.float64, device=device))
    best_C = lam; best_f = np.maximum(f_np, 0.0).copy()

    for outer in range(n_outer):
        v_t = torch.tensor(v_init, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [v_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_v = [v_t.data.clone()]
        best_C_inner = [0.0]

        def closure():
            optimizer.zero_grad()
            f = torch.exp(v_t)
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
            if C_exact > best_C_inner[0] and C_exact < 1.0:
                best_C_inner[0] = C_exact
                best_v[0] = v_t.data.clone()
            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass

        f_new = torch.exp(best_v[0]).cpu().numpy()
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new
        v_init = best_v[0].cpu().numpy()

        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()

        print(f"    exp outer={outer}: C={C_new:.13f}", flush=True)
        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def dinkelbach_w2(f_np, n_outer=5, n_inner=5000, beta=1e8):
    """Standard w² Dinkelbach."""
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    lam = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))
    best_C = lam; best_f = np.maximum(f_np, 0.0).copy()

    for outer in range(n_outer):
        w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [w_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-15, tolerance_change=1e-16
        )
        best_w = [w_t.data.clone()]
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
                best_w[0] = w_t.data.clone()
            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass

        w = best_w[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new
        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)):
            best_C = C_new; best_f = f_new.copy()
        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


if __name__ == "__main__":
    print("=" * 70)
    print("Zero-region exploration for 100k")
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

    f_max = np.max(f_best)
    n_zero = np.sum(f_best < f_max * 0.001)
    n_nonzero = n - n_zero
    print(f"Nonzero positions: {n_nonzero}, Zero positions: {n_zero} ({100*n_zero/n:.1f}%)")
    sys.stdout.flush()

    save_file = 'best_impevo_100k.npy'
    rng = np.random.default_rng(2025)
    n_improvements = 0

    # Experiment 1: Add uniform floor to zero regions, then optimize with exp(v)
    print("\n--- Experiment 1: Uniform floor + exp(v) optimization ---")
    for floor_frac in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]:
        floor = f_max * floor_frac
        f_floored = f_best.copy()
        f_floored[f_best < f_max * 0.001] = floor

        C_floored = gpu_score_exact(torch.tensor(f_floored, dtype=torch.float64, device=device))
        print(f"  floor={floor_frac:.0e}: C_init={C_floored:.13f} (Δ={C_floored-C_best:.2e})")

        # Optimize with exp(v) — this can freely adjust the floor
        for beta in [1e7, 1e8]:
            f_opt, C_opt = dinkelbach_exp(f_floored, n_outer=3, n_inner=3000, beta=beta)

            improved = ""
            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                C_best = C_opt; f_best = f_opt.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                improved = " *** NEW BEST ***"

            # Check if optimizer kept the floor or pushed it back to zero
            n_new_nonzero = np.sum(f_opt > f_max * 0.001)
            print(f"    beta={beta:.0e}: C={C_opt:.13f}{improved} "
                  f"(nonzero: {n_nonzero}→{n_new_nonzero}) [{time.time()-t0:.0f}s]", flush=True)

    # Experiment 2: Interpolation with constant
    print("\n--- Experiment 2: Interpolate with constant ---")
    mean_val = np.mean(f_best[f_best > 0])
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
        f_interp = (1 - alpha) * f_best + alpha * mean_val

        C_interp = gpu_score_exact(torch.tensor(f_interp, dtype=torch.float64, device=device))

        # Optimize with w² (standard)
        for beta in [1e7, 1e8]:
            f_opt, C_opt = dinkelbach_w2(f_interp, n_outer=3, n_inner=5000, beta=beta)

            improved = ""
            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                C_best = C_opt; f_best = f_opt.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                improved = " *** NEW BEST ***"

            print(f"  α={alpha}, beta={beta:.0e}: init={C_interp:.10f} → opt={C_opt:.13f}{improved} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # Experiment 3: Add teeth in largest gaps
    print("\n--- Experiment 3: Systematic tooth insertion ---")
    # Find teeth
    teeth = []
    in_tooth = False
    for i in range(n):
        if f_best[i] > f_max * 0.001 and not in_tooth:
            start = i
            in_tooth = True
        elif (f_best[i] <= f_max * 0.001 or i == n-1) and in_tooth:
            end = i
            peak = start + np.argmax(f_best[start:end])
            teeth.append({'start': start, 'end': end, 'peak': peak,
                          'height': f_best[peak], 'width': end - start})
            in_tooth = False

    print(f"  Found {len(teeth)} teeth")

    # Find gaps between consecutive teeth
    gaps = []
    for i in range(len(teeth) - 1):
        gap_start = teeth[i]['end']
        gap_end = teeth[i+1]['start']
        gap_size = gap_end - gap_start
        if gap_size > 0:
            gaps.append((gap_size, gap_start, gap_end, i))

    gaps.sort(reverse=True)
    print(f"  Largest gaps: {[g[0] for g in gaps[:10]]}")

    # Try inserting teeth in the top gaps
    for n_gaps_to_fill in [1, 3, 5, 10, 20]:
        f_inserted = f_best.copy()
        for gi in range(min(n_gaps_to_fill, len(gaps))):
            gap_size, gap_start, gap_end, tooth_idx = gaps[gi]
            center = (gap_start + gap_end) // 2

            # Use average tooth profile
            ref = teeth[tooth_idx]
            ref_width = ref['width']
            ref_height = ref['height']

            half_w = min(ref_width // 2, gap_size // 4)
            if half_w < 1:
                continue
            lo = max(gap_start, center - half_w)
            hi = min(gap_end, center + half_w + 1)
            w = hi - lo
            x = np.linspace(-1, 1, w)
            tooth = ref_height * 0.5 * np.maximum(1 - np.abs(x), 0)
            f_inserted[lo:hi] = tooth

        C_inserted = gpu_score_exact(torch.tensor(f_inserted, dtype=torch.float64, device=device))
        print(f"  +{n_gaps_to_fill} teeth: C_init={C_inserted:.10f}", flush=True)

        # Optimize with exp(v) — let optimizer freely adjust new teeth
        f_opt, C_opt = dinkelbach_exp(f_inserted, n_outer=3, n_inner=5000, beta=1e8)
        # Then polish with w²
        f_opt2, C_opt2 = dinkelbach_w2(f_opt, n_outer=3, n_inner=5000, beta=1e8)

        improved = ""
        if C_opt2 > C_best and C_opt2 < 1.0 and np.all(np.isfinite(f_opt2)) and np.max(f_opt2) < 1e6:
            C_best = C_opt2; f_best = f_opt2.copy()
            np.save(save_file, f_best)
            n_improvements += 1
            improved = " *** NEW BEST ***"

        print(f"  +{n_gaps_to_fill} teeth: exp→{C_opt:.13f}, w²→{C_opt2:.13f}{improved} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # Experiment 4: Random floor with exp(v), many trials
    print("\n--- Experiment 4: Random floor patterns + exp(v) ---")
    for trial in range(30):
        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best and C_shared < 1.0:
                    C_best = C_shared; f_best = f_shared.copy()
        except: pass

        f_trial = f_best.copy()
        zero_mask = f_trial < f_max * 0.001

        # Random floor pattern in zero regions
        pattern = rng.choice(['uniform', 'gaussian_bumps', 'random_teeth', 'mirror'])

        if pattern == 'uniform':
            floor = f_max * 10 ** rng.uniform(-6, -2)
            f_trial[zero_mask] = floor

        elif pattern == 'gaussian_bumps':
            # Add random Gaussian bumps in zero regions
            n_bumps = rng.integers(1, 20)
            zero_positions = np.where(zero_mask)[0]
            if len(zero_positions) > 0:
                for _ in range(n_bumps):
                    center = rng.choice(zero_positions)
                    width = rng.integers(5, 50)
                    height = f_max * 10 ** rng.uniform(-4, -1)
                    lo = max(0, center - width)
                    hi = min(n, center + width)
                    x = np.linspace(-2, 2, hi - lo)
                    f_trial[lo:hi] = np.maximum(f_trial[lo:hi], height * np.exp(-x**2))

        elif pattern == 'random_teeth':
            # Add teeth mimicking existing structure
            n_new = rng.integers(1, 10)
            zero_positions = np.where(zero_mask)[0]
            if len(zero_positions) > 0 and len(teeth) > 0:
                avg_width = int(np.mean([t['width'] for t in teeth]))
                avg_height = np.mean([t['height'] for t in teeth])
                for _ in range(n_new):
                    center = rng.choice(zero_positions)
                    half_w = max(1, avg_width // 2)
                    lo = max(0, center - half_w)
                    hi = min(n, center + half_w + 1)
                    w = hi - lo
                    x = np.linspace(-1, 1, w)
                    height = avg_height * rng.uniform(0.1, 1.0)
                    f_trial[lo:hi] = np.maximum(f_trial[lo:hi],
                                                 height * np.maximum(1 - np.abs(x), 0))

        elif pattern == 'mirror':
            # Mirror the nonzero pattern into zero regions
            nonzero_vals = f_best[~zero_mask]
            n_nz = len(nonzero_vals)
            # Place mirror at random offset
            if n_nz > 0:
                scale = rng.uniform(0.01, 0.5)
                offset = rng.integers(0, n)
                for i, val in enumerate(nonzero_vals):
                    idx = (np.where(~zero_mask)[0][i] + offset) % n
                    if zero_mask[idx]:
                        f_trial[idx] = val * scale

        # Optimize with exp(v)
        f_opt, C_opt = dinkelbach_exp(f_trial, n_outer=3, n_inner=3000, beta=1e8)

        # If promising, polish with w²
        if C_opt > C_best * 0.9999:
            f_opt2, C_opt2 = dinkelbach_w2(f_opt, n_outer=3, n_inner=5000, beta=1e8)
            if C_opt2 > C_best and C_opt2 < 1.0 and np.all(np.isfinite(f_opt2)) and np.max(f_opt2) < 1e6:
                C_best = C_opt2; f_best = f_opt2.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  Trial {trial} {pattern}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        if trial % 5 == 0:
            print(f"  Trial {trial} {pattern}: C={C_opt:.13f}, best={C_best:.13f} [{time.time()-t0:.0f}s]",
                  flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

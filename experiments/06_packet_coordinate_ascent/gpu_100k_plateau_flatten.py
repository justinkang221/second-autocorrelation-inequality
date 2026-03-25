#!/usr/bin/env python3
"""
Plateau-flattening optimization for 100k autocorrelation problem.

Key insight from Einstein Arena discussion:
- The autoconvolution g = f*f has a near-flat plateau at its maximum
- To improve C, we need to flatten the residual dips in this plateau
- For each "run" (contiguous nonzero block), compute its first-order
  contribution to the plateau dips
- Solve a small optimization to flatten the dips

Exact scoring: C = (2·Σc²ᵢ + Σcᵢcᵢ₊₁) / (3·Σcᵢ · max cᵢ)

Strategy:
1. Compute autoconv plateau and find dips
2. Compute per-run Jacobian: how each run's scale affects g at each dip
3. Solve small balancing problem to flatten dips
4. Apply the correction and polish with Dinkelbach
5. Repeat
"""

import numpy as np
import torch
import time
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# ============================================================
# Exact scoring (matches platform verifier)
# ============================================================
def compute_autoconv_gpu(f_t):
    """Compute autoconvolution via FFT on GPU."""
    n = f_t.shape[0]; nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F * F, n=nfft)[:nc]
    return conv


def exact_score_from_conv(c):
    """Compute C using exact platform formula from autoconvolution c."""
    # C = (2·Σc²ᵢ + Σcᵢcᵢ₊₁) / (3·Σcᵢ · max cᵢ)
    if isinstance(c, torch.Tensor):
        sum_c2 = torch.sum(c ** 2).item()
        sum_cc = torch.sum(c[:-1] * c[1:]).item()
        sum_c = torch.sum(c).item()
        max_c = torch.max(c).item()
    else:
        sum_c2 = np.sum(c ** 2)
        sum_cc = np.sum(c[:-1] * c[1:])
        sum_c = np.sum(c)
        max_c = np.max(c)
    return (2 * sum_c2 + sum_cc) / (3 * sum_c * max_c)


def gpu_score(f_t):
    """Compute C(f) using exact platform formula."""
    conv = compute_autoconv_gpu(f_t)
    return exact_score_from_conv(conv)


def score_np(f_np):
    return gpu_score(torch.tensor(f_np, dtype=torch.float64, device=device))


# Also compute with Simpson's rule for comparison
def gpu_score_simpson(f_t):
    n = f_t.shape[0]; nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F * F, n=nfft)[:nc]
    h = 1.0 / (nc + 1)
    z = torch.zeros(1, device=f_t.device, dtype=f_t.dtype)
    y = torch.cat([z, conv, z])
    y0, y1 = y[:-1], y[1:]
    l2sq = (h / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv) / (nc + 1)
    linf = torch.max(conv)
    return (l2sq / (l1 * linf)).item()


# ============================================================
# Block identification
# ============================================================
def find_blocks(f, threshold=1e-10):
    mask = f > threshold
    blocks = []
    in_block = False
    for i in range(len(f)):
        if mask[i] and not in_block:
            in_block = True; start = i
        elif not mask[i] and in_block:
            in_block = False
            blocks.append((start, i))
    if in_block:
        blocks.append((start, len(f)))
    return blocks


# ============================================================
# Plateau analysis
# ============================================================
def analyze_plateau(f_np, top_frac=0.001):
    """Analyze the autoconvolution plateau.

    Returns:
        conv: autoconvolution array
        plateau_mask: boolean mask of near-max positions
        dip_positions: positions of the deepest dips in the plateau
        max_val: maximum autoconv value
    """
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    conv = compute_autoconv_gpu(f_t).cpu().numpy()

    max_val = np.max(conv)
    threshold = max_val * (1 - top_frac)
    plateau_mask = conv >= threshold

    n_plateau = np.sum(plateau_mask)

    # Find dips within the plateau region
    # Look at positions near the top and find where g is lowest
    plateau_vals = conv[plateau_mask]
    plateau_idx = np.where(plateau_mask)[0]

    # Sort by value (ascending = deepest dips first)
    sorted_idx = np.argsort(plateau_vals)
    dip_positions = plateau_idx[sorted_idx]

    print(f"  Plateau analysis (top {top_frac*100}%): {n_plateau} positions")
    print(f"  Max: {max_val:.6f}, threshold: {threshold:.6f}")
    print(f"  Plateau range: [{np.min(plateau_vals):.6f}, {max_val:.6f}]")
    print(f"  Plateau std: {np.std(plateau_vals):.6e}")
    print(f"  Deepest 5 dips: {conv[dip_positions[:5]]}")

    return conv, plateau_mask, dip_positions, max_val


# ============================================================
# Per-run Jacobian computation
# ============================================================
def compute_run_jacobian(f_np, blocks, dip_positions, n_dips=100):
    """For each run, compute how scaling it by (1+ε) affects g at each dip.

    Jacobian J[i,j] = d(g[dip_j]) / d(alpha_i) evaluated at alpha_i = 1.

    Uses finite differences: J[i,j] ≈ (g_new[dip_j] - g[dip_j]) / epsilon
    """
    n = len(f_np)
    nc = 2*n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    dips = dip_positions[:n_dips]
    n_blocks = len(blocks)
    eps = 1e-6

    # Baseline autoconv at dip positions
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    conv_base = compute_autoconv_gpu(f_t).cpu().numpy()
    g_base = conv_base[dips]

    J = np.zeros((n_blocks, len(dips)))

    for bi, (s, e) in enumerate(blocks):
        # Perturb block bi by eps
        f_trial = f_np.copy()
        f_trial[s:e] *= (1 + eps)

        f_t_trial = torch.tensor(f_trial, dtype=torch.float64, device=device)
        conv_trial = compute_autoconv_gpu(f_t_trial).cpu().numpy()
        g_trial = conv_trial[dips]

        J[bi, :] = (g_trial - g_base) / eps

        if (bi + 1) % 100 == 0:
            print(f"    Jacobian: {bi+1}/{n_blocks} blocks computed")

    return J, g_base, dips


# ============================================================
# Flatten optimization: solve for run multipliers
# ============================================================
def solve_flattening(J, g_base, g_max, blocks, f_np, max_iter=100):
    """Find run multiplier adjustments that flatten the plateau.

    Want to maximize min(g[dip_j]) subject to max(g) not increasing too much.

    Simple approach: iteratively adjust the run with the largest positive
    gradient at the deepest current dip.

    More sophisticated: solve min-max problem.
    """
    n_blocks = len(blocks)
    n_dips = len(g_base)

    # Current multipliers (all start at 1.0)
    alphas = np.ones(n_blocks)

    # Current g values at dips (linearized)
    g_current = g_base.copy()

    best_min_g = np.min(g_current)
    best_alphas = alphas.copy()

    for iteration in range(max_iter):
        # Find the deepest dip
        worst_dip = np.argmin(g_current)
        worst_val = g_current[worst_dip]

        # Find which block has the largest positive Jacobian at the worst dip
        jac_at_worst = J[:, worst_dip]

        # Only consider blocks with positive contribution
        # and that haven't been adjusted too much
        candidates = np.where((jac_at_worst > 0) & (alphas > 0.1) & (alphas < 5.0))[0]

        if len(candidates) == 0:
            break

        # Pick the one with largest Jacobian
        best_block = candidates[np.argmax(jac_at_worst[candidates])]

        # How much to increase this block?
        # Simple step: increase by a small amount
        step = 0.01
        alphas[best_block] += step
        g_current += step * J[best_block, :]

        new_min = np.min(g_current)
        if new_min > best_min_g:
            best_min_g = new_min
            best_alphas = alphas.copy()

    return best_alphas


def solve_flattening_scipy(J, g_base, g_max, blocks, f_np):
    """Use scipy to solve the max-min problem more carefully."""
    from scipy.optimize import minimize

    n_blocks = len(blocks)

    # Objective: maximize the minimum g value across dips
    # i.e., minimize -min(g_base + J.T @ (alpha - 1))

    # Actually: maximize min_j(g_base_j + sum_i J_ij * delta_i) / g_max
    # where delta_i = alpha_i - 1

    # This is a linear max-min problem, solvable as LP
    # But let's use a smooth approximation

    def objective(delta):
        g_new = g_base + J.T @ delta
        # Smooth min approximation
        beta = 1000.0
        smooth_min = -np.log(np.sum(np.exp(-beta * g_new))) / beta
        return -smooth_min  # minimize negative of smooth_min

    def jac(delta):
        g_new = g_base + J.T @ delta
        beta = 1000.0
        weights = np.exp(-beta * g_new)
        weights /= np.sum(weights)
        return -J @ weights

    delta0 = np.zeros(n_blocks)

    # Bounds: alpha in [0.5, 3.0] so delta in [-0.5, 2.0]
    bounds = [(-0.5, 2.0)] * n_blocks

    result = minimize(objective, delta0, jac=jac, bounds=bounds, method='L-BFGS-B',
                     options={'maxiter': 500, 'ftol': 1e-15})

    alphas = 1.0 + result.x
    return alphas


# ============================================================
# Apply multipliers and evaluate
# ============================================================
def apply_multipliers(f_np, blocks, alphas):
    """Apply run multipliers to f."""
    f_new = f_np.copy()
    for (s, e), alpha in zip(blocks, alphas):
        f_new[s:e] *= alpha
    f_new = np.maximum(f_new, 0.0)
    return f_new


# ============================================================
# Dinkelbach polish
# ============================================================
def dinkelbach_polish(f_np, n_outer=5, n_inner=5000, beta=1e8):
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    lam = score_np(f_np)
    best_C = lam; best_f = f_np.copy()

    for outer in range(n_outer):
        w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [w_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_w = [w_t.data.clone()]
        best_Ci = [0.0]

        def closure():
            optimizer.zero_grad()
            f = w_t ** 2
            F = torch.fft.rfft(f, n=nfft)
            conv = torch.fft.irfft(F * F, n=nfft)[:nc]
            conv = torch.maximum(conv, torch.zeros(1, device=device, dtype=torch.float64))
            hh = 1.0 / (nc + 1)
            zz = torch.zeros(1, device=device, dtype=torch.float64)
            y = torch.cat([zz, conv, zz])
            y0, y1 = y[:-1], y[1:]
            l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
            l1 = torch.sum(conv) / (nc + 1)
            g_max = torch.max(conv)
            linf_proxy = g_max * torch.exp(
                torch.logsumexp(beta * (conv / (g_max + 1e-18) - 1.0), dim=0) / beta
            )
            obj = l2sq - current_lam * l1 * linf_proxy
            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_Ci[0]:
                best_Ci[0] = C_exact
                best_w[0] = w_t.data.clone()
            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except:
            pass

        w = best_w[0].cpu().numpy()
        f_new = w ** 2
        C_new = score_np(f_new)
        lam = C_new
        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()
        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


# ============================================================
# Direct per-run line search (exact, not linearized)
# ============================================================
def exact_per_run_search(f_np, blocks, n_cycles=10):
    """Exact per-run line search using actual C computation."""
    f_best = f_np.copy()
    C_best = score_np(f_best)

    for cycle in range(n_cycles):
        n_improved = 0
        C_cycle_start = C_best

        for bi, (s, e) in enumerate(blocks):
            block_vals = f_best[s:e].copy()
            if np.sum(block_vals) < 1e-15:
                continue

            # Golden section search for optimal alpha
            lo, hi = 0.5, 2.0
            C_lo = score_np(apply_single_block(f_best, s, e, block_vals, lo))
            C_hi = score_np(apply_single_block(f_best, s, e, block_vals, hi))

            for _ in range(30):
                if hi - lo < 1e-10:
                    break
                m1 = lo + 0.382 * (hi - lo)
                m2 = lo + 0.618 * (hi - lo)
                C1 = score_np(apply_single_block(f_best, s, e, block_vals, m1))
                C2 = score_np(apply_single_block(f_best, s, e, block_vals, m2))
                if C1 > C2:
                    hi = m2
                else:
                    lo = m1

            # Evaluate at center of final interval
            alpha_opt = (lo + hi) / 2
            f_trial = apply_single_block(f_best, s, e, block_vals, alpha_opt)
            C_trial = score_np(f_trial)

            # Also check alpha=1.0 (no change)
            if C_trial > C_best + 1e-15:
                f_best = f_trial
                C_best = C_trial
                n_improved += 1

        gain = C_best - C_cycle_start
        print(f"    Cycle {cycle}: C={C_best:.13f} (ΔC={gain:+.2e}), {n_improved} blocks improved")

        if n_improved == 0:
            break

    return f_best, C_best


def apply_single_block(f_np, s, e, block_vals, alpha):
    """Apply a scalar to one block, returning the modified f."""
    f_new = f_np.copy()
    f_new[s:e] = block_vals * alpha
    return f_new


# ============================================================
# Gradient-based plateau flattening with autograd
# ============================================================
def autograd_flatten(f_np, n_iter=5000):
    """Directly maximize C using PyTorch autograd with exact formula."""
    n = len(f_np); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    # Use w² parameterization
    w = np.sqrt(np.maximum(f_np, 0.0))
    w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)

    C_best = score_np(f_np)
    best_w = w_t.data.clone()

    # Use Adam for non-convex optimization
    optimizer = torch.optim.Adam([w_t], lr=1e-5)

    for step in range(n_iter):
        optimizer.zero_grad()

        f = w_t ** 2
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        conv = torch.maximum(conv, torch.zeros(1, device=device, dtype=torch.float64))

        # Exact platform formula (differentiable except for max)
        sum_c2 = torch.sum(conv ** 2)
        sum_cc = torch.sum(conv[:-1] * conv[1:])
        sum_c = torch.sum(conv)
        # Smooth max
        g_max = torch.max(conv)
        beta = 1e6
        max_smooth = g_max * torch.exp(
            torch.logsumexp(beta * (conv / (g_max + 1e-18) - 1.0), dim=0) / beta
        )

        C_approx = (2 * sum_c2 + sum_cc) / (3 * sum_c * max_smooth)

        loss = -C_approx
        loss.backward()
        optimizer.step()

        # Track exact C
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                f_cur = (w_t ** 2).cpu().numpy()
                f_cur = np.maximum(f_cur, 0.0)
                C_cur = score_np(f_cur)
                if C_cur > C_best and C_cur < 1.0:
                    C_best = C_cur
                    best_w = w_t.data.clone()
                print(f"    Adam step {step+1}: C={C_cur:.13f} (best={C_best:.13f})")

    f_out = (best_w ** 2).cpu().numpy()
    f_out = np.maximum(f_out, 0.0)
    return f_out, C_best


# ============================================================
# Targeted dip surgery
# ============================================================
def targeted_dip_surgery(f_np, n_rounds=20):
    """Find the autoconv dip positions and surgically modify f to fix them.

    For each dip position d in the autoconvolution:
    - g[d] = sum_{k} f[k] * f[d-k]
    - To increase g[d], we can increase f at positions k and d-k
    - Choose positions where f is small but in the support
    """
    f_best = f_np.copy()
    C_best = score_np(f_best)

    for round_idx in range(n_rounds):
        f_t = torch.tensor(f_best, dtype=torch.float64, device=device)
        conv = compute_autoconv_gpu(f_t).cpu().numpy()

        max_val = np.max(conv)
        n_conv = len(conv)
        n_f = len(f_best)

        # Find the dips in the top 0.01% of conv
        threshold = max_val * 0.9999
        near_max_mask = conv >= threshold
        near_max_idx = np.where(near_max_mask)[0]

        if len(near_max_idx) == 0:
            threshold = max_val * 0.999
            near_max_mask = conv >= threshold
            near_max_idx = np.where(near_max_mask)[0]

        # Sort by value (ascending) to find dips
        sorted_idx = near_max_idx[np.argsort(conv[near_max_idx])]

        # Try to fix the deepest dips
        improved = False
        for dip_pos in sorted_idx[:50]:
            # g[d] = sum_k f[k]*f[d-k]
            # To increase g[d], find which k pairs contribute most
            # and see if slightly increasing one of them helps C

            # Find support positions k where both f[k]>0 and f[d-k]>0
            for delta in [-2, -1, 0, 1, 2]:
                d = dip_pos + delta
                if d < 0 or d >= n_conv:
                    continue

                # Try adding a small amount at each position in the support
                for k in range(max(0, d - n_f + 1), min(n_f, d + 1)):
                    dk = d - k
                    if dk < 0 or dk >= n_f:
                        continue

                    # Only try if at least one position has some mass
                    if f_best[k] < 1e-10 and f_best[dk] < 1e-10:
                        continue

                    # Try increasing the smaller one
                    if f_best[k] < f_best[dk]:
                        pos = k
                    else:
                        pos = dk

                    for frac in [1.01, 1.05, 1.1, 1.5, 2.0]:
                        f_trial = f_best.copy()
                        f_trial[pos] *= frac
                        C_trial = score_np(f_trial)
                        if C_trial > C_best:
                            C_best = C_trial
                            f_best = f_trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            print(f"    Round {round_idx}: C={C_best:.13f}")
        else:
            print(f"    Round {round_idx}: no improvement from dip surgery")
            break

    return f_best, C_best


# ============================================================
# Main
# ============================================================
def main():
    save_file = 'best_impevo_100k.npy'
    f = np.maximum(np.load(save_file).astype(np.float64), 0.0)
    n = len(f)

    C_initial_exact = score_np(f)
    C_initial_simpson = gpu_score_simpson(torch.tensor(f, dtype=torch.float64, device=device))

    print(f"=" * 70)
    print(f"Plateau-flattening optimization for 100k")
    print(f"=" * 70)
    print(f"Starting: n={n}")
    print(f"C (exact platform formula): {C_initial_exact:.13f}")
    print(f"C (Simpson's rule):          {C_initial_simpson:.13f}")
    print(f"Difference: {C_initial_exact - C_initial_simpson:+.2e}")

    blocks = find_blocks(f, threshold=1e-10)
    print(f"Blocks: {len(blocks)}")

    C_best = C_initial_exact
    f_best = f.copy()

    # ============================================================
    # Phase 0: Check scoring consistency
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 0: Verify scoring")
    print(f"{'='*60}")

    # Compute with both methods
    f_t = torch.tensor(f, dtype=torch.float64, device=device)
    conv = compute_autoconv_gpu(f_t)
    c = conv.cpu().numpy()

    sum_c2 = np.sum(c**2)
    sum_cc = np.sum(c[:-1] * c[1:])
    sum_c = np.sum(c)
    max_c = np.max(c)
    C_manual = (2*sum_c2 + sum_cc) / (3 * sum_c * max_c)
    print(f"C (manual formula): {C_manual:.13f}")

    # ============================================================
    # Phase 1: Plateau analysis
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Plateau analysis")
    print(f"{'='*60}")

    conv_np, plateau_mask, dip_positions, max_val = analyze_plateau(f, top_frac=0.001)

    # Show the near-flat structure
    print(f"\n  Top 20 dip positions and their depths:")
    for i in range(min(20, len(dip_positions))):
        pos = dip_positions[i]
        depth = max_val - conv_np[pos]
        print(f"    pos={pos}: g={conv_np[pos]:.6f}, depth={depth:.6e} ({depth/max_val*100:.6f}%)")

    # ============================================================
    # Phase 2: Exact per-run line search
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Exact per-run line search (using platform formula)")
    print(f"{'='*60}")

    f_new, C_new = exact_per_run_search(f_best, blocks, n_cycles=10)
    if C_new > C_best:
        C_best = C_new
        f_best = f_new
        np.save(save_file, f_best)
        print(f"  Per-run search improved: C={C_best:.13f}")
    else:
        print(f"  Per-run search: no improvement")

    # ============================================================
    # Phase 3: Jacobian-based flattening
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 3: Jacobian-based plateau flattening")
    print(f"{'='*60}")

    blocks = find_blocks(f_best, threshold=1e-10)
    conv_np, plateau_mask, dip_positions, max_val = analyze_plateau(f_best, top_frac=0.0001)

    if len(dip_positions) > 0:
        n_dips = min(200, len(dip_positions))
        print(f"  Computing Jacobian ({len(blocks)} blocks × {n_dips} dips)...")
        t0 = time.time()
        J, g_base, dips = compute_run_jacobian(f_best, blocks, dip_positions, n_dips=n_dips)
        dt = time.time() - t0
        print(f"  Jacobian computed [{dt:.0f}s]")

        # Solve with scipy
        print(f"  Solving flattening optimization...")
        alphas = solve_flattening_scipy(J, g_base, max_val, blocks, f_best)

        # Apply and check
        f_trial = apply_multipliers(f_best, blocks, alphas)
        C_trial = score_np(f_trial)
        print(f"  Linearized solution: C={C_trial:.13f} (Δ={C_trial-C_best:+.2e})")

        if C_trial > C_best:
            # Polish with Dinkelbach
            f_polished, C_polished = dinkelbach_polish(f_trial, n_outer=3, n_inner=3000, beta=1e8)
            C_final = max(C_trial, C_polished)
            f_final = f_polished if C_polished > C_trial else f_trial
            if C_final > C_best:
                C_best = C_final
                f_best = f_final
                np.save(save_file, f_best)
                print(f"  *** Jacobian flattening improved: C={C_best:.13f} ***")

        # Also try the greedy approach
        print(f"\n  Greedy flattening...")
        alphas_greedy = solve_flattening(J, g_base, max_val, blocks, f_best)
        f_trial2 = apply_multipliers(f_best, blocks, alphas_greedy)
        C_trial2 = score_np(f_trial2)
        print(f"  Greedy solution: C={C_trial2:.13f} (Δ={C_trial2-C_best:+.2e})")

        if C_trial2 > C_best:
            f_polished2, C_polished2 = dinkelbach_polish(f_trial2, n_outer=3, n_inner=3000, beta=1e8)
            C_final2 = max(C_trial2, C_polished2)
            f_final2 = f_polished2 if C_polished2 > C_trial2 else f_trial2
            if C_final2 > C_best:
                C_best = C_final2
                f_best = f_final2
                np.save(save_file, f_best)
                print(f"  *** Greedy flattening improved: C={C_best:.13f} ***")

    # ============================================================
    # Phase 4: Targeted dip surgery
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 4: Targeted dip surgery")
    print(f"{'='*60}")

    f_new, C_new = targeted_dip_surgery(f_best, n_rounds=20)
    if C_new > C_best:
        C_best = C_new
        f_best = f_new
        np.save(save_file, f_best)
        print(f"  Dip surgery improved: C={C_best:.13f}")

    # ============================================================
    # Phase 5: Adam direct optimization
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 5: Adam direct optimization (exact formula)")
    print(f"{'='*60}")

    f_new, C_new = autograd_flatten(f_best, n_iter=10000)
    if C_new > C_best:
        C_best = C_new
        f_best = f_new
        np.save(save_file, f_best)
        print(f"  Adam improved: C={C_best:.13f}")

    # ============================================================
    # Phase 6: Final Dinkelbach polish
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 6: Final Dinkelbach polish")
    print(f"{'='*60}")

    for beta in [1e7, 5e7, 1e8, 5e8, 1e9]:
        f_new, C_new = dinkelbach_polish(f_best, n_outer=5, n_inner=5000, beta=beta)
        if C_new > C_best:
            C_best = C_new
            f_best = f_new
            np.save(save_file, f_best)
            print(f"  beta={beta:.0e}: C={C_new:.13f} ***")
        else:
            print(f"  beta={beta:.0e}: no improvement")

    print(f"\n{'='*70}")
    print(f"FINAL: C = {C_best:.13f}")
    print(f"Total improvement: ΔC = {C_best - C_initial_exact:+.2e}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

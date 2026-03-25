#!/usr/bin/env python3
"""
Structural exploration for 100k autocorrelation problem.

The packet-coordinate ascent gives only ~5e-9 because individual block scalars
are already near-optimal. We need LARGER structural changes:
1. Block removal + re-optimization (delete a block, Dinkelbach polish)
2. Block splitting (cut a large block, re-optimize)
3. Block merging (merge two nearby blocks, re-optimize)
4. New block injection (add blocks in large gaps, re-optimize)
5. Global modulation (multiply by 1+ε·cos(2πkx/n), re-optimize)
6. Structure transfer from 1.6M solution
7. Stochastic multi-block perturbation + Dinkelbach

After each structural change, full Dinkelbach polish to find new local optimum.
"""

import numpy as np
import torch
import time
import sys
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Scoring and autoconvolution
# ============================================================
def gpu_score(f_t):
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

def score_np(f_np):
    return gpu_score(torch.tensor(f_np, dtype=torch.float64, device=device))

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
        best_C_inner = [0.0]

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
            if C_exact > best_C_inner[0]:
                best_C_inner[0] = C_exact
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


def quick_dinkelbach(f_np, n_inner=1000, beta=1e8):
    """Fast single-outer Dinkelbach."""
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    lam = score_np(f_np)
    best_C = lam; best_f = f_np.copy()

    w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)

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
        beta_val = 1e8
        linf_proxy = g_max * torch.exp(
            torch.logsumexp(beta_val * (conv / (g_max + 1e-18) - 1.0), dim=0) / beta_val
        )
        obj = l2sq - lam * l1 * linf_proxy
        C_exact = l2sq.item() / (l1.item() * g_max.item())
        if C_exact > best_Ci[0]:
            best_Ci[0] = C_exact
            best_w[0] = w_t.data.clone()
        return (-obj).tap(lambda x: x.backward()) if False else (lambda: ((-obj).backward(), -obj))()[-1]

    # Simpler closure
    def closure2():
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
            torch.logsumexp(1e8 * (conv / (g_max + 1e-18) - 1.0), dim=0) / 1e8
        )
        obj = l2sq - lam * l1 * linf_proxy
        C_exact = l2sq.item() / (l1.item() * g_max.item())
        if C_exact > best_Ci[0]:
            best_Ci[0] = C_exact
            best_w[0] = w_t.data.clone()
        loss = -obj
        loss.backward()
        return loss

    try:
        optimizer.step(closure2)
    except:
        pass

    f_new = (best_w[0] ** 2).cpu().numpy()
    C_new = score_np(f_new)
    if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
        return f_new, C_new
    return best_f, best_C


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
# Experiment 1: Block removal + re-optimization
# ============================================================
def try_block_removal(f_np, blocks, C_best, n_try=50):
    """Remove a block entirely, then re-optimize with Dinkelbach."""
    print(f"\n  --- Block removal ({n_try} trials) ---")
    n_improved = 0

    # Sort blocks by mass (try removing smallest first)
    block_masses = [(i, np.sum(f_np[s:e])) for i, (s, e) in enumerate(blocks)]
    block_masses.sort(key=lambda x: x[1])

    for trial, (bi, mass) in enumerate(block_masses[:n_try]):
        s, e = blocks[bi]
        f_trial = f_np.copy()
        f_trial[s:e] = 0.0

        # Quick Dinkelbach to see if this basin is better
        f_polished, C_polished = quick_dinkelbach(f_trial, n_inner=1000, beta=1e8)
        print(f"    remove block {bi} [{s}:{e}] w={e-s} m={mass:.4f}: C={C_polished:.13f} (Δ={C_polished-C_best:+.2e})")

        if C_polished > C_best:
            # Full Dinkelbach polish
            f_polished2, C_polished2 = dinkelbach_polish(f_polished, n_outer=5, n_inner=3000, beta=1e8)
            if C_polished2 > C_best:
                print(f"    *** NEW BEST: C={C_polished2:.13f} (Δ={C_polished2-C_best:+.2e}) ***")
                return f_polished2, C_polished2
            elif C_polished > C_best:
                print(f"    *** NEW BEST: C={C_polished:.13f} (Δ={C_polished-C_best:+.2e}) ***")
                return f_polished, C_polished

    return f_np, C_best


# ============================================================
# Experiment 2: Block splitting
# ============================================================
def try_block_splitting(f_np, blocks, C_best, n_try=30):
    """Split a large block into two, then re-optimize."""
    print(f"\n  --- Block splitting ({n_try} trials) ---")

    # Sort by width descending
    large_blocks = [(i, s, e, e-s) for i, (s, e) in enumerate(blocks) if e-s >= 10]
    large_blocks.sort(key=lambda x: -x[3])

    for trial, (bi, s, e, w) in enumerate(large_blocks[:n_try]):
        # Try splitting at different positions
        for split_frac in [0.3, 0.5, 0.7]:
            split_pos = s + int(w * split_frac)
            gap_width = max(1, w // 20)  # Small gap

            f_trial = f_np.copy()
            # Zero out a small gap in the middle
            gap_start = split_pos - gap_width // 2
            gap_end = gap_start + gap_width
            f_trial[gap_start:gap_end] = 0.0

            f_polished, C_polished = quick_dinkelbach(f_trial, n_inner=1000, beta=1e8)

            if C_polished > C_best:
                f_polished2, C_polished2 = dinkelbach_polish(f_polished, n_outer=3, n_inner=3000, beta=1e8)
                C_final = max(C_polished, C_polished2)
                f_final = f_polished2 if C_polished2 > C_polished else f_polished
                if C_final > C_best:
                    print(f"    split block {bi} [{s}:{e}] at {split_pos}: C={C_final:.13f} (Δ={C_final-C_best:+.2e}) ***")
                    return f_final, C_final

        if trial % 5 == 0:
            print(f"    split trial {trial}/{n_try}: no improvement yet")

    return f_np, C_best


# ============================================================
# Experiment 3: Global modulation
# ============================================================
def try_global_modulation(f_np, C_best, n_freqs=50):
    """Multiply f by (1 + ε·cos(2πkx/n)) for various k and ε."""
    print(f"\n  --- Global modulation ({n_freqs} frequencies) ---")
    n = len(f_np)
    x = np.arange(n, dtype=np.float64) / n

    for k in range(1, n_freqs + 1):
        cos_k = np.cos(2 * np.pi * k * x)

        for eps in [0.001, 0.005, 0.01, 0.05, 0.1, -0.001, -0.005, -0.01, -0.05, -0.1]:
            f_trial = f_np * (1.0 + eps * cos_k)
            f_trial = np.maximum(f_trial, 0.0)
            C_trial = score_np(f_trial)

            if C_trial > C_best:
                # Polish
                f_polished, C_polished = quick_dinkelbach(f_trial, n_inner=1000, beta=1e8)
                C_final = max(C_trial, C_polished)
                f_final = f_polished if C_polished > C_trial else f_trial
                if C_final > C_best:
                    print(f"    mod k={k} eps={eps}: C={C_final:.13f} (Δ={C_final-C_best:+.2e}) ***")
                    # Full polish
                    f_p2, C_p2 = dinkelbach_polish(f_final, n_outer=3, n_inner=3000, beta=1e8)
                    if C_p2 > C_final:
                        return f_p2, C_p2
                    return f_final, C_final

        if k % 10 == 0:
            print(f"    modulation: tested {k}/{n_freqs} frequencies, no improvement")

    return f_np, C_best


# ============================================================
# Experiment 4: Stochastic multi-block perturbation
# ============================================================
def stochastic_exploration(f_np, C_best, n_trials=200, seed=42):
    """Randomly perturb multiple blocks simultaneously, then Dinkelbach."""
    print(f"\n  --- Stochastic multi-block exploration ({n_trials} trials) ---")
    rng = np.random.RandomState(seed)
    blocks = find_blocks(f_np, threshold=1e-10)
    n_blocks = len(blocks)

    for trial in range(n_trials):
        f_trial = f_np.copy()

        # Randomly select 5-50% of blocks to perturb
        n_perturb = rng.randint(max(1, n_blocks // 20), max(2, n_blocks // 2))
        perturb_idx = rng.choice(n_blocks, size=n_perturb, replace=False)

        for bi in perturb_idx:
            s, e = blocks[bi]
            perturbation_type = rng.choice(['scale', 'noise', 'shift', 'zero', 'double'])

            if perturbation_type == 'scale':
                alpha = rng.uniform(0.5, 2.0)
                f_trial[s:e] *= alpha
            elif perturbation_type == 'noise':
                sigma = rng.uniform(0.01, 0.3) * np.mean(f_trial[s:e])
                f_trial[s:e] += rng.randn(e - s) * sigma
            elif perturbation_type == 'shift':
                shift = rng.randint(-3, 4)
                vals = f_trial[s:e].copy()
                f_trial[s:e] = 0.0
                ns = max(0, s + shift)
                ne = min(len(f_trial), e + shift)
                os_ = max(0, -shift)
                oe = os_ + (ne - ns)
                if oe <= len(vals) and ne > ns:
                    f_trial[ns:ne] = vals[os_:oe]
            elif perturbation_type == 'zero':
                f_trial[s:e] = 0.0
            elif perturbation_type == 'double':
                f_trial[s:e] *= 2.0

        f_trial = np.maximum(f_trial, 0.0)

        # Quick screen
        C_trial = score_np(f_trial)
        if C_trial > C_best * 0.999:  # Within 0.1% - worth polishing
            f_polished, C_polished = quick_dinkelbach(f_trial, n_inner=500, beta=1e8)
            if C_polished > C_best:
                f_p2, C_p2 = dinkelbach_polish(f_polished, n_outer=3, n_inner=3000, beta=1e8)
                C_final = max(C_polished, C_p2)
                f_final = f_p2 if C_p2 > C_polished else f_polished
                if C_final > C_best:
                    print(f"    trial {trial}: perturbed {n_perturb} blocks, C={C_final:.13f} (Δ={C_final-C_best:+.2e}) ***")
                    return f_final, C_final

        if (trial + 1) % 20 == 0:
            print(f"    stochastic: {trial+1}/{n_trials} trials, no improvement yet")

    return f_np, C_best


# ============================================================
# Experiment 5: Structure transfer from 1.6M
# ============================================================
def try_structure_transfer(f_100k, C_best):
    """Transfer block structure from 1.6M solution to 100k."""
    print(f"\n  --- Structure transfer from 1.6M ---")

    if not os.path.exists('best_1600k.npy'):
        print("    best_1600k.npy not found, skipping")
        return f_100k, C_best

    f_16 = np.load('best_1600k.npy').astype(np.float64)
    n16 = len(f_16)
    n100 = len(f_100k)
    ratio = n16 / n100  # 16.0

    # Method 1: Simple subsampling
    print("    Method 1: Subsampling...")
    indices = np.linspace(0, n16 - 1, n100).astype(int)
    f_sub = f_16[indices]
    f_sub = np.maximum(f_sub, 0.0)
    # Normalize to same L1 as original
    if np.sum(f_sub) > 0:
        f_sub *= np.sum(f_100k) / np.sum(f_sub)
    C_sub = score_np(f_sub)
    print(f"    Subsampled: C={C_sub:.13f}")

    if C_sub > 0.95:  # Worth polishing
        f_pol, C_pol = dinkelbach_polish(f_sub, n_outer=5, n_inner=5000, beta=1e8)
        print(f"    After Dinkelbach: C={C_pol:.13f}")
        if C_pol > C_best:
            print(f"    *** NEW BEST from transfer: {C_pol:.13f} ***")
            return f_pol, C_pol

    # Method 2: Block averaging
    print("    Method 2: Block averaging...")
    f_avg = np.zeros(n100)
    for i in range(n100):
        lo = int(i * ratio)
        hi = min(n16, int((i + 1) * ratio))
        f_avg[i] = np.mean(f_16[lo:hi])
    f_avg = np.maximum(f_avg, 0.0)
    if np.sum(f_avg) > 0:
        f_avg *= np.sum(f_100k) / np.sum(f_avg)
    C_avg = score_np(f_avg)
    print(f"    Block-averaged: C={C_avg:.13f}")

    if C_avg > 0.95:
        f_pol, C_pol = dinkelbach_polish(f_avg, n_outer=5, n_inner=5000, beta=1e8)
        print(f"    After Dinkelbach: C={C_pol:.13f}")
        if C_pol > C_best:
            print(f"    *** NEW BEST from transfer: {C_pol:.13f} ***")
            return f_pol, C_pol

    # Method 3: Hybrid — use 1.6M structure as support hint for 100k
    print("    Method 3: Support hint from 1.6M...")
    blocks_16 = find_blocks(f_16, threshold=1e-10)
    # Map block positions to 100k scale
    f_hybrid = np.zeros(n100)
    for s16, e16 in blocks_16:
        s100 = int(s16 / ratio)
        e100 = min(n100, int(e16 / ratio) + 1)
        if e100 > s100:
            # Use original 100k values where they exist, otherwise interpolate
            for j in range(s100, e100):
                if f_100k[j] > 1e-10:
                    f_hybrid[j] = f_100k[j]
                else:
                    # Interpolate from 1.6M
                    j16 = int(j * ratio)
                    f_hybrid[j] = f_16[min(j16, n16-1)] * (np.sum(f_100k) / np.sum(f_16)) if np.sum(f_16) > 0 else 0
    # Add back original support
    mask = f_100k > 1e-10
    f_hybrid[mask] = np.maximum(f_hybrid[mask], f_100k[mask])
    f_hybrid = np.maximum(f_hybrid, 0.0)
    if np.sum(f_hybrid) > 0:
        f_hybrid *= np.sum(f_100k) / np.sum(f_hybrid)
    C_hyb = score_np(f_hybrid)
    print(f"    Hybrid: C={C_hyb:.13f}")

    if C_hyb > 0.95:
        f_pol, C_pol = dinkelbach_polish(f_hybrid, n_outer=5, n_inner=5000, beta=1e8)
        print(f"    After Dinkelbach: C={C_pol:.13f}")
        if C_pol > C_best:
            print(f"    *** NEW BEST from transfer: {C_pol:.13f} ***")
            return f_pol, C_pol

    return f_100k, C_best


# ============================================================
# Experiment 6: Exp parameterization with noise restart
# ============================================================
def exp_noise_restart(f_np, C_best, n_trials=20, seed=123):
    """Restart from log(f) + noise in exp parameterization."""
    print(f"\n  --- Exp-parameterization noise restarts ({n_trials} trials) ---")
    rng = np.random.RandomState(seed)
    n = len(f_np); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    # Compute log(f) for nonzero positions
    mask = f_np > 1e-20
    v_base = np.full(n, -50.0)  # log(tiny) for zeros
    v_base[mask] = np.log(f_np[mask])

    for trial in range(n_trials):
        # Add noise to v
        sigma = rng.uniform(0.01, 0.5)
        v = v_base.copy()
        v[mask] += rng.randn(np.sum(mask)) * sigma

        # Also try adding noise to zero positions with small prob
        zero_activate = rng.random(n) < 0.01  # 1% chance
        v[~mask & zero_activate] = rng.uniform(-10, -2, size=np.sum(~mask & zero_activate))

        f_trial = np.exp(v)
        f_trial = np.maximum(f_trial, 0.0)
        # Clip very large values
        f_trial = np.minimum(f_trial, f_np.max() * 10)

        # Optimize with exp parameterization via Dinkelbach
        lam = score_np(f_trial)
        v_t = torch.tensor(v, dtype=torch.float64, device=device).requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [v_t], lr=1.0, max_iter=2000,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_v = [v_t.data.clone()]
        best_Ci = [0.0]
        beta = 1e8

        def closure():
            optimizer.zero_grad()
            f = torch.exp(v_t)
            f = torch.clamp(f, min=0.0, max=1e4)
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
            obj = l2sq - lam * l1 * linf_proxy
            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_Ci[0]:
                best_Ci[0] = C_exact
                best_v[0] = v_t.data.clone()
            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except:
            pass

        f_new = torch.exp(best_v[0]).clamp(min=0.0, max=1e4).cpu().numpy()
        C_new = score_np(f_new)

        if C_new > C_best:
            # Full w² Dinkelbach polish
            f_pol, C_pol = dinkelbach_polish(f_new, n_outer=3, n_inner=3000, beta=1e8)
            C_final = max(C_new, C_pol)
            f_final = f_pol if C_pol > C_new else f_new
            if C_final > C_best:
                print(f"    trial {trial} σ={sigma:.3f}: C={C_final:.13f} (Δ={C_final-C_best:+.2e}) ***")
                return f_final, C_final

        if (trial + 1) % 5 == 0:
            print(f"    exp restart: {trial+1}/{n_trials}, best_this_round={best_Ci[0]:.13f}")

    return f_np, C_best


# ============================================================
# Experiment 7: Support swap between mirror positions
# ============================================================
def try_support_swap(f_np, C_best, n_trials=100):
    """Swap support between two random positions and polish."""
    print(f"\n  --- Support swap ({n_trials} trials) ---")
    rng = np.random.RandomState(456)
    n = len(f_np)
    nz_idx = np.where(f_np > 1e-10)[0]
    z_idx = np.where(f_np <= 1e-10)[0]

    if len(nz_idx) == 0 or len(z_idx) == 0:
        return f_np, C_best

    for trial in range(n_trials):
        f_trial = f_np.copy()

        # Select random nonzero and zero positions
        n_swap = rng.randint(1, min(20, len(nz_idx) // 10, len(z_idx)))
        nz_sel = rng.choice(nz_idx, size=n_swap, replace=False)
        z_sel = rng.choice(z_idx, size=n_swap, replace=False)

        # Move mass from nonzero to zero positions
        for nzi, zi in zip(nz_sel, z_sel):
            f_trial[zi] = f_trial[nzi]
            f_trial[nzi] = 0.0

        C_trial = score_np(f_trial)

        if C_trial > C_best * 0.999:
            f_pol, C_pol = quick_dinkelbach(f_trial, n_inner=500, beta=1e8)
            if C_pol > C_best:
                f_p2, C_p2 = dinkelbach_polish(f_pol, n_outer=3, n_inner=3000, beta=1e8)
                C_final = max(C_pol, C_p2)
                f_final = f_p2 if C_p2 > C_pol else f_pol
                if C_final > C_best:
                    print(f"    swap trial {trial}: {n_swap} swaps, C={C_final:.13f} (Δ={C_final-C_best:+.2e}) ***")
                    return f_final, C_final

        if (trial + 1) % 20 == 0:
            print(f"    swap: {trial+1}/{n_trials}, no improvement yet")

    return f_np, C_best


# ============================================================
# Main
# ============================================================
def main():
    save_file = 'best_impevo_100k.npy'

    f = np.maximum(np.load(save_file).astype(np.float64), 0.0)
    n = len(f)
    C_initial = score_np(f)
    print(f"=" * 70)
    print(f"Structural exploration for 100k")
    print(f"=" * 70)
    print(f"Starting: n={n}, C = {C_initial:.13f}")

    blocks = find_blocks(f, threshold=1e-10)
    print(f"Blocks: {len(blocks)}")

    C_best = C_initial
    f_best = f.copy()

    experiments = [
        ("Global modulation", lambda f, C: try_global_modulation(f, C, n_freqs=100)),
        ("Stochastic multi-block", lambda f, C: stochastic_exploration(f, C, n_trials=200)),
        ("Block removal", lambda f, C: try_block_removal(f, find_blocks(f, 1e-10), C, n_try=50)),
        ("Block splitting", lambda f, C: try_block_splitting(f, find_blocks(f, 1e-10), C, n_try=30)),
        ("Support swap", lambda f, C: try_support_swap(f, C, n_trials=100)),
        ("Exp noise restart", lambda f, C: exp_noise_restart(f, C, n_trials=20)),
        ("Structure transfer", lambda f, C: try_structure_transfer(f, C)),
    ]

    for name, exp_fn in experiments:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")

        f_new, C_new = exp_fn(f_best, C_best)
        dt = time.time() - t0

        if C_new > C_best:
            C_best = C_new
            f_best = f_new.copy()
            np.save(save_file, f_best)
            print(f"\n  *** {name}: NEW BEST C = {C_best:.13f} (Δ={C_best-C_initial:+.2e}) [{dt:.0f}s] ***")
        else:
            print(f"\n  {name}: no improvement [{dt:.0f}s]")

    # Final mega-polish
    print(f"\n{'='*60}")
    print(f"Final Dinkelbach polish cascade")
    print(f"{'='*60}")
    for beta in [1e7, 5e7, 1e8, 5e8, 1e9, 5e9]:
        f_new, C_new = dinkelbach_polish(f_best, n_outer=5, n_inner=5000, beta=beta)
        if C_new > C_best:
            C_best = C_new
            f_best = f_new
            np.save(save_file, f_best)
            print(f"  beta={beta:.0e}: C={C_new:.13f} (Δ={C_new-C_initial:+.2e})")
        else:
            print(f"  beta={beta:.0e}: no improvement")

    print(f"\n{'='*70}")
    print(f"FINAL: C = {C_best:.13f}")
    print(f"Total improvement: ΔC = {C_best - C_initial:+.2e}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

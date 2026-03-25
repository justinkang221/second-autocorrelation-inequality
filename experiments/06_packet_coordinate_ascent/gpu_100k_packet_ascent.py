#!/usr/bin/env python3
"""
Packet-coordinate ascent for 100k autocorrelation problem.

Key idea from Einstein Arena discussion: represent f as blocks (packets)
of consecutive nonzero values, and do line search over each block's
scalar multiplier. Documented to give ~1.6e-5 improvement.

After packet ascent cycles, polish with full Dinkelbach.
Also try support modifications (extend/contract blocks, add new teeth).
"""

import numpy as np
import torch
import time
import sys
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Scoring
# ============================================================
def gpu_score(f_t):
    """Compute C(f) using Simpson's rule for L2^2."""
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


def gpu_autoconv(f_t):
    """Compute autoconvolution g = f*f."""
    n = f_t.shape[0]; nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F * F, n=nfft)[:nc]
    return conv


# ============================================================
# Block identification
# ============================================================
def find_blocks(f, threshold=1e-10):
    """Find consecutive blocks of nonzero values."""
    mask = f > threshold
    blocks = []
    in_block = False
    for i in range(len(f)):
        if mask[i] and not in_block:
            in_block = True; start = i
        elif not mask[i] and in_block:
            in_block = False
            blocks.append((start, i))  # [start, end) exclusive
    if in_block:
        blocks.append((start, len(f)))
    return blocks


# ============================================================
# Packet-coordinate ascent: line search over block scalars
# ============================================================
def packet_line_search(f_t, block_start, block_end, n_eval=30):
    """Find optimal scalar multiplier for a single block.

    Returns (best_alpha, best_C).
    """
    original_block = f_t[block_start:block_end].clone()
    block_mass = torch.sum(original_block).item()
    if block_mass < 1e-15:
        return 1.0, gpu_score(f_t)

    # Golden section search in [0.5, 2.0] first, then refine
    # Actually use scipy-style bracket search
    best_alpha = 1.0
    best_C = gpu_score(f_t)

    # Test a range of alphas
    alphas = np.concatenate([
        np.linspace(0.8, 1.2, 15),
        np.linspace(0.5, 2.0, 10),
        [0.0, 0.3, 3.0, 5.0],  # extreme values
    ])
    alphas = np.unique(alphas)

    for alpha in alphas:
        f_t[block_start:block_end] = original_block * alpha
        C = gpu_score(f_t)
        if C > best_C:
            best_C = C
            best_alpha = alpha

    # Refine around best alpha with golden section
    lo, hi = max(0.0, best_alpha - 0.1), best_alpha + 0.1
    for _ in range(15):
        if hi - lo < 1e-8:
            break
        m1 = lo + 0.382 * (hi - lo)
        m2 = lo + 0.618 * (hi - lo)
        f_t[block_start:block_end] = original_block * m1
        C1 = gpu_score(f_t)
        f_t[block_start:block_end] = original_block * m2
        C2 = gpu_score(f_t)
        if C1 > C2:
            hi = m2
            if C1 > best_C:
                best_C = C1; best_alpha = m1
        else:
            lo = m1
            if C2 > best_C:
                best_C = C2; best_alpha = m2

    # Restore and apply best
    f_t[block_start:block_end] = original_block * best_alpha
    return best_alpha, best_C


def packet_ascent_cycle(f_np, blocks, verbose=True):
    """One full cycle of packet-coordinate ascent over all blocks."""
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    C_start = gpu_score(f_t)
    n_improved = 0
    total_gain = 0.0

    for i, (s, e) in enumerate(blocks):
        C_before = gpu_score(f_t)
        alpha, C_after = packet_line_search(f_t, s, e)
        if C_after > C_before + 1e-15:
            n_improved += 1
            total_gain += C_after - C_before
            if verbose and (C_after - C_before > 1e-10):
                print(f"    block {i} [{s}:{e}] w={e-s}: α={alpha:.6f}, ΔC={C_after-C_before:+.2e}")
        else:
            # Restore original
            f_t[s:e] = torch.tensor(f_np[s:e], dtype=torch.float64, device=device)

    C_end = gpu_score(f_t)
    f_out = f_t.cpu().numpy()
    f_out = np.maximum(f_out, 0.0)
    return f_out, C_end, n_improved, total_gain


# ============================================================
# Support modification: extend/contract blocks
# ============================================================
def try_extend_blocks(f_np, blocks, verbose=True):
    """Try extending each block by 1 position on each side."""
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    C_best = gpu_score(f_t)
    f_best = f_np.copy()
    n_improved = 0

    for i, (s, e) in enumerate(blocks):
        # Try extending left
        if s > 0 and f_np[s-1] < 1e-10:
            f_trial = f_best.copy()
            # Set new position to average of block boundary
            val = f_best[s] * 0.5
            f_trial[s-1] = val
            f_t_trial = torch.tensor(f_trial, dtype=torch.float64, device=device)
            C_trial = gpu_score(f_t_trial)
            if C_trial > C_best:
                if verbose:
                    print(f"    extend_left block {i} [{s}:{e}]: ΔC={C_trial-C_best:+.2e}")
                C_best = C_trial
                f_best = f_trial
                n_improved += 1

        # Try extending right
        if e < len(f_np) and f_np[e] < 1e-10:
            f_trial = f_best.copy()
            val = f_best[e-1] * 0.5
            f_trial[e] = val
            f_t_trial = torch.tensor(f_trial, dtype=torch.float64, device=device)
            C_trial = gpu_score(f_t_trial)
            if C_trial > C_best:
                if verbose:
                    print(f"    extend_right block {i} [{s}:{e}]: ΔC={C_trial-C_best:+.2e}")
                C_best = C_trial
                f_best = f_trial
                n_improved += 1

    return f_best, C_best, n_improved


def try_contract_blocks(f_np, blocks, verbose=True):
    """Try removing boundary positions from each block."""
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    C_best = gpu_score(f_t)
    f_best = f_np.copy()
    n_improved = 0

    for i, (s, e) in enumerate(blocks):
        if e - s <= 1:
            continue

        # Try removing leftmost position
        f_trial = f_best.copy()
        f_trial[s] = 0.0
        f_t_trial = torch.tensor(f_trial, dtype=torch.float64, device=device)
        C_trial = gpu_score(f_t_trial)
        if C_trial > C_best:
            if verbose:
                print(f"    contract_left block {i} [{s}:{e}]: ΔC={C_trial-C_best:+.2e}")
            C_best = C_trial
            f_best = f_trial
            n_improved += 1

        # Try removing rightmost position
        f_trial = f_best.copy()
        f_trial[e-1] = 0.0
        f_t_trial = torch.tensor(f_trial, dtype=torch.float64, device=device)
        C_trial = gpu_score(f_t_trial)
        if C_trial > C_best:
            if verbose:
                print(f"    contract_right block {i} [{s}:{e}]: ΔC={C_trial-C_best:+.2e}")
            C_best = C_trial
            f_best = f_trial
            n_improved += 1

    return f_best, C_best, n_improved


# ============================================================
# Support modification: add new teeth in gaps
# ============================================================
def try_add_teeth(f_np, blocks, verbose=True):
    """Try adding small teeth in the middle of large gaps."""
    f_best = f_np.copy()
    C_best = gpu_score(torch.tensor(f_best, dtype=torch.float64, device=device))
    n_improved = 0

    # Find gaps between blocks
    gaps = []
    for i in range(len(blocks) - 1):
        gap_start = blocks[i][1]
        gap_end = blocks[i+1][0]
        gap_size = gap_end - gap_start
        if gap_size >= 3:  # Only try in gaps large enough
            gaps.append((gap_start, gap_end, gap_size))

    # Sort by gap size descending
    gaps.sort(key=lambda x: -x[2])

    # Average nonzero value for setting tooth heights
    nz_vals = f_np[f_np > 1e-10]
    mean_val = np.median(nz_vals) if len(nz_vals) > 0 else 1.0

    for gap_start, gap_end, gap_size in gaps[:100]:  # Try top 100 largest gaps
        mid = (gap_start + gap_end) // 2

        # Try different tooth sizes and heights
        for width in [1, 3, 5]:
            for height_frac in [0.1, 0.3, 0.5, 1.0]:
                tooth_start = max(gap_start, mid - width // 2)
                tooth_end = min(gap_end, tooth_start + width)

                f_trial = f_best.copy()
                height = mean_val * height_frac
                f_trial[tooth_start:tooth_end] = height

                f_t_trial = torch.tensor(f_trial, dtype=torch.float64, device=device)
                C_trial = gpu_score(f_t_trial)
                if C_trial > C_best:
                    if verbose:
                        print(f"    add_tooth gap [{gap_start}:{gap_end}] w={width} h={height:.4f}: ΔC={C_trial-C_best:+.2e}")
                    C_best = C_trial
                    f_best = f_trial
                    n_improved += 1

    return f_best, C_best, n_improved


# ============================================================
# Dinkelbach polish
# ============================================================
def dinkelbach_polish(f_np, n_outer=5, n_inner=3000, beta=1e8):
    """Full Dinkelbach polish with L-BFGS."""
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    lam = gpu_score(f_t)
    best_C = lam; best_f = f_np.copy()

    for outer in range(n_outer):
        w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [w_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_w_inner = [w_t.data.clone()]
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
        C_new = gpu_score(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new
        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new
            best_f = f_new.copy()
            print(f"    Dinkelbach outer={outer}: C={C_new:.13f}")
        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


# ============================================================
# Mass redistribution between blocks
# ============================================================
def try_mass_transfer(f_np, blocks, verbose=True):
    """Try transferring mass between nearby block pairs."""
    f_best = f_np.copy()
    C_best = gpu_score(torch.tensor(f_best, dtype=torch.float64, device=device))
    n_improved = 0

    for i in range(len(blocks) - 1):
        s1, e1 = blocks[i]
        s2, e2 = blocks[i+1]

        mass1 = np.sum(f_best[s1:e1])
        mass2 = np.sum(f_best[s2:e2])
        total_mass = mass1 + mass2
        if total_mass < 1e-10:
            continue

        # Try different mass ratios
        for frac in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]:
            f_trial = f_best.copy()
            # Scale block i by frac, adjust block i+1 to conserve total mass
            new_mass1 = mass1 * frac
            if new_mass1 > total_mass * 0.99:
                continue
            scale2 = (total_mass - new_mass1) / mass2 if mass2 > 1e-10 else 0
            if scale2 < 0:
                continue
            f_trial[s1:e1] = f_best[s1:e1] * frac
            f_trial[s2:e2] = f_best[s2:e2] * scale2

            C_trial = gpu_score(torch.tensor(f_trial, dtype=torch.float64, device=device))
            if C_trial > C_best:
                if verbose:
                    print(f"    mass_transfer blocks {i},{i+1}: frac={frac:.1f}, ΔC={C_trial-C_best:+.2e}")
                C_best = C_trial
                f_best = f_trial
                n_improved += 1

    return f_best, C_best, n_improved


# ============================================================
# Argmax analysis: understand which conv index achieves max
# ============================================================
def analyze_autoconv_max(f_np):
    """Analyze the autoconvolution maximum and near-max region."""
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    conv = gpu_autoconv(f_t).cpu().numpy()

    max_val = np.max(conv)
    argmax = np.argmax(conv)
    near_max = np.sum(conv > 0.999 * max_val)
    near_max_99 = np.sum(conv > 0.99 * max_val)

    print(f"  Autoconv: max={max_val:.6f} at pos={argmax}, near-max(0.1%)={near_max}, near-max(1%)={near_max_99}")

    # Top 10 positions
    top10 = np.argsort(conv)[-10:][::-1]
    for j, idx in enumerate(top10):
        print(f"    #{j+1}: pos={idx}, val={conv[idx]:.6f} ({conv[idx]/max_val*100:.4f}%)")

    return argmax, max_val


# ============================================================
# Advanced: per-element gradient ascent on block values
# ============================================================
def block_gradient_ascent(f_np, blocks, n_steps=100, lr=1e-4):
    """Gradient ascent on individual values within each block."""
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    C_start = gpu_score(f_t)
    f_best = f_np.copy()
    C_best = C_start

    n = len(f_np); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    for bi, (s, e) in enumerate(blocks):
        if e - s < 2:
            continue

        # Optimize this block's values with Adam
        block_vals = torch.tensor(f_np[s:e], dtype=torch.float64, device=device, requires_grad=True)
        opt = torch.optim.Adam([block_vals], lr=lr)

        f_fixed = torch.tensor(f_best, dtype=torch.float64, device=device)

        for step in range(n_steps):
            opt.zero_grad()
            f_full = f_fixed.clone()
            f_full[s:e] = torch.clamp(block_vals, min=0.0)

            F = torch.fft.rfft(f_full, n=nfft)
            conv = torch.fft.irfft(F * F, n=nfft)[:nc]
            conv = torch.maximum(conv, torch.zeros(1, device=device, dtype=torch.float64))

            hh = 1.0 / (nc + 1)
            zz = torch.zeros(1, device=device, dtype=torch.float64)
            y = torch.cat([zz, conv, zz])
            y0, y1 = y[:-1], y[1:]
            l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
            l1 = torch.sum(conv) / (nc + 1)
            linf = torch.max(conv)

            C = l2sq / (l1 * linf)
            loss = -C
            loss.backward()
            opt.step()

        # Check if improvement
        with torch.no_grad():
            f_trial = f_best.copy()
            f_trial[s:e] = torch.clamp(block_vals, min=0.0).detach().cpu().numpy()
            C_trial = gpu_score(torch.tensor(f_trial, dtype=torch.float64, device=device))
            if C_trial > C_best:
                C_best = C_trial
                f_best = f_trial
                if C_trial - C_start > 1e-12:
                    print(f"    block_grad {bi} [{s}:{e}]: ΔC={C_trial-C_start:+.2e}")

    return f_best, C_best


# ============================================================
# Main
# ============================================================
def main():
    save_file = 'best_impevo_100k.npy'

    # Load best solution
    f = np.maximum(np.load(save_file).astype(np.float64), 0.0)
    n = len(f)
    C_initial = gpu_score(torch.tensor(f, dtype=torch.float64, device=device))
    print(f"=" * 70)
    print(f"Packet-coordinate ascent for 100k")
    print(f"=" * 70)
    print(f"Starting: n={n}, C = {C_initial:.13f}")

    # Save backup
    backup_file = f'backup_100k_{C_initial:.10f}.npy'
    if not os.path.exists(backup_file):
        np.save(backup_file, f)
        print(f"Backup saved to {backup_file}")

    # Analyze structure
    blocks = find_blocks(f, threshold=1e-10)
    print(f"Blocks: {len(blocks)} (threshold=1e-10)")

    # Analyze autoconv maximum
    analyze_autoconv_max(f)

    C_best_overall = C_initial
    f_best_overall = f.copy()

    # ============================================================
    # Phase 1: Packet-coordinate ascent cycles
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Packet-coordinate ascent")
    print(f"{'='*60}")

    for cycle in range(20):
        t0 = time.time()
        blocks = find_blocks(f_best_overall, threshold=1e-10)
        print(f"\n  Cycle {cycle} ({len(blocks)} blocks)...")

        f_new, C_new, n_impr, gain = packet_ascent_cycle(f_best_overall, blocks, verbose=True)
        dt = time.time() - t0

        print(f"  Cycle {cycle}: C={C_new:.13f} (ΔC={C_new-C_best_overall:+.2e}), {n_impr} blocks improved [{dt:.0f}s]")

        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)

        if n_impr == 0 or gain < 1e-14:
            print("  No improvement, moving to next phase.")
            break

    # ============================================================
    # Phase 2: Support modifications
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Support modifications")
    print(f"{'='*60}")

    for mod_round in range(5):
        print(f"\n  --- Modification round {mod_round} ---")
        blocks = find_blocks(f_best_overall, threshold=1e-10)
        any_improved = False

        # 2a: Try extending blocks
        print(f"  Extending blocks...")
        f_new, C_new, n_ext = try_extend_blocks(f_best_overall, blocks, verbose=True)
        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)
            any_improved = True
            print(f"  After extend: C={C_new:.13f} ({n_ext} extensions)")

        # 2b: Try contracting blocks
        blocks = find_blocks(f_best_overall, threshold=1e-10)
        print(f"  Contracting blocks...")
        f_new, C_new, n_con = try_contract_blocks(f_best_overall, blocks, verbose=True)
        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)
            any_improved = True
            print(f"  After contract: C={C_new:.13f} ({n_con} contractions)")

        # 2c: Try adding teeth in gaps
        blocks = find_blocks(f_best_overall, threshold=1e-10)
        print(f"  Adding teeth in gaps...")
        f_new, C_new, n_teeth = try_add_teeth(f_best_overall, blocks, verbose=True)
        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)
            any_improved = True
            print(f"  After teeth: C={C_new:.13f} ({n_teeth} new teeth)")

        # 2d: Try mass transfer between adjacent blocks
        blocks = find_blocks(f_best_overall, threshold=1e-10)
        print(f"  Mass transfer...")
        f_new, C_new, n_xfer = try_mass_transfer(f_best_overall, blocks, verbose=True)
        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)
            any_improved = True
            print(f"  After transfer: C={C_new:.13f} ({n_xfer} transfers)")

        # If any support change helped, do another packet ascent cycle
        if any_improved:
            print(f"\n  Re-running packet ascent after support changes...")
            blocks = find_blocks(f_best_overall, threshold=1e-10)
            f_new, C_new, n_impr, gain = packet_ascent_cycle(f_best_overall, blocks, verbose=True)
            if C_new > C_best_overall:
                C_best_overall = C_new
                f_best_overall = f_new
                np.save(save_file, f_best_overall)
                print(f"  Post-support packet ascent: C={C_new:.13f}")

        if not any_improved:
            print("  No support modifications helped.")
            break

    # ============================================================
    # Phase 3: Dinkelbach polish at multiple betas
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 3: Dinkelbach polish")
    print(f"{'='*60}")

    for beta in [1e7, 5e7, 1e8, 5e8, 1e9]:
        print(f"\n  Beta = {beta:.0e}...")
        f_new, C_new = dinkelbach_polish(f_best_overall, n_outer=5, n_inner=5000, beta=beta)
        print(f"  Beta={beta:.0e}: C={C_new:.13f} (ΔC={C_new-C_best_overall:+.2e})")
        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)

    # ============================================================
    # Phase 4: Alternating packet ascent + Dinkelbach
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 4: Alternating packet ascent + Dinkelbach")
    print(f"{'='*60}")

    for alt_round in range(10):
        C_before = C_best_overall

        # Packet ascent
        blocks = find_blocks(f_best_overall, threshold=1e-10)
        f_new, C_new, n_impr, gain = packet_ascent_cycle(f_best_overall, blocks, verbose=False)
        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)

        # Dinkelbach
        f_new, C_new = dinkelbach_polish(f_best_overall, n_outer=3, n_inner=3000, beta=1e8)
        if C_new > C_best_overall:
            C_best_overall = C_new
            f_best_overall = f_new
            np.save(save_file, f_best_overall)

        print(f"  Alt round {alt_round}: C={C_best_overall:.13f} (ΔC={C_best_overall-C_before:+.2e})")

        if C_best_overall - C_before < 1e-13:
            print("  Converged.")
            break

    # ============================================================
    # Phase 5: Large-scale block gradient ascent
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 5: Block gradient ascent (Adam)")
    print(f"{'='*60}")

    # Only do this for the largest blocks
    blocks = find_blocks(f_best_overall, threshold=1e-10)
    large_blocks = [(s, e) for s, e in blocks if e - s >= 20]
    print(f"  {len(large_blocks)} large blocks (width >= 20)")

    f_new, C_new = block_gradient_ascent(f_best_overall, large_blocks, n_steps=200, lr=1e-5)
    if C_new > C_best_overall:
        print(f"  Block gradient: C={C_new:.13f} (ΔC={C_new-C_best_overall:+.2e})")
        C_best_overall = C_new
        f_best_overall = f_new
        np.save(save_file, f_best_overall)

    # Final Dinkelbach polish
    print(f"\n  Final Dinkelbach polish...")
    f_new, C_new = dinkelbach_polish(f_best_overall, n_outer=5, n_inner=5000, beta=5e8)
    if C_new > C_best_overall:
        C_best_overall = C_new
        f_best_overall = f_new
        np.save(save_file, f_best_overall)

    print(f"\n{'='*70}")
    print(f"FINAL: C = {C_best_overall:.13f}")
    print(f"Total improvement: ΔC = {C_best_overall - C_initial:+.2e}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

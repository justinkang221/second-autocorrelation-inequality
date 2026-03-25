#!/usr/bin/env python3
"""
Sparse comb construction + Dinkelbach optimization for 100k.

Key insight: Einstein Arena discussion describes optimal 100k solutions as
~498 blocks of 3-8 values, spaced ~200 apart. Our current solution has
771 dense blocks with gaps of only 4 — a fundamentally different structure.

This script:
1. Constructs many different sparse comb structures
2. Optimizes each with Dinkelbach
3. Keeps the best

If the sparse comb basin is better than our dense basin, this could give
the ~1.4e-5 improvement we need.
"""

import numpy as np
import torch
import time
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Scoring
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
# Dinkelbach optimization (multi-beta cascade)
# ============================================================
def dinkelbach_cascade(f_np, betas=[1e6, 1e7, 5e7, 1e8], n_outer=5, n_inner=3000):
    """Full Dinkelbach with beta cascade."""
    best_f = f_np.copy()
    best_C = score_np(f_np)

    for beta in betas:
        f_new, C_new = dinkelbach_single_beta(best_f, n_outer=n_outer, n_inner=n_inner, beta=beta)
        if C_new > best_C and C_new < 1.0:
            best_C = C_new
            best_f = f_new
    return best_f, best_C


def dinkelbach_single_beta(f_np, n_outer=5, n_inner=3000, beta=1e8):
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
# Comb constructors
# ============================================================
def make_uniform_comb(n, n_teeth, tooth_width, offset=0):
    """Uniform comb: equally spaced rectangular teeth."""
    f = np.zeros(n)
    spacing = n // n_teeth
    for i in range(n_teeth):
        center = offset + i * spacing
        start = max(0, center - tooth_width // 2)
        end = min(n, start + tooth_width)
        f[start:end] = 1.0
    return f


def make_gaussian_comb(n, n_teeth, tooth_sigma, offset=0):
    """Comb with Gaussian-shaped teeth."""
    f = np.zeros(n)
    spacing = n // n_teeth
    x = np.arange(n)
    for i in range(n_teeth):
        center = offset + i * spacing
        f += np.exp(-0.5 * ((x - center) / tooth_sigma) ** 2)
    f = np.maximum(f, 0.0)
    return f


def make_random_comb(n, n_teeth, tooth_width_range=(3, 8), rng=None):
    """Random comb with variable tooth widths and slight position jitter."""
    if rng is None:
        rng = np.random.RandomState(42)
    f = np.zeros(n)
    spacing = n / n_teeth
    for i in range(n_teeth):
        center = int(i * spacing + rng.randint(-5, 6))
        center = max(0, min(n - 1, center))
        w = rng.randint(tooth_width_range[0], tooth_width_range[1] + 1)
        start = max(0, center - w // 2)
        end = min(n, start + w)
        height = rng.uniform(0.5, 1.5)
        f[start:end] = height
    return f


def make_modulated_comb(n, n_teeth, tooth_width, mod_freq=3, mod_amp=0.3):
    """Comb with sinusoidal amplitude modulation."""
    f = make_uniform_comb(n, n_teeth, tooth_width)
    x = np.arange(n) / n
    modulation = 1.0 + mod_amp * np.cos(2 * np.pi * mod_freq * x)
    f *= modulation
    return np.maximum(f, 0.0)


def make_qr_comb(n, p):
    """Quadratic residue comb: teeth at quadratic residue positions mod p."""
    f = np.zeros(n)
    residues = set()
    for k in range(p):
        residues.add((k * k) % p)
    # Map to positions in [0, n)
    scale = n / p
    for r in residues:
        pos = int(r * scale)
        if pos < n:
            w = max(1, int(scale * 0.3))
            start = max(0, pos - w // 2)
            end = min(n, start + w)
            f[start:end] = 1.0
    return f


def make_from_existing_support(f_existing, n_teeth_target):
    """Extract the top n_teeth_target blocks from existing solution."""
    n = len(f_existing)
    # Find blocks
    blocks = []
    mask = f_existing > 1e-10
    in_block = False
    for i in range(n):
        if mask[i] and not in_block:
            in_block = True; start = i
        elif not mask[i] and in_block:
            in_block = False
            blocks.append((start, i, np.sum(f_existing[start:i])))
    if in_block:
        blocks.append((start, n, np.sum(f_existing[start:n])))

    # Keep top blocks by mass
    blocks.sort(key=lambda x: -x[2])
    keep = blocks[:n_teeth_target]

    f_new = np.zeros(n)
    for s, e, _ in keep:
        f_new[s:e] = f_existing[s:e]
    return f_new


# ============================================================
# Main: sweep over comb constructions
# ============================================================
def main():
    save_file = 'best_impevo_100k.npy'

    f_current = np.maximum(np.load(save_file).astype(np.float64), 0.0)
    n = len(f_current)
    C_current = score_np(f_current)

    print(f"=" * 70)
    print(f"Sparse comb construction + Dinkelbach for 100k")
    print(f"=" * 70)
    print(f"Current best: C = {C_current:.13f}")
    print(f"Target: C > 0.9620")

    C_best = C_current
    f_best = f_current.copy()

    # ============================================================
    # Part 1: Sweep over comb parameters
    # ============================================================
    configs = []

    # Uniform combs with different tooth counts and widths
    for n_teeth in [200, 300, 400, 490, 498, 500, 550, 600, 700, 800]:
        for tooth_w in [3, 5, 7, 10, 15]:
            for offset in [0, n_teeth // 4]:
                configs.append(('uniform', {'n_teeth': n_teeth, 'tooth_width': tooth_w, 'offset': offset}))

    # Gaussian combs
    for n_teeth in [400, 490, 498, 500, 600]:
        for sigma in [1.0, 2.0, 3.0, 5.0]:
            configs.append(('gaussian', {'n_teeth': n_teeth, 'tooth_sigma': sigma}))

    # Modulated combs
    for n_teeth in [490, 498, 500]:
        for mod_freq in [1, 2, 3, 5]:
            for mod_amp in [0.1, 0.3, 0.5]:
                configs.append(('modulated', {'n_teeth': n_teeth, 'tooth_width': 5, 'mod_freq': mod_freq, 'mod_amp': mod_amp}))

    # QR combs
    for p in [499, 503, 509, 521, 541, 547]:  # primes near 500
        configs.append(('qr', {'p': p}))

    # Random combs
    for seed in range(20):
        for n_teeth in [490, 498, 500, 550]:
            configs.append(('random', {'n_teeth': n_teeth, 'seed': seed}))

    # From existing support (subsample blocks)
    for n_teeth in [100, 200, 300, 400, 498, 500]:
        configs.append(('from_existing', {'n_teeth': n_teeth}))

    print(f"\nTotal configs to test: {len(configs)}")

    best_initial = []
    t_start = time.time()

    for ci, (ctype, params) in enumerate(configs):
        if ctype == 'uniform':
            f_init = make_uniform_comb(n, **params)
        elif ctype == 'gaussian':
            f_init = make_gaussian_comb(n, **params)
        elif ctype == 'modulated':
            f_init = make_modulated_comb(n, **params)
        elif ctype == 'qr':
            f_init = make_qr_comb(n, **params)
        elif ctype == 'random':
            rng = np.random.RandomState(params['seed'])
            f_init = make_random_comb(n, params['n_teeth'], rng=rng)
        elif ctype == 'from_existing':
            f_init = make_from_existing_support(f_current, params['n_teeth'])
        else:
            continue

        f_init = np.maximum(f_init, 0.0)
        if np.sum(f_init) < 1e-10:
            continue

        C_init = score_np(f_init)

        # Quick screen: only optimize if initial C is decent
        if C_init > 0.3:
            best_initial.append((C_init, ci, ctype, params))

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  Screened {ci+1}/{len(configs)} [{elapsed:.0f}s]")

    # Sort by initial C
    best_initial.sort(key=lambda x: -x[0])
    print(f"\nTop 20 initial scores:")
    for C_init, ci, ctype, params in best_initial[:20]:
        print(f"  C={C_init:.10f} {ctype} {params}")

    # ============================================================
    # Part 2: Optimize top candidates with Dinkelbach cascade
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Part 2: Dinkelbach optimization of top candidates")
    print(f"{'='*60}")

    n_optimize = min(30, len(best_initial))
    for rank, (C_init, ci, ctype, params) in enumerate(best_initial[:n_optimize]):
        # Reconstruct
        if ctype == 'uniform':
            f_init = make_uniform_comb(n, **params)
        elif ctype == 'gaussian':
            f_init = make_gaussian_comb(n, **params)
        elif ctype == 'modulated':
            f_init = make_modulated_comb(n, **params)
        elif ctype == 'qr':
            f_init = make_qr_comb(n, **params)
        elif ctype == 'random':
            rng = np.random.RandomState(params['seed'])
            f_init = make_random_comb(n, params['n_teeth'], rng=rng)
        elif ctype == 'from_existing':
            f_init = make_from_existing_support(f_current, params['n_teeth'])
        f_init = np.maximum(f_init, 0.0)

        t0 = time.time()
        f_opt, C_opt = dinkelbach_cascade(
            f_init,
            betas=[1e5, 1e6, 1e7, 5e7, 1e8],
            n_outer=5, n_inner=3000
        )
        dt = time.time() - t0
        print(f"  #{rank}: {ctype} {params}: C_init={C_init:.6f} → C_opt={C_opt:.13f} [{dt:.0f}s]")

        if C_opt > C_best:
            C_best = C_opt
            f_best = f_opt
            np.save(save_file, f_best)
            print(f"  *** NEW BEST: C = {C_best:.13f} ***")

        # Early termination: if top candidates are very bad, stop
        if rank >= 5 and C_opt < 0.95:
            print(f"  Top candidates not reaching 0.95, skipping rest")
            break

    # ============================================================
    # Part 3: Hybrid — start from best sparse, mix with current
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Part 3: Hybrid mixing")
    print(f"{'='*60}")

    # Try mixing current dense solution with sparse structure
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        for _, ci, ctype, params in best_initial[:5]:
            # Reconstruct sparse
            if ctype == 'from_existing':
                f_sparse = make_from_existing_support(f_current, params['n_teeth'])
            elif ctype == 'uniform':
                f_sparse = make_uniform_comb(n, **params)
            else:
                continue

            f_sparse = np.maximum(f_sparse, 0.0)
            if np.sum(f_sparse) > 0:
                f_sparse *= np.sum(f_current) / np.sum(f_sparse)

            f_mix = alpha * f_current + (1 - alpha) * f_sparse
            f_mix = np.maximum(f_mix, 0.0)

            C_mix = score_np(f_mix)
            if C_mix > 0.95:
                f_opt, C_opt = dinkelbach_cascade(f_mix, betas=[1e6, 1e7, 1e8], n_outer=3, n_inner=2000)
                if C_opt > C_best:
                    C_best = C_opt
                    f_best = f_opt
                    np.save(save_file, f_best)
                    print(f"  Hybrid α={alpha} {ctype}: C={C_opt:.13f} *** NEW BEST ***")

    # ============================================================
    # Part 4: Random restart — many random combs with full optimization
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Part 4: Random restarts")
    print(f"{'='*60}")

    for seed in range(100):
        rng = np.random.RandomState(seed + 1000)
        n_teeth = rng.randint(300, 700)
        tooth_w = rng.randint(2, 12)
        jitter = rng.randint(0, max(1, n // n_teeth))

        f_init = np.zeros(n)
        spacing = n / n_teeth
        for i in range(n_teeth):
            center = int(i * spacing + rng.randint(-jitter, jitter + 1))
            center = max(0, min(n - 1, center))
            w = rng.randint(max(1, tooth_w - 2), tooth_w + 3)
            start = max(0, center - w // 2)
            end = min(n, start + w)
            height = rng.uniform(0.3, 2.0)
            f_init[start:end] = height

        C_init = score_np(f_init)

        # Quick Dinkelbach
        f_opt, C_opt = dinkelbach_single_beta(f_init, n_outer=3, n_inner=2000, beta=1e7)

        if C_opt > 0.96:
            # Full cascade
            f_opt2, C_opt2 = dinkelbach_cascade(f_opt, betas=[1e7, 5e7, 1e8], n_outer=3, n_inner=3000)
            C_final = max(C_opt, C_opt2)
            f_final = f_opt2 if C_opt2 > C_opt else f_opt
            print(f"  seed={seed}: T={n_teeth} w={tooth_w}: C_init={C_init:.6f} → C={C_final:.13f}")

            if C_final > C_best:
                C_best = C_final
                f_best = f_final
                np.save(save_file, f_best)
                print(f"  *** NEW BEST: C = {C_best:.13f} ***")
        elif seed % 20 == 0:
            print(f"  seed={seed}: T={n_teeth} w={tooth_w}: C_init={C_init:.6f} → C={C_opt:.10f}")

    print(f"\n{'='*70}")
    print(f"FINAL: C = {C_best:.13f}")
    print(f"Improvement over initial: ΔC = {C_best - C_current:+.2e}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

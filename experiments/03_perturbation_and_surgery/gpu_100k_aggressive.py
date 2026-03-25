#!/usr/bin/env python3
"""
Aggressive 100k optimizer: beta sweep + structural perturbation + Dinkelbach.
Combines ultra-beta sweep, surgery, spectral perturbation, and SA-style acceptance.
Goal: push from ~0.96187 to 0.962+
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


def dinkelbach_improve(f_np, n_outer=5, n_inner=5000, beta=1e8):
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

        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def get_autoconv(f_np):
    """Compute autoconvolution on GPU, return numpy."""
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    n = len(f_np); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F*F, n=nfft)[:nc].cpu().numpy()
    return conv


def find_teeth(f_np, min_height_frac=0.01):
    """Find tooth positions and properties."""
    f_max = np.max(f_np)
    threshold = f_max * min_height_frac
    # Find regions above threshold
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
                'start': start, 'end': i, 'peak': peak_idx,
                'height': f_np[peak_idx], 'width': i - start
            })
            in_tooth = False
    if in_tooth:
        peak_idx = start + np.argmax(f_np[start:])
        teeth.append({
            'start': start, 'end': len(f_np), 'peak': peak_idx,
            'height': f_np[peak_idx], 'width': len(f_np) - start
        })
    return teeth


def perturb_spectral(f_np, rng, intensity=1.0):
    """Perturb in Fourier space — changes global structure."""
    f = f_np.copy()
    F = np.fft.rfft(f)
    n_freq = len(F)
    # Perturb a random band of frequencies
    band_width = max(1, int(n_freq * 0.01 * intensity))
    band_start = rng.integers(0, max(1, n_freq - band_width))
    noise_re = rng.normal(0, np.abs(F[band_start:band_start+band_width]).mean() * 0.01 * intensity, size=band_width)
    noise_im = rng.normal(0, np.abs(F[band_start:band_start+band_width]).mean() * 0.01 * intensity, size=band_width)
    F[band_start:band_start+band_width] += noise_re + 1j * noise_im
    f = np.fft.irfft(F, n=len(f_np))
    return np.maximum(f, 0.0)


def perturb_tooth_scale(f_np, rng, intensity=1.0):
    """Scale random teeth up/down."""
    f = f_np.copy()
    teeth = find_teeth(f_np)
    if len(teeth) == 0:
        return f
    n_teeth_to_perturb = max(1, int(len(teeth) * 0.1 * intensity))
    idxs = rng.choice(len(teeth), size=min(n_teeth_to_perturb, len(teeth)), replace=False)
    for idx in idxs:
        tooth = teeth[idx]
        scale = 1.0 + rng.normal(0, 0.02 * intensity)
        scale = max(0.5, min(2.0, scale))
        f[tooth['start']:tooth['end']] *= scale
    return np.maximum(f, 0.0)


def perturb_tooth_width(f_np, rng, intensity=1.0):
    """Widen or narrow random teeth."""
    f = f_np.copy()
    teeth = find_teeth(f_np)
    if len(teeth) == 0:
        return f
    idx = rng.integers(len(teeth))
    tooth = teeth[idx]
    center = tooth['peak']
    lo, hi = tooth['start'], tooth['end']
    width = hi - lo

    # Extract tooth profile
    profile = f[lo:hi].copy()
    new_width = max(1, int(width * (1.0 + rng.normal(0, 0.1 * intensity))))
    new_width = max(1, min(width * 3, new_width))

    # Resample tooth to new width
    x_old = np.linspace(0, 1, width)
    x_new = np.linspace(0, 1, new_width)
    new_profile = np.interp(x_new, x_old, profile)

    # Place back
    new_center = center
    new_lo = max(0, new_center - new_width // 2)
    new_hi = min(len(f), new_lo + new_width)
    actual_width = new_hi - new_lo

    # Clear old tooth
    f[lo:hi] = 0
    # Place new tooth
    f[new_lo:new_hi] = new_profile[:actual_width]

    return np.maximum(f, 0.0)


def perturb_add_remove_tooth(f_np, rng, intensity=1.0):
    """Add or remove a tooth."""
    f = f_np.copy()
    teeth = find_teeth(f_np)

    if rng.uniform() < 0.5 and len(teeth) > 5:
        # Remove a random tooth
        idx = rng.integers(len(teeth))
        tooth = teeth[idx]
        f[tooth['start']:tooth['end']] = 0
    else:
        # Add a tooth in a gap
        if len(teeth) < 2:
            return f
        # Find largest gap between consecutive teeth
        gaps = []
        for i in range(len(teeth) - 1):
            gap_start = teeth[i]['end']
            gap_end = teeth[i+1]['start']
            if gap_end > gap_start:
                gaps.append((gap_end - gap_start, gap_start, gap_end, i))
        if len(gaps) == 0:
            return f

        # Pick gap randomly (biased toward larger gaps)
        gap_sizes = np.array([g[0] for g in gaps], dtype=float)
        gap_probs = gap_sizes / gap_sizes.sum()
        gap_idx = rng.choice(len(gaps), p=gap_probs)
        _, gap_start, gap_end, tooth_idx = gaps[gap_idx]

        # Create tooth similar to neighbors
        ref_tooth = teeth[tooth_idx]
        ref_width = ref_tooth['width']
        ref_height = ref_tooth['height']

        # Place in center of gap
        center = (gap_start + gap_end) // 2
        width = min(ref_width, gap_end - gap_start)
        height = ref_height * rng.uniform(0.8, 1.2)

        lo = max(gap_start, center - width // 2)
        hi = min(gap_end, lo + width)

        # Triangular tooth profile
        x = np.linspace(-1, 1, hi - lo)
        profile = height * np.maximum(1 - np.abs(x), 0)
        f[lo:hi] = profile

    return np.maximum(f, 0.0)


def perturb_global_shift(f_np, rng, intensity=1.0):
    """Shift entire function circularly by a small amount."""
    f = f_np.copy()
    shift = rng.integers(-max(1, int(len(f) * 0.001 * intensity)),
                          max(2, int(len(f) * 0.001 * intensity) + 1))
    f = np.roll(f, shift)
    return f


def perturb_symmetric(f_np, rng, intensity=1.0):
    """Nudge toward more symmetric function (can help autoconv flatness)."""
    f = f_np.copy()
    n = len(f)
    # Partial symmetrization
    alpha = rng.uniform(0.001, 0.01) * intensity
    f_rev = f[::-1].copy()
    f = (1 - alpha) * f + alpha * f_rev
    return np.maximum(f, 0.0)


def perturb_conv_guided(f_np, rng, intensity=1.0):
    """Guided perturbation: boost f at positions that contribute to autoconv dips."""
    f = f_np.copy()
    n = len(f)
    conv = get_autoconv(f_np)
    g_max = np.max(conv)

    # Find dips in autoconvolution
    plateau = conv / (g_max + 1e-18)
    threshold = 1.0 - 0.01 * intensity
    dip_mask = plateau < threshold
    dip_positions = np.where(dip_mask)[0]

    if len(dip_positions) == 0:
        return f

    # For each dip position in conv, boost the f values that contribute
    # conv[k] = sum_{j} f[j] * f[k-j]
    # To boost conv[k], we can boost f at positions j and k-j
    n_dips_to_fix = min(5, len(dip_positions))
    selected_dips = rng.choice(dip_positions, size=n_dips_to_fix, replace=False)

    for dip_k in selected_dips:
        # Positions in f that contribute to conv[dip_k]
        # f[j] * f[dip_k - j] for j in range(max(0, dip_k-n+1), min(n, dip_k+1))
        j_lo = max(0, dip_k - n + 1)
        j_hi = min(n, dip_k + 1)

        # Find the pair (j, dip_k-j) with largest product (most leverage)
        best_j = j_lo
        best_prod = 0
        for j in range(j_lo, j_hi):
            k_j = dip_k - j
            if 0 <= k_j < n:
                prod = f[j] * f[k_j]
                if prod > best_prod:
                    best_prod = prod
                    best_j = j

        # Boost both positions slightly
        k_j = dip_k - best_j
        boost = rng.uniform(0.001, 0.01) * intensity
        f[best_j] *= (1 + boost)
        if 0 <= k_j < n:
            f[k_j] *= (1 + boost)

    return np.maximum(f, 0.0)


def perturb_noise(f_np, rng, intensity=1.0):
    """Simple additive noise."""
    noise = rng.normal(0, np.max(f_np) * 0.001 * intensity, size=len(f_np))
    return np.maximum(f_np + noise, 0.0)


PERTURB_FUNCTIONS = [
    ('spectral', perturb_spectral),
    ('tooth_scale', perturb_tooth_scale),
    ('tooth_width', perturb_tooth_width),
    ('add_remove', perturb_add_remove_tooth),
    ('shift', perturb_global_shift),
    ('symmetric', perturb_symmetric),
    ('conv_guided', perturb_conv_guided),
    ('noise', perturb_noise),
]


if __name__ == "__main__":
    print("=" * 70)
    print("Aggressive 100k optimizer")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Load best 100k
    try:
        f_best = np.maximum(np.load('best_impevo_100k.npy').astype(np.float64), 0.0)
    except:
        f_best = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)

    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")

    teeth = find_teeth(f_best)
    print(f"Found {len(teeth)} teeth")
    sys.stdout.flush()

    rng = np.random.default_rng(123)
    n_improvements = 0
    save_file = 'best_impevo_100k.npy'

    # Phase 1: Beta sweep to recover gains
    print("\n--- Phase 1: Beta sweep ---")
    for beta in [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]:
        for n_inner in [5000, 10000]:
            # Cross-pollinate
            try:
                f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
                if len(f_shared) == n:
                    C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                    if C_shared > C_best and C_shared < 1.0:
                        C_best = C_shared; f_best = f_shared.copy()
            except: pass

            f_opt, C_opt = dinkelbach_improve(f_best, n_outer=5, n_inner=n_inner, beta=beta)
            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                C_best = C_opt; f_best = f_opt.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  beta={beta:.0e}, iters={n_inner}: NEW BEST C = {C_best:.13f} !!!")
            print(f"  beta={beta:.0e}, iters={n_inner}: C={C_opt:.13f} [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nAfter beta sweep: C = {C_best:.13f}")
    sys.stdout.flush()

    # Phase 2: Aggressive structural perturbation + Dinkelbach
    print("\n--- Phase 2: Structural perturbation + Dinkelbach ---")
    best_beta = 1e8  # Use high beta for polishing
    stall_count = 0

    for round_num in range(500):
        improved_this_round = False

        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best and C_shared < 1.0:
                    C_best = C_shared; f_best = f_shared.copy()
                    print(f"  Cross-pollinated: C = {C_best:.13f}")
        except: pass

        # Try each perturbation type at multiple intensities
        for intensity in [2.0, 1.0, 0.5, 0.1, 0.01]:
            # Pick perturbation method
            perturb_name, perturb_fn = PERTURB_FUNCTIONS[round_num % len(PERTURB_FUNCTIONS)]

            try:
                f_perturbed = perturb_fn(f_best, rng, intensity)
            except Exception:
                continue

            f_improved, C_improved = dinkelbach_improve(
                f_perturbed, n_outer=3, n_inner=5000, beta=best_beta
            )

            if C_improved > C_best and C_improved < 1.0 and np.all(np.isfinite(f_improved)) and np.max(f_improved) < 1e6:
                C_best = C_improved; f_best = f_improved.copy()
                np.save(save_file, f_best)
                n_improvements += 1
                improved_this_round = True
                print(f"  R{round_num} {perturb_name} int={intensity}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()
                break  # Move to next round on improvement

        if not improved_this_round:
            stall_count += 1
        else:
            stall_count = 0

        # Adaptive: if stalling, try more aggressive multi-perturbation
        if stall_count >= 10 and stall_count % 5 == 0:
            print(f"  R{round_num}: Stalled for {stall_count} rounds, trying combo perturbation...", flush=True)
            for _ in range(5):
                f_combo = f_best.copy()
                # Apply 2-3 random perturbations
                n_perturbs = rng.integers(2, 4)
                for _ in range(n_perturbs):
                    p_idx = rng.integers(len(PERTURB_FUNCTIONS))
                    p_name, p_fn = PERTURB_FUNCTIONS[p_idx]
                    try:
                        f_combo = p_fn(f_combo, rng, rng.uniform(0.1, 3.0))
                    except Exception:
                        pass

                f_improved, C_improved = dinkelbach_improve(
                    f_combo, n_outer=5, n_inner=10000, beta=best_beta
                )

                if C_improved > C_best and C_improved < 1.0 and np.all(np.isfinite(f_improved)) and np.max(f_improved) < 1e6:
                    C_best = C_improved; f_best = f_improved.copy()
                    np.save(save_file, f_best)
                    n_improvements += 1
                    stall_count = 0
                    print(f"  R{round_num} COMBO: NEW BEST C = {C_best:.13f} !!!")
                    sys.stdout.flush()
                    break

        if round_num % 10 == 0:
            elapsed = time.time() - t0
            print(f"  R{round_num}: C_best={C_best:.13f} [{elapsed:.0f}s, {n_improvements} impr, stall={stall_count}]",
                  flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

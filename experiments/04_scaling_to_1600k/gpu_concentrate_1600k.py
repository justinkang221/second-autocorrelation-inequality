#!/usr/bin/env python3
"""
Concentration experiment for n=1600k.
Takes the current best solution and tries concentrating it:
1. Keep only the most massive teeth
2. Redistribute mass to remaining teeth
3. Adjust envelope to be steeper
4. Optimize from concentrated starting point
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
        F = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
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

def gpu_softplus_lbfgs(f_np, maxiter=150, expand_radius=500, soft_temp=0.001):
    n = len(f_np); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    support = torch.nonzero(f_t > 1e-12).squeeze()
    if support.numel() == 0: return f_np, 0.0
    opt_mask = torch.zeros(n, dtype=torch.bool, device=device)
    opt_mask[support] = True
    for r in range(1, expand_radius+1):
        opt_mask[torch.clamp(support+r, 0, n-1)] = True
        opt_mask[torch.clamp(support-r, 0, n-1)] = True
    indices = torch.nonzero(opt_mask).squeeze()
    f_vals = f_t[indices].clone()
    f_clamped = torch.clamp(f_vals, min=1e-10, max=50.0)
    h_init = torch.log(torch.exp(f_clamped) - 1.0 + 1e-30)
    h_init[f_vals < 1e-10] = -20.0
    h_param = h_init.detach().requires_grad_(True)
    best_C = [0.0]; best_h = [h_param.data.clone()]
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
    def closure():
        optimizer.zero_grad()
        f = torch.zeros(n, device=device, dtype=torch.float64)
        f[indices] = torch.nn.functional.softplus(h_param)
        F = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
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
    f_out[indices.cpu().numpy()] = torch.nn.functional.softplus(best_h[0]).cpu().numpy()
    return f_out, best_C[0]

def gpu_dinkelbach(f_np, n_iters=10, maxiter_inner=150, soft_temp=0.001):
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
            F = torch.fft.rfft(f, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
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

def full_dk_cycle(f, expand_radius, soft_temp):
    f_sp, C_sp = gpu_softplus_lbfgs(f, maxiter=150, expand_radius=expand_radius, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C_sp > C_cur: f = f_sp
    f1, C1 = gpu_lbfgsb(f, maxiter=500, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C1 > C_cur: f = f1
    f2, C2 = gpu_dinkelbach(f, n_iters=5, maxiter_inner=150, soft_temp=soft_temp)
    C_cur = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    if C2 > C_cur: f = f2
    return f, gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))

def surgery_round(f_best, n_dips=50, scale_factor=0.001):
    n = len(f_best); nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_best, dtype=torch.float64, device=device)
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F*F, n=nfft)[:nc]
    conv_np = conv.cpu().numpy()
    max_conv = np.max(conv_np)
    plateau = conv_np > 0.5 * max_conv
    plateau_indices = np.where(plateau)[0]
    if len(plateau_indices) == 0: return f_best
    plateau_vals = conv_np[plateau_indices]
    deviation = max_conv - plateau_vals
    worst_k = min(300, len(deviation))
    worst_indices = plateau_indices[np.argsort(-deviation)[:worst_k]]
    f_new = f_best.copy()
    for conv_idx in worst_indices[:n_dips]:
        i_min = max(0, conv_idx - n + 1)
        i_max = min(n, conv_idx + 1)
        i_arr = np.arange(i_min, i_max)
        j_arr = conv_idx - i_arr
        valid = (j_arr >= 0) & (j_arr < n)
        i_arr, j_arr = i_arr[valid], j_arr[valid]
        if len(i_arr) == 0: continue
        contribs = f_best[i_arr] * f_best[j_arr]
        best_idx = np.argmax(contribs)
        best_pos = i_arr[best_idx]; j = j_arr[best_idx]
        dev_ratio = deviation[np.searchsorted(plateau_indices, conv_idx)] / max_conv
        scale = 1.0 + scale_factor * dev_ratio
        f_new[best_pos] *= scale
        if best_pos != j: f_new[j] *= scale
    return np.maximum(f_new, 0.0)


def concentrate_solution(f_np, keep_frac=0.7):
    """Concentrate mass by removing low-mass teeth and redistributing."""
    n = len(f_np)
    nonzero = np.nonzero(f_np > 1e-10)[0]
    if len(nonzero) == 0:
        return f_np

    gaps = np.diff(nonzero)
    cluster_breaks = np.where(gaps > 50)[0]
    clusters = []
    start = 0
    for b in cluster_breaks:
        clusters.append(nonzero[start:b+1])
        start = b+1
    clusters.append(nonzero[start:])

    # Sort clusters by mass
    cluster_masses = [(i, np.sum(f_np[c]), c) for i, c in enumerate(clusters)]
    cluster_masses.sort(key=lambda x: -x[1])

    # Keep top fraction by mass
    total_mass = sum(m for _, m, _ in cluster_masses)
    cumulative = 0
    keep_clusters = []
    for idx, mass, c in cluster_masses:
        keep_clusters.append((idx, mass, c))
        cumulative += mass
        if cumulative >= keep_frac * total_mass:
            break

    # Build concentrated solution
    f_new = np.zeros(n, dtype=np.float64)
    kept_mass = 0
    for idx, mass, c in keep_clusters:
        f_new[c] = f_np[c]
        kept_mass += mass

    # Scale up to compensate for removed mass
    if kept_mass > 0:
        f_new *= total_mass / kept_mass

    return f_new


def steepen_envelope(f_np, power=1.5):
    """Make the envelope steeper (amplify high values, reduce low values)."""
    n = len(f_np)
    nonzero = np.nonzero(f_np > 1e-10)[0]
    if len(nonzero) == 0:
        return f_np

    f_new = f_np.copy()
    vals = f_np[nonzero]
    max_val = np.max(vals)
    # Apply power law: (f/max)^power * max
    f_new[nonzero] = max_val * (vals / max_val) ** power
    return f_new


if __name__ == "__main__":
    print("=" * 70)
    print("Concentration experiments for n=1600k")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f_orig = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n = len(f_orig)
    C_orig = gpu_score_exact(torch.tensor(f_orig, dtype=torch.float64, device=device))
    print(f"Original: n={n}, C = {C_orig:.13f}")
    sys.stdout.flush()

    C_best = C_orig
    f_best = f_orig.copy()

    expand_schedule = [r*16 for r in [200, 500, 1000, 300, 750]]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    scale_schedule = [0.001, 0.002, 0.005, 0.0005, 0.003]

    # Experiment 1: Try different concentration levels
    print("\n--- Exp 1: Concentration levels ---")
    for keep_frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
        f_conc = concentrate_solution(f_orig, keep_frac=keep_frac)
        C_conc = gpu_score_exact(torch.tensor(f_conc, dtype=torch.float64, device=device))
        print(f"  keep={keep_frac:.1f}: C = {C_conc:.10f} (delta={C_conc-C_orig:.2e})")

        if C_conc > 0.90:  # Reasonable starting point
            # Optimize it
            f_opt, C_opt = gpu_lbfgsb(f_conc, maxiter=500, soft_temp=0.001)
            print(f"    After LBFGS: C = {C_opt:.10f}")
            f_opt, C_opt2 = gpu_dinkelbach(f_opt, n_iters=5, maxiter_inner=150, soft_temp=0.001)
            C_opt = max(C_opt, C_opt2)
            print(f"    After DK: C = {C_opt:.10f}")

            if C_opt > C_best:
                C_best = C_opt; f_best = f_opt.copy()
                np.save('best_1600k.npy', f_best)
                print(f"    *** NEW BEST: {C_best:.13f} ***")
    sys.stdout.flush()

    # Experiment 2: Envelope steepening
    print("\n--- Exp 2: Envelope steepening ---")
    for power in [1.2, 1.5, 2.0, 0.8, 0.5]:
        f_steep = steepen_envelope(f_orig, power=power)
        C_steep = gpu_score_exact(torch.tensor(f_steep, dtype=torch.float64, device=device))
        print(f"  power={power:.1f}: C = {C_steep:.10f} (delta={C_steep-C_orig:.2e})")

        if C_steep > 0.90:
            f_opt, C_opt = gpu_lbfgsb(f_steep, maxiter=500, soft_temp=0.001)
            print(f"    After LBFGS: C = {C_opt:.10f}")
            f_opt, C_opt2 = gpu_dinkelbach(f_opt, n_iters=5, maxiter_inner=150, soft_temp=0.001)
            C_opt = max(C_opt, C_opt2)
            print(f"    After DK: C = {C_opt:.10f}")

            if C_opt > C_best:
                C_best = C_opt; f_best = f_opt.copy()
                np.save('best_1600k.npy', f_best)
                print(f"    *** NEW BEST: {C_best:.13f} ***")
    sys.stdout.flush()

    # Experiment 3: Concentrate + steepen combinations
    print("\n--- Exp 3: Concentrate + steepen ---")
    for keep_frac, power in [(0.6, 1.5), (0.7, 1.3), (0.8, 1.2), (0.5, 2.0)]:
        f_conc = concentrate_solution(f_orig, keep_frac=keep_frac)
        f_mod = steepen_envelope(f_conc, power=power)
        C_mod = gpu_score_exact(torch.tensor(f_mod, dtype=torch.float64, device=device))
        print(f"  keep={keep_frac}, power={power}: C = {C_mod:.10f}")

        if C_mod > 0.90:
            f_opt, C_opt = gpu_lbfgsb(f_mod, maxiter=500, soft_temp=0.001)
            f_opt, C_opt2 = gpu_dinkelbach(f_opt, n_iters=5, maxiter_inner=150, soft_temp=0.001)
            C_opt = max(C_opt, C_opt2)
            print(f"    After opt: C = {C_opt:.10f}")

            if C_opt > C_best:
                C_best = C_opt; f_best = f_opt.copy()
                np.save('best_1600k.npy', f_best)
                print(f"    *** NEW BEST: {C_best:.13f} ***")
    sys.stdout.flush()

    # Phase 2: If we found a better starting point, run surgery on it
    if C_best > C_orig:
        print(f"\n--- Phase 2: Surgery on concentrated solution ---")
        print(f"Starting from C = {C_best:.13f}")
        for round_num in range(100):
            # Cross-pollinate
            try:
                f_shared = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
                if len(f_shared) == n:
                    C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                    if C_shared > C_best:
                        C_best = C_shared; f_best = f_shared.copy()
                        print(f"  Cross-pollinated: C = {C_best:.13f}")
            except: pass

            sf = scale_schedule[round_num % len(scale_schedule)]
            n_dips = [30, 50, 80, 100, 150][round_num % 5]
            f_work = surgery_round(f_best, n_dips=n_dips, scale_factor=sf)

            for cycle in range(6):
                temp = temps[cycle % len(temps)]
                expand = expand_schedule[cycle % len(expand_schedule)]
                f_work, C_work = full_dk_cycle(f_work, expand, temp)
                if C_work > C_best:
                    C_best = C_work; f_best = f_work.copy()
                    np.save('best_1600k.npy', f_best)
                    print(f"  Round {round_num}, cycle {cycle}: NEW BEST C = {C_best:.13f} !!!")
                    sys.stdout.flush()

            if round_num % 10 == 9:
                elapsed = time.time() - t0
                print(f"  Round {round_num}: C = {C_best:.13f} [{elapsed:.0f}s]")
                sys.stdout.flush()
    else:
        # Concentration didn't help - try surgery from original with larger perturbations
        print(f"\n--- Concentration experiments didn't help. Trying large-step surgery ---")
        for round_num in range(100):
            try:
                f_shared = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
                if len(f_shared) == n:
                    C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                    if C_shared > C_best:
                        C_best = C_shared; f_best = f_shared.copy()
                        print(f"  Cross-pollinated: C = {C_best:.13f}")
            except: pass

            # Large-step surgery
            sf = [0.005, 0.01, 0.02, 0.003, 0.015][round_num % 5]
            n_dips = [100, 150, 200, 250, 300][round_num % 5]
            f_work = surgery_round(f_best, n_dips=n_dips, scale_factor=sf)

            for cycle in range(6):
                temp = temps[cycle % len(temps)]
                expand = expand_schedule[cycle % len(expand_schedule)]
                f_work, C_work = full_dk_cycle(f_work, expand, temp)
                if C_work > C_best:
                    C_best = C_work; f_best = f_work.copy()
                    np.save('best_1600k.npy', f_best)
                    print(f"  Round {round_num}, cycle {cycle}: NEW BEST C = {C_best:.13f} !!!")
                    sys.stdout.flush()

            if round_num % 10 == 9:
                elapsed = time.time() - t0
                print(f"  Round {round_num}: C = {C_best:.13f} [{elapsed:.0f}s]")
                sys.stdout.flush()

    np.save('best_1600k.npy', f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s")

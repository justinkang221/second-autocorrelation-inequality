#!/usr/bin/env python3
"""
ImprovEvolve progressive optimizer targeting 100k for EinsteinArena.
Strategy: downsample to 6k, then progressively upscale 6k→12k→25k→50k→100k
with L-BFGS optimization at each stage. Then basin-hop at 100k.
"""
import numpy as np
import torch
import time
import sys

device = torch.device('cuda')

def gpu_score_exact(f_t):
    """Exact C(f) score."""
    n = f_t.shape[0]; nc = 2*n-1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft); conv = torch.fft.irfft(F*F, n=nfft)[:nc]
    h = 1.0/(nc+1); z = torch.zeros(1, device=device, dtype=torch.float64)
    y = torch.cat([z, conv, z]); y0, y1 = y[:-1], y[1:]
    l2sq = (h/3.0)*torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv)/(nc+1); linf = torch.max(conv)
    return (l2sq/(l1*linf)).item()


def improve(f_np, stages, beta=20000.0):
    """ImprovEvolve-style improve: progressive resolution with L-BFGS."""
    w = np.sqrt(np.maximum(f_np, 0.0))

    best_C_global = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))
    best_f_global = f_np.copy()

    for stage_idx, (res, n_iters, b) in enumerate(stages):
        if len(w) != res:
            x_old = np.linspace(0, 1, len(w))
            x_new = np.linspace(0, 1, res)
            w = np.interp(x_new, x_old, w)

        n = len(w)
        nc = 2*n - 1
        nfft = 1
        while nfft < nc: nfft <<= 1

        w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)
        best_C = [0.0]
        best_w = [w_t.data.clone()]

        optimizer = torch.optim.LBFGS(
            [w_t], lr=1.0, max_iter=n_iters,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-15, tolerance_change=1e-16
        )
        eval_count = [0]

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
                torch.logsumexp(b * (conv / (g_max + 1e-18) - 1.0), dim=0) / b
            )

            C_exact = l2sq.item() / (l1.item() * g_max.item())
            eval_count[0] += 1

            if C_exact > best_C[0]:
                best_C[0] = C_exact
                best_w[0] = w_t.data.clone()

            loss = -(l2sq / (l1 * linf_proxy + 1e-18))
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    Stage {stage_idx} exception: {e}", flush=True)

        w = best_w[0].cpu().numpy()
        f_stage = w ** 2
        C_stage = gpu_score_exact(torch.tensor(f_stage, dtype=torch.float64, device=device))

        if C_stage > best_C_global:
            best_C_global = C_stage
            best_f_global = f_stage.copy()

        print(f"    Stage {stage_idx}: res={res}, beta={b}, C={C_stage:.13f} "
              f"(evals={eval_count[0]})", flush=True)

    return best_f_global, best_C_global


def perturb(f_np, intensity, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    f = f_np.copy()
    n = len(f)
    if rng.uniform() > 0.5:
        f = f[::-1].copy()
    noise = rng.uniform(0, intensity * 0.1, size=n)
    f = np.maximum(f + noise, 0.0)
    return f


if __name__ == "__main__":
    print("=" * 70)
    print("ImprovEvolve progressive optimizer targeting 100k")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    # Load best 100k solution
    f_best_100k = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)
    n100k = len(f_best_100k)
    C_best = gpu_score_exact(torch.tensor(f_best_100k, dtype=torch.float64, device=device))
    print(f"Starting 100k solution: n={n100k}, C = {C_best:.13f}")

    # Also try from the 1.6M solution downsampled
    try:
        f_1600k = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
        x_old = np.linspace(0, 1, len(f_1600k))
        x_new = np.linspace(0, 1, 100000)
        f_from_1600k = np.maximum(np.interp(x_new, x_old, f_1600k), 0.0)
        C_from_1600k = gpu_score_exact(torch.tensor(f_from_1600k, dtype=torch.float64, device=device))
        print(f"1.6M downsampled to 100k: C = {C_from_1600k:.13f}")
        if C_from_1600k > C_best:
            f_best_100k = f_from_1600k
            C_best = C_from_1600k
            print(f"  Using 1.6M downsampled as starting point!")
    except:
        pass

    f_best = f_best_100k.copy()
    sys.stdout.flush()

    # Strategy 1: Progressive from low res (ImprovEvolve style)
    # Downsample to 6k, optimize, then progressively upscale
    print("\n--- Strategy 1: Progressive 6k→12k→25k→50k→100k ---")
    prog_stages = [
        (6250, 10000, 20000),
        (12500, 10000, 20000),
        (25000, 10000, 20000),
        (50000, 10000, 20000),
        (100000, 10000, 20000),
    ]
    f_prog, C_prog = improve(f_best, prog_stages)
    if C_prog > C_best:
        C_best = C_prog; f_best = f_prog.copy()
        np.save('best_impevo_100k.npy', f_best)
        print(f"  NEW BEST: C = {C_best:.13f} !!!")
    print(f"  After progressive: C = {C_best:.13f} [{time.time()-t0:.0f}s]")
    sys.stdout.flush()

    # Strategy 2: Direct optimization with different betas
    print("\n--- Strategy 2: Direct L-BFGS at 100k with various betas ---")
    for beta in [5000, 10000, 20000, 50000, 100000]:
        f_opt, C_opt = improve(f_best, [(100000, 10000, beta)])
        if C_opt > C_best:
            C_best = C_opt; f_best = f_opt.copy()
            np.save('best_impevo_100k.npy', f_best)
            print(f"  beta={beta}: NEW BEST C = {C_best:.13f} !!!")
        print(f"  beta={beta}: C = {C_opt:.13f} [{time.time()-t0:.0f}s]", flush=True)

    # Strategy 3: Basin-hopping with progressive improve
    print("\n--- Strategy 3: Basin-hopping with progressive improve ---")
    rng = np.random.default_rng(42)
    n_improvements = 0

    for round_num in range(30):
        for sigma in [1.0, 0.1, 0.01, 0.001]:
            f_perturbed = perturb(f_best, sigma, rng)

            # Use progressive for larger perturbations, direct for small
            if sigma >= 0.1:
                stages = [
                    (25000, 5000, 20000),
                    (50000, 5000, 20000),
                    (100000, 10000, 20000),
                ]
            else:
                stages = [(100000, 10000, 20000)]

            f_improved, C_improved = improve(f_perturbed, stages)

            if C_improved > C_best:
                C_best = C_improved; f_best = f_improved.copy()
                np.save('best_impevo_100k.npy', f_best)
                n_improvements += 1
                print(f"  Round {round_num}, sigma={sigma}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

            print(f"  Round {round_num}, sigma={sigma:.3f}: C_trial={C_improved:.13f}, "
                  f"C_best={C_best:.13f} [{time.time()-t0:.0f}s]", flush=True)

    np.save('best_impevo_100k.npy', f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

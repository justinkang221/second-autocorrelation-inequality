#!/usr/bin/env python3
"""
Block-coordinate Dinkelbach for 1.6M.
Split f into blocks, optimize one block at a time while holding others fixed.
Each sub-problem is small (~16k params) → fast L-BFGS.
Also tries: random subspace optimization, and Adam-based fast screening.
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


def block_dinkelbach(w_full, block_indices, n_inner=1000, beta=1e9):
    """
    Optimize only the block of w at block_indices, holding the rest fixed.
    w_full: full sqrt(f) vector on GPU
    block_indices: slice or index array for the block to optimize
    Returns: updated w_full, new C score
    """
    n = w_full.shape[0]; nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    # Current score as lambda
    f_full = w_full ** 2
    lam = gpu_score_exact(f_full)
    best_C = lam

    # Extract block for optimization
    w_block = w_full[block_indices].detach().clone().requires_grad_(True)
    best_w_block = w_block.data.clone()

    optimizer = torch.optim.LBFGS(
        [w_block], lr=1.0, max_iter=n_inner,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-14, tolerance_change=1e-15
    )
    best_C_inner = [0.0]
    best_w_inner = [w_block.data.clone()]

    def closure():
        optimizer.zero_grad()
        # Reconstruct full w with updated block
        w_new = w_full.detach().clone()
        w_new[block_indices] = w_block
        f = w_new ** 2

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

        obj = l2sq - lam * l1 * linf_proxy

        C_exact = l2sq.item() / (l1.item() * g_max.item())
        if C_exact > best_C_inner[0] and C_exact < 1.0:
            best_C_inner[0] = C_exact
            best_w_inner[0] = w_block.data.clone()

        loss = -obj
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        pass

    # Update full w with best block
    w_result = w_full.detach().clone()
    w_result[block_indices] = best_w_inner[0]

    f_result = w_result ** 2
    C_result = gpu_score_exact(f_result)

    return w_result, C_result


def adam_fast_screen(f_np, perturbation_fn, rng, n_steps=200, lr=1e-4, beta=1e9):
    """
    Quick screening with Adam optimizer to test if a perturbation direction is promising.
    Much faster than L-BFGS for initial screening.
    """
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)
    lam = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))

    optimizer = torch.optim.Adam([w_t], lr=lr)
    best_C = lam
    best_w = w_t.data.clone()

    for step in range(n_steps):
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

        obj = l2sq - lam * l1 * linf_proxy
        loss = -obj
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                C_exact = gpu_score_exact(w_t ** 2)
                if C_exact > best_C and C_exact < 1.0:
                    best_C = C_exact
                    best_w = w_t.data.clone()

    f_out = (best_w ** 2).cpu().numpy()
    return f_out, best_C


def full_dinkelbach(f_np, n_outer=3, n_inner=5000, beta=1e9):
    """Standard full-vector Dinkelbach."""
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
        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)):
            best_C = C_new; best_f = f_new.copy()
        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


if __name__ == "__main__":
    print("=" * 70)
    print("Block-coordinate + Adam screening for 1.6M")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f_best = np.maximum(np.load('best_1600k.npy').astype(np.float64), 0.0)
    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")
    sys.stdout.flush()

    save_file = 'best_1600k.npy'
    rng = np.random.default_rng(88)
    n_improvements = 0

    # Phase 1: Block-coordinate Dinkelbach
    # Split into blocks and cycle through them
    n_blocks = 100  # 100 blocks of 16000 each
    block_size = n // n_blocks

    print(f"\n--- Phase 1: Block-coordinate Dinkelbach ({n_blocks} blocks of {block_size}) ---")

    w_full = torch.tensor(np.sqrt(np.maximum(f_best, 0.0)), dtype=torch.float64, device=device)

    for cycle in range(5):
        cycle_improved = False
        # Randomize block order each cycle
        block_order = rng.permutation(n_blocks)

        for bi, block_idx in enumerate(block_order):
            lo = block_idx * block_size
            hi = min(n, lo + block_size)
            indices = slice(lo, hi)

            w_new, C_new = block_dinkelbach(w_full, indices, n_inner=500, beta=1e9)

            if C_new > C_best and C_new < 1.0:
                C_best = C_new
                w_full = w_new
                f_best = (w_full ** 2).cpu().numpy()
                np.save(save_file, f_best)
                n_improvements += 1
                cycle_improved = True
                print(f"  Cycle {cycle} block {bi}/{n_blocks}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

            if bi % 20 == 0:
                print(f"  Cycle {cycle} block {bi}/{n_blocks}: C_best={C_best:.13f} [{time.time()-t0:.0f}s]",
                      flush=True)

        print(f"  Cycle {cycle} done: C_best={C_best:.13f} [{time.time()-t0:.0f}s, {n_improvements} impr]",
              flush=True)

        if not cycle_improved:
            print(f"  No improvement in cycle {cycle}, moving to Phase 2", flush=True)
            break

    # Phase 2: Adam fast screening + L-BFGS polish
    print(f"\n--- Phase 2: Perturb + Adam screen + L-BFGS polish ---")

    for round_num in range(100):
        # Generate perturbation
        ptype = rng.choice(['noise', 'spectral', 'block_swap', 'scale_region'])

        f_perturbed = f_best.copy()

        if ptype == 'noise':
            sigma = rng.choice([0.1, 0.01, 0.001, 0.0001])
            noise = rng.normal(0, np.max(f_best) * sigma * 0.001, size=n)
            f_perturbed = np.maximum(f_best + noise, 0.0)

        elif ptype == 'spectral':
            F = np.fft.rfft(f_best)
            band = max(1, int(len(F) * rng.uniform(0.0001, 0.01)))
            start = rng.integers(0, max(1, len(F) - band))
            scale = rng.uniform(0.001, 0.01)
            noise_f = rng.normal(0, np.abs(F[start:start+band]).mean() * scale, size=band)
            F[start:start+band] += noise_f
            f_perturbed = np.maximum(np.fft.irfft(F, n=n), 0.0)

        elif ptype == 'block_swap':
            # Swap two random blocks
            b1, b2 = rng.choice(n_blocks, size=2, replace=False)
            lo1, hi1 = b1*block_size, min(n, (b1+1)*block_size)
            lo2, hi2 = b2*block_size, min(n, (b2+1)*block_size)
            sz = min(hi1-lo1, hi2-lo2)
            f_perturbed[lo1:lo1+sz], f_perturbed[lo2:lo2+sz] = \
                f_best[lo2:lo2+sz].copy(), f_best[lo1:lo1+sz].copy()

        elif ptype == 'scale_region':
            # Scale a random contiguous region
            region_size = rng.integers(block_size, 5*block_size)
            start = rng.integers(0, max(1, n - region_size))
            scale = 1.0 + rng.normal(0, 0.01)
            f_perturbed[start:start+region_size] *= max(0.5, scale)
            f_perturbed = np.maximum(f_perturbed, 0.0)

        # Quick Adam screen (fast — ~200 steps)
        f_screened, C_screened = adam_fast_screen(f_perturbed, None, rng, n_steps=200, lr=1e-5, beta=1e9)

        if C_screened > C_best * 0.9999:  # Promising if within 0.01% of best
            # Full L-BFGS polish
            f_polished, C_polished = full_dinkelbach(f_screened, n_outer=2, n_inner=5000, beta=1e9)

            if C_polished > C_best and C_polished < 1.0 and np.all(np.isfinite(f_polished)) and np.max(f_polished) < 1e6:
                C_best = C_polished; f_best = f_polished.copy()
                w_full = torch.tensor(np.sqrt(np.maximum(f_best, 0.0)), dtype=torch.float64, device=device)
                np.save(save_file, f_best)
                n_improvements += 1
                print(f"  R{round_num} {ptype}: NEW BEST C = {C_best:.13f} !!!")
                sys.stdout.flush()

        if round_num % 5 == 0:
            print(f"  R{round_num} {ptype}: screen={C_screened:.13f}, best={C_best:.13f} [{time.time()-t0:.0f}s]",
                  flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

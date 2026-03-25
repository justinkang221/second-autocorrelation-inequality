#!/usr/bin/env python3
"""
Iterated Dinkelbach + L-BFGS pipeline.

This approach works:
1. L-BFGS (300 iters, h² param) — refine values on fixed support
2. Dinkelbach (10 iters, 100 inner iters each) — linearize fractional objective
3. Tight L-BFGS (300 iters, soft_temp=0.0001) — sharpen soft-max

Repeat this cycle many times with different soft-max temperatures.
"""
import numpy as np
import torch
import time
import os
import sys

device = torch.device('cuda')


def gpu_score_exact(f_t):
    n = f_t.shape[0]
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F * F, n=nfft)[:nc]
    h = 1.0 / (nc + 1)
    z = torch.zeros(1, device=device, dtype=torch.float64)
    y = torch.cat([z, conv, z])
    y0, y1 = y[:-1], y[1:]
    l2sq = (h / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
    l1 = torch.sum(conv) / (nc + 1)
    linf = torch.max(conv)
    return (l2sq / (l1 * linf)).item()


def gpu_lbfgsb(f_np, maxiter=300, soft_temp=0.001):
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    support = f_t > 1e-12
    indices = torch.nonzero(support).squeeze()
    if indices.numel() == 0:
        return f_np, 0.0

    h_param = torch.sqrt(f_t[indices]).detach().requires_grad_(True)
    best_C = [0.0]
    best_h = [h_param.data.clone()]

    optimizer = torch.optim.LBFGS(
        [h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-15, tolerance_change=1e-16
    )

    def closure():
        optimizer.zero_grad()
        f = torch.zeros(n, device=device, dtype=torch.float64)
        f[indices] = h_param ** 2
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        hh = 1.0 / (nc + 1)
        z = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([z, conv, z])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv) / (nc + 1)
        linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item() / (l1.item() * linf_exact)
        if C_exact > best_C[0]:
            best_C[0] = C_exact
            best_h[0] = h_param.data.clone()
        loss = -l2sq / (l1 * linf_soft)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except:
        pass

    f_out = np.zeros(n, dtype=np.float64)
    f_out[indices.cpu().numpy()] = (best_h[0] ** 2).cpu().numpy()
    return f_out, best_C[0]


def gpu_dinkelbach(f_np, n_iters=10, maxiter_inner=100, soft_temp=0.001):
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    C_current = gpu_score_exact(f_t)
    best_C = C_current
    best_f = f_np.copy()

    for dk_iter in range(n_iters):
        lam = C_current
        support = f_t > 1e-12
        indices = torch.nonzero(support).squeeze()
        if indices.numel() == 0:
            break

        h_param = torch.sqrt(f_t[indices]).detach().requires_grad_(True)
        best_h_dk = [h_param.data.clone()]
        best_C_dk = [0.0]

        optimizer = torch.optim.LBFGS(
            [h_param], lr=1.0, max_iter=maxiter_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-15, tolerance_change=1e-16
        )

        def closure():
            optimizer.zero_grad()
            f = torch.zeros(n, device=device, dtype=torch.float64)
            f[indices] = h_param ** 2
            F = torch.fft.rfft(f, n=nfft)
            conv = torch.fft.irfft(F * F, n=nfft)[:nc]
            hh = 1.0 / (nc + 1)
            z = torch.zeros(1, device=device, dtype=torch.float64)
            y = torch.cat([z, conv, z])
            y0, y1 = y[:-1], y[1:]
            l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
            l1 = torch.sum(conv) / (nc + 1)
            linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
            linf_exact = torch.max(conv).item()
            C_exact = l2sq.item() / (l1.item() * linf_exact)
            if C_exact > best_C_dk[0]:
                best_C_dk[0] = C_exact
                best_h_dk[0] = h_param.data.clone()
            loss = -(l2sq - lam * l1 * linf_soft)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except:
            pass

        f_new = np.zeros(n, dtype=np.float64)
        f_new[indices.cpu().numpy()] = (best_h_dk[0] ** 2).cpu().numpy()
        f_t = torch.tensor(f_new, dtype=torch.float64, device=device)
        C_new = gpu_score_exact(f_t)

        if C_new > best_C:
            best_C = C_new
            best_f = f_new.copy()
        C_current = C_new

    return best_f, best_C


def gpu_softplus_lbfgs(f_np, maxiter=100, expand_radius=30, soft_temp=0.001):
    """Softplus L-BFGS for support discovery."""
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    support = torch.nonzero(f_t > 1e-12).squeeze()
    opt_mask = torch.zeros(n, dtype=torch.bool, device=device)
    opt_mask[support] = True
    for r in range(1, expand_radius + 1):
        opt_mask[torch.clamp(support + r, 0, n-1)] = True
        opt_mask[torch.clamp(support - r, 0, n-1)] = True

    indices = torch.nonzero(opt_mask).squeeze()
    f_vals = f_t[indices].clone()
    f_clamped = torch.clamp(f_vals, min=1e-10, max=50.0)
    h_init = torch.log(torch.exp(f_clamped) - 1.0 + 1e-30)
    h_init[f_vals < 1e-10] = -20.0
    h_param = h_init.detach().requires_grad_(True)

    best_C = [0.0]
    best_h = [h_param.data.clone()]

    optimizer = torch.optim.LBFGS(
        [h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-14, tolerance_change=1e-15
    )

    def closure():
        optimizer.zero_grad()
        f = torch.zeros(n, device=device, dtype=torch.float64)
        f[indices] = torch.nn.functional.softplus(h_param)
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        hh = 1.0 / (nc + 1)
        z = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([z, conv, z])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv) / (nc + 1)
        linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item() / (l1.item() * linf_exact)
        if C_exact > best_C[0]:
            best_C[0] = C_exact
            best_h[0] = h_param.data.clone()
        loss = -l2sq / (l1 * linf_soft)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except:
        pass

    f_out = np.zeros(n, dtype=np.float64)
    f_out[indices.cpu().numpy()] = torch.nn.functional.softplus(best_h[0]).cpu().numpy()
    return f_out, best_C[0]


if __name__ == "__main__":
    print("=" * 60)
    print("Iterated Dinkelbach + L-BFGS Pipeline")
    print("=" * 60)

    # Load best
    candidates = [
        'best_gpu_worker_C.npy', 'best_gpu_worker_A.npy',
        'best_gpu_worker_B.npy', 'best_from_downloaded.npy',
        'best_gpu_basin_refined.npy',
    ]
    f = None
    best_init = -1
    for fname in candidates:
        if os.path.exists(fname):
            fc = np.maximum(np.load(fname).astype(np.float64), 0.0)
            ft = torch.tensor(fc, dtype=torch.float64, device=device)
            sc = gpu_score_exact(ft)
            if sc > best_init:
                best_init = sc
                f = fc

    C_best = best_init
    print(f"Initial: C = {C_best:.12f}")

    t0 = time.time()
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    class long_stall:
        count = 0

    # Vary temperatures across cycles to explore different landscapes
    temp_schedule = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    expand_schedule = [20, 30, 50, 80, 100, 150]

    for cycle in range(n_cycles):
        C_start = C_best
        temp = temp_schedule[cycle % len(temp_schedule)]
        tight_temp = temp / 10.0

        # Phase 1: L-BFGS with varying temp
        f1, C1 = gpu_lbfgsb(f, maxiter=300, soft_temp=temp)
        if C1 > C_best:
            C_best = C1
            f = f1

        # Phase 2: Dinkelbach (10 iterations)
        f2, C2 = gpu_dinkelbach(f, n_iters=10, maxiter_inner=100, soft_temp=temp)
        if C2 > C_best:
            C_best = C2
            f = f2

        # Phase 3: Tight L-BFGS
        f3, C3 = gpu_lbfgsb(f, maxiter=300, soft_temp=tight_temp)
        if C3 > C_best:
            C_best = C3
            f = f3

        # Phase 4: Very tight Dinkelbach
        f4, C4 = gpu_dinkelbach(f, n_iters=5, maxiter_inner=100, soft_temp=tight_temp)
        if C4 > C_best:
            C_best = C4
            f = f4

        # Phase 5: Softplus support discovery (every 3 cycles with increasing radius)
        if cycle % 3 == 2:
            expand = expand_schedule[min(cycle // 3, len(expand_schedule) - 1)]
            f5, C5 = gpu_softplus_lbfgs(f, maxiter=100, expand_radius=expand, soft_temp=temp)
            if C5 > C_best:
                C_best = C5
                f = f5
                print(f"  Cycle {cycle:3d}: SOFTPLUS improved! (expand={expand})")

        delta = C_best - C_start
        elapsed = time.time() - t0

        np.save('best_dinkelbach_iter.npy', f)
        # Checkpoint every 10 cycles
        if cycle % 10 == 0:
            np.save(f'checkpoint_C{C_best:.6f}_cycle{cycle}.npy', f)
        # Update shared files
        for sib in ['best_gpu_worker_A.npy', 'best_gpu_worker_B.npy',
                   'best_gpu_worker_C.npy']:
            try:
                fc = np.maximum(np.load(sib).astype(np.float64), 0.0)
                ft = torch.tensor(fc, dtype=torch.float64, device=device)
                if gpu_score_exact(ft) < C_best:
                    np.save(sib, f)
            except:
                pass

        print(f"  Cycle {cycle:3d}: C = {C_best:.12f} "
              f"(Δ={delta:.2e}, {elapsed:.0f}s)")

        # No early stopping — run all cycles

    print(f"\n{'='*60}")
    print(f"Final: C = {C_best:.12f}")
    print(f"Improvement: Δ = {C_best - best_init:.2e}")
    print(f"Time: {time.time() - t0:.0f}s")

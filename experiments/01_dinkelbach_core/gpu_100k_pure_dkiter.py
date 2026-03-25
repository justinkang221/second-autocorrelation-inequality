#!/usr/bin/env python3
"""
Pure dk_iter on 100k — test hypothesis that surgery has moved us out of
the old basin and dk_iter alone can now find new gains.

Same approach that's working great on 200k (0.96186 → 0.96202+).
Large expand radii (up to 5000), no surgery.
"""
import numpy as np
import torch
import time
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

def gpu_softplus_lbfgs(f_np, maxiter=150, expand_radius=500, soft_temp=0.001):
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
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
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

def gpu_lbfgsb(f_np, maxiter=500, soft_temp=0.001):
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
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
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

def gpu_dinkelbach(f_np, n_iters=10, maxiter_inner=150, soft_temp=0.001):
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
        if indices.numel() == 0: break
        h_param = torch.sqrt(f_t[indices]).detach().requires_grad_(True)
        best_h_dk = [h_param.data.clone()]
        best_C_dk = [0.0]
        optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter_inner,
            line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)
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


if __name__ == "__main__":
    print("=" * 70)
    print("100K Pure dk_iter (post-surgery basin test)")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    f = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
    n = len(f)
    C_best = gpu_score_exact(torch.tensor(f, dtype=torch.float64, device=device))
    f_best = f.copy()
    print(f"Starting: n={n}, C = {C_best:.12f}")
    sys.stdout.flush()

    expand_schedule = [500, 1000, 2000, 3000, 5000, 1500, 2500, 4000]
    temps = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]

    for cycle in range(500):
        expand = expand_schedule[cycle % len(expand_schedule)]
        temp = temps[cycle % len(temps)]

        # Softplus with large radius
        f_sp, C_sp = gpu_softplus_lbfgs(f_best, maxiter=150,
                                         expand_radius=expand, soft_temp=temp)
        if C_sp > C_best:
            C_best = C_sp
            f_best = f_sp.copy()
            np.save('best_dinkelbach_iter.npy', f_best)
            print(f"  Cycle {cycle} [softplus r={expand}]: NEW BEST C = {C_best:.12f} !!!")

        # L-BFGS refinement
        f1, C1 = gpu_lbfgsb(f_best, maxiter=500, soft_temp=temp)
        if C1 > C_best:
            C_best = C1
            f_best = f1.copy()
            np.save('best_dinkelbach_iter.npy', f_best)
            print(f"  Cycle {cycle} [lbfgs]: NEW BEST C = {C_best:.12f} !!!")

        # Dinkelbach
        f2, C2 = gpu_dinkelbach(f_best, n_iters=10, maxiter_inner=150, soft_temp=temp)
        if C2 > C_best:
            C_best = C2
            f_best = f2.copy()
            np.save('best_dinkelbach_iter.npy', f_best)
            print(f"  Cycle {cycle} [dinkelbach]: NEW BEST C = {C_best:.12f} !!!")

        if cycle % 10 == 9:
            elapsed = time.time() - t0
            print(f"  Cycle {cycle}: C_best = {C_best:.12f} [{elapsed:.0f}s]")
        sys.stdout.flush()

    np.save('best_dinkelbach_iter.npy', f_best)
    print(f"\nFINAL: C = {C_best:.12f}")
    print(f"Total time: {time.time() - t0:.0f}s")

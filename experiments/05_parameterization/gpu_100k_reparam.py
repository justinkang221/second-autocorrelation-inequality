#!/usr/bin/env python3
"""
Re-parameterization experiments for 100k.
Try multiple parameterizations and Dinkelbach decompositions to escape
the current local optimum.

Key insight: w² parameterization locks in sparsity pattern (gradient=0 at w=0).
Alternative parameterizations create different landscapes with different local optima.

Parameterizations:
  A) f = exp(v)           — no zero-gradient trap
  B) f = softplus(v)      — smooth, no zero trap
  C) f directly + clamp   — full gradient everywhere
  D) f = w² in Fourier    — different Hessian structure

Dinkelbach decompositions:
  1) Standard: max l2sq - λ·l1·linf     (current)
  2) L1-ratio: max l2sq/l1 - λ·linf
  3) Linf-ratio: max l2sq/linf - λ·l1
  4) Log-space: max log(l2sq) - log(l1) - log(linf)  (not fractional, direct)
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


def compute_components(f, nfft, nc, beta):
    """Compute l2sq, l1, linf_proxy, g_max from f."""
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
    return l2sq, l1, linf_proxy, g_max


def optimize_exp_param(f_np, n_outer=5, n_inner=3000, beta=1e8, dink_mode='standard'):
    """Optimize with f = exp(v) parameterization."""
    n = len(f_np); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    # Initialize v = log(f + eps)
    eps = 1e-20
    v_init = np.log(np.maximum(f_np, eps))
    # For truly zero positions, set v to very negative
    v_init[f_np < 1e-15] = -40.0

    lam = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))
    best_C = lam; best_f = f_np.copy()

    for outer in range(n_outer):
        v_t = torch.tensor(v_init, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [v_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_v = [v_t.data.clone()]
        best_C_inner = [0.0]

        def closure():
            optimizer.zero_grad()
            f = torch.exp(v_t)
            l2sq, l1, linf_proxy, g_max = compute_components(f, nfft, nc, beta)

            if dink_mode == 'standard':
                obj = l2sq - current_lam * l1 * linf_proxy
            elif dink_mode == 'l1_ratio':
                obj = l2sq / (l1 + 1e-18) - current_lam * linf_proxy
            elif dink_mode == 'linf_ratio':
                obj = l2sq / (linf_proxy + 1e-18) - current_lam * l1
            elif dink_mode == 'log':
                obj = torch.log(l2sq + 1e-18) - torch.log(l1 + 1e-18) - torch.log(linf_proxy + 1e-18)

            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_C_inner[0] and C_exact < 1.0:
                best_C_inner[0] = C_exact
                best_v[0] = v_t.data.clone()

            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            pass

        f_new = torch.exp(best_v[0]).cpu().numpy()
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new
        v_init = best_v[0].cpu().numpy()

        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def optimize_softplus_param(f_np, n_outer=5, n_inner=3000, beta=1e8, dink_mode='standard'):
    """Optimize with f = softplus(v) = log(1 + exp(v))."""
    n = len(f_np); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    # Initialize: v = log(exp(f) - 1) = inverse softplus
    # For small f, v ≈ log(f); for large f, v ≈ f
    f_safe = np.maximum(f_np, 1e-20)
    v_init = np.log(np.expm1(f_safe))
    v_init[~np.isfinite(v_init)] = -40.0

    lam = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))
    best_C = lam; best_f = f_np.copy()

    for outer in range(n_outer):
        v_t = torch.tensor(v_init, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [v_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_v = [v_t.data.clone()]
        best_C_inner = [0.0]

        def closure():
            optimizer.zero_grad()
            f = torch.nn.functional.softplus(v_t)
            l2sq, l1, linf_proxy, g_max = compute_components(f, nfft, nc, beta)

            if dink_mode == 'standard':
                obj = l2sq - current_lam * l1 * linf_proxy
            elif dink_mode == 'l1_ratio':
                obj = l2sq / (l1 + 1e-18) - current_lam * linf_proxy
            elif dink_mode == 'linf_ratio':
                obj = l2sq / (linf_proxy + 1e-18) - current_lam * l1
            elif dink_mode == 'log':
                obj = torch.log(l2sq + 1e-18) - torch.log(l1 + 1e-18) - torch.log(linf_proxy + 1e-18)

            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_C_inner[0] and C_exact < 1.0:
                best_C_inner[0] = C_exact
                best_v[0] = v_t.data.clone()

            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass

        f_new = torch.nn.functional.softplus(best_v[0]).cpu().numpy()
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new
        v_init = best_v[0].cpu().numpy()

        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def optimize_direct_param(f_np, n_outer=5, n_inner=3000, beta=1e8, dink_mode='standard'):
    """Optimize f directly, clamp to >=0 inside closure."""
    n = len(f_np); nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    lam = gpu_score_exact(torch.tensor(f_np, dtype=torch.float64, device=device))
    best_C = lam; best_f = f_np.copy()

    for outer in range(n_outer):
        f_t = torch.tensor(best_f, dtype=torch.float64, device=device).requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [f_t], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_ft = [f_t.data.clone()]
        best_C_inner = [0.0]

        def closure():
            optimizer.zero_grad()
            # Soft clamp: use relu to enforce non-negativity
            f = torch.relu(f_t)
            l2sq, l1, linf_proxy, g_max = compute_components(f, nfft, nc, beta)

            if dink_mode == 'standard':
                obj = l2sq - current_lam * l1 * linf_proxy
            elif dink_mode == 'l1_ratio':
                obj = l2sq / (l1 + 1e-18) - current_lam * linf_proxy
            elif dink_mode == 'linf_ratio':
                obj = l2sq / (linf_proxy + 1e-18) - current_lam * l1
            elif dink_mode == 'log':
                obj = torch.log(l2sq + 1e-18) - torch.log(l1 + 1e-18) - torch.log(linf_proxy + 1e-18)

            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_C_inner[0] and C_exact < 1.0:
                best_C_inner[0] = C_exact
                best_ft[0] = f_t.data.clone()

            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass

        f_new = torch.relu(best_ft[0]).cpu().numpy()
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new

        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


def optimize_w2_param(f_np, n_outer=5, n_inner=3000, beta=1e8, dink_mode='standard'):
    """Standard w² with alternative Dinkelbach decomposition."""
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
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_w = [w_t.data.clone()]
        best_C_inner = [0.0]

        def closure():
            optimizer.zero_grad()
            f = w_t ** 2
            l2sq, l1, linf_proxy, g_max = compute_components(f, nfft, nc, beta)

            if dink_mode == 'standard':
                obj = l2sq - current_lam * l1 * linf_proxy
            elif dink_mode == 'l1_ratio':
                obj = l2sq / (l1 + 1e-18) - current_lam * linf_proxy
            elif dink_mode == 'linf_ratio':
                obj = l2sq / (linf_proxy + 1e-18) - current_lam * l1
            elif dink_mode == 'log':
                obj = torch.log(l2sq + 1e-18) - torch.log(l1 + 1e-18) - torch.log(linf_proxy + 1e-18)

            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_C_inner[0] and C_exact < 1.0:
                best_C_inner[0] = C_exact
                best_w[0] = w_t.data.clone()

            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass

        w = best_w[0].cpu().numpy()
        f_new = w ** 2
        C_new = gpu_score_exact(torch.tensor(f_new, dtype=torch.float64, device=device))
        lam = C_new

        if C_new > best_C and C_new < 1.0 and np.all(np.isfinite(f_new)) and np.max(f_new) < 1e6:
            best_C = C_new; best_f = f_new.copy()

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            break

    return best_f, best_C


if __name__ == "__main__":
    print("=" * 70)
    print("Re-parameterization experiments for 100k")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    try:
        f_best = np.maximum(np.load('best_impevo_100k.npy').astype(np.float64), 0.0)
    except:
        f_best = np.maximum(np.load('best_sa_surgery_v2.npy').astype(np.float64), 0.0)

    n = len(f_best)
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"Starting: n={n}, C = {C_best:.13f}")
    sys.stdout.flush()

    save_file = 'best_impevo_100k.npy'
    n_improvements = 0

    # Test all combinations of parameterization × Dinkelbach decomposition
    parameterizations = [
        ('exp', optimize_exp_param),
        ('softplus', optimize_softplus_param),
        ('direct+relu', optimize_direct_param),
        ('w2', optimize_w2_param),
    ]

    dink_modes = ['standard', 'l1_ratio', 'linf_ratio', 'log']

    for beta in [1e6, 1e7, 1e8, 5e8]:
        print(f"\n=== Beta = {beta:.0e} ===")
        for param_name, param_fn in parameterizations:
            for dink_mode in dink_modes:
                # Cross-pollinate
                try:
                    f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
                    if len(f_shared) == n:
                        C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                        if C_shared > C_best and C_shared < 1.0:
                            C_best = C_shared; f_best = f_shared.copy()
                except: pass

                try:
                    f_opt, C_opt = param_fn(
                        f_best, n_outer=3, n_inner=3000, beta=beta, dink_mode=dink_mode
                    )
                except Exception as e:
                    print(f"  {param_name}/{dink_mode}: FAILED ({e})")
                    continue

                improved = ""
                if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                    C_best = C_opt; f_best = f_opt.copy()
                    np.save(save_file, f_best)
                    n_improvements += 1
                    improved = " *** NEW BEST ***"

                print(f"  {param_name}/{dink_mode}: C={C_opt:.13f} (Δ={C_opt-C_best:+.2e}){improved} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    # Phase 2: Best-performing combos get more iterations
    print(f"\n=== Phase 2: Extended optimization with best combos ===")
    print(f"Current best: C = {C_best:.13f}")

    # Run the most promising combinations with more iterations and higher beta
    for beta in [1e8, 5e8, 1e9, 5e9]:
        for param_name, param_fn in parameterizations:
            for dink_mode in dink_modes:
                # Cross-pollinate
                try:
                    f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
                    if len(f_shared) == n:
                        C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                        if C_shared > C_best and C_shared < 1.0:
                            C_best = C_shared; f_best = f_shared.copy()
                except: pass

                try:
                    f_opt, C_opt = param_fn(
                        f_best, n_outer=5, n_inner=5000, beta=beta, dink_mode=dink_mode
                    )
                except Exception as e:
                    continue

                improved = ""
                if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                    C_best = C_opt; f_best = f_opt.copy()
                    np.save(save_file, f_best)
                    n_improvements += 1
                    improved = " *** NEW BEST ***"

                print(f"  {param_name}/{dink_mode} beta={beta:.0e}: C={C_opt:.13f}{improved} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

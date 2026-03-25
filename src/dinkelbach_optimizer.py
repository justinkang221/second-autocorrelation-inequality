"""
Dinkelbach optimizer for the second autocorrelation inequality.

Maximizes C(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf) over non-negative f.

Key ideas:
1. Dinkelbach iteration converts the fractional program into a sequence of
   smooth subproblems: max_f [l2sq - lambda * l1 * linf_approx]
2. LogSumExp provides a smooth approximation to Linf, parameterized by beta.
3. Square-root parameterization f = w^2 enforces non-negativity.
4. L-BFGS with strong Wolfe line search solves each inner problem.
5. Beta cascade from low to high refines the approximation.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU optimization. "
                          "Install with: pip install torch")


def gpu_score(f_t, device=None):
    """Compute C(f) on GPU using Simpson's rule (matches platform verifier)."""
    _check_torch()
    n = f_t.shape[0]; nc = 2 * n - 1; nfft = 1
    while nfft < nc: nfft <<= 1
    F = torch.fft.rfft(f_t, n=nfft)
    conv = torch.fft.irfft(F * F, n=nfft)[:nc]
    h = 1.0 / (nc + 1)
    z = torch.zeros(1, device=f_t.device, dtype=f_t.dtype)
    y = torch.cat([z, conv, z])
    y0, y1 = y[:-1], y[1:]
    l2sq = (h / 3.0) * torch.sum(y0**2 + y0 * y1 + y1**2)
    l1 = torch.sum(conv) / (nc + 1)
    linf = torch.max(conv)
    return (l2sq / (l1 * linf)).item()


def dinkelbach_inner(f_np, lam, beta=1e8, n_inner=5000, device='cuda'):
    """Solve one Dinkelbach inner problem via L-BFGS.

    Maximizes: l2sq - lam * l1 * linf_proxy
    Subject to: f = w^2 >= 0

    Args:
        f_np: Current solution (numpy array).
        lam: Dinkelbach parameter (current C estimate).
        beta: LogSumExp temperature for Linf approximation.
        n_inner: Maximum L-BFGS iterations.
        device: 'cuda' or 'cpu'.

    Returns:
        (f_new, C_new): Optimized solution and its exact score.
    """
    _check_torch()
    w = np.sqrt(np.maximum(f_np, 0.0))
    n = len(w); nc = 2 * n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    w_t = torch.tensor(w, dtype=torch.float64, device=device).requires_grad_(True)
    best_w = w_t.data.clone()
    best_C = 0.0

    optimizer = torch.optim.LBFGS(
        [w_t], lr=1.0, max_iter=n_inner,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-14, tolerance_change=1e-15
    )

    def closure():
        nonlocal best_w, best_C
        optimizer.zero_grad()
        f = w_t ** 2
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        conv = torch.maximum(conv, torch.zeros(1, device=device, dtype=torch.float64))

        # Simpson's rule L2^2
        hh = 1.0 / (nc + 1)
        zz = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([zz, conv, zz])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh / 3.0) * torch.sum(y0**2 + y0 * y1 + y1**2)

        # L1 norm
        l1 = torch.sum(conv) / (nc + 1)

        # Smooth Linf via LogSumExp (numerically stable)
        g_max = torch.max(conv)
        linf_proxy = g_max * torch.exp(
            torch.logsumexp(beta * (conv / (g_max + 1e-18) - 1.0), dim=0) / beta
        )

        # Dinkelbach objective
        obj = l2sq - lam * l1 * linf_proxy

        # Track exact C
        C_exact = l2sq.item() / (l1.item() * g_max.item())
        if C_exact > best_C:
            best_C = C_exact
            best_w = w_t.data.clone()

        loss = -obj
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        pass

    f_new = (best_w ** 2).cpu().numpy()
    C_new = gpu_score(torch.tensor(f_new, dtype=torch.float64, device=device))
    return f_new, C_new


def optimize(f_init, betas=None, n_outer=5, n_inner=5000, device='cuda', verbose=True):
    """Full Dinkelbach optimization with beta cascade.

    Args:
        f_init: Initial non-negative function values (numpy array).
        betas: List of beta values to sweep (low to high).
               Default: [1e5, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9].
        n_outer: Number of Dinkelbach outer iterations per beta.
        n_inner: Maximum L-BFGS iterations per inner problem.
        device: 'cuda' or 'cpu'.
        verbose: Print progress.

    Returns:
        (f_opt, C_opt): Optimized solution and its score.
    """
    _check_torch()

    if betas is None:
        betas = [1e5, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9]

    f_best = np.maximum(f_init.astype(np.float64), 0.0)
    C_best = gpu_score(torch.tensor(f_best, dtype=torch.float64, device=device))

    if verbose:
        print(f"Initial C = {C_best:.13f}")

    for beta in betas:
        f_current = f_best.copy()
        lam = C_best

        for outer in range(n_outer):
            f_new, C_new = dinkelbach_inner(
                f_current, lam, beta=beta, n_inner=n_inner, device=device
            )
            lam = C_new

            if C_new > C_best and C_new < 1.0 and np.all(np.isfinite(f_new)):
                C_best = C_new
                f_best = f_new.copy()
                f_current = f_new

            if outer > 0 and abs(C_new - C_best) < 1e-14:
                break

        if verbose:
            print(f"  beta={beta:.0e}: C = {C_best:.13f}")

    return f_best, C_best

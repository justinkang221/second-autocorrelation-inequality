#!/usr/bin/env python3
"""
Fourier Domain Optimization: Optimize f's Fourier representation directly.

Key insight: autoconvolution f★f has Fourier transform |F(f)|².
For C to be maximized, we want |F(f)|² to be as "flat" as possible
in the support region. This is equivalent to designing f with a
constant-magnitude Fourier transform — like a Barker code or chirp.

This approach parameterizes f differently from value-space optimization
and may escape the basin.
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


def fourier_to_f(amp_param, phase_param, n, nfft):
    """Convert Fourier parameters to non-negative f."""
    # Build complex spectrum
    amp = torch.nn.functional.softplus(amp_param)
    spectrum = amp * torch.exp(1j * phase_param)

    # Full spectrum (with conjugate symmetry for real signal)
    f_full = torch.fft.irfft(spectrum, n=nfft)[:n]

    # Ensure non-negativity via softplus
    f_pos = torch.nn.functional.softplus(f_full * 10.0) / 10.0

    return f_pos


def optimize_fourier(f_np, n_freq=500, maxiter=200, soft_temp=0.001):
    """Optimize in Fourier domain."""
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1

    # Initialize from current solution's Fourier transform
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    F_init = torch.fft.rfft(f_t, n=nfft)

    # Keep only n_freq components (low-pass approximation)
    n_rfft = nfft // 2 + 1
    n_use = min(n_freq, n_rfft)

    amp_init = torch.abs(F_init[:n_use])
    phase_init = torch.angle(F_init[:n_use])

    # Convert amplitude to softplus inverse
    amp_param = torch.log(torch.exp(amp_init.clamp(min=1e-10)) - 1.0 + 1e-30)
    amp_param = amp_param.detach().requires_grad_(True)
    phase_param = phase_init.detach().requires_grad_(True)

    best_C = [0.0]
    best_amp = [amp_param.data.clone()]
    best_phase = [phase_param.data.clone()]

    optimizer = torch.optim.LBFGS([amp_param, phase_param], lr=0.1, max_iter=maxiter,
        line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-16)

    def closure():
        optimizer.zero_grad()

        amp = torch.nn.functional.softplus(amp_param)
        spectrum = torch.zeros(n_rfft, dtype=torch.complex128, device=device)
        spectrum[:n_use] = amp * torch.exp(1j * phase_param.to(torch.complex128))

        f_full = torch.fft.irfft(spectrum, n=nfft)[:n]
        # Softplus for non-negativity
        f = torch.nn.functional.softplus(f_full * 5.0) / 5.0

        # Score
        F2 = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F2 * F2, n=nfft)[:nc]
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
            best_amp[0] = amp_param.data.clone()
            best_phase[0] = phase_param.data.clone()

        loss = -l2sq / (l1 * linf_soft)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as e:
        print(f"  Fourier opt exception: {e}")

    # Reconstruct best f
    amp = torch.nn.functional.softplus(best_amp[0])
    spectrum = torch.zeros(n_rfft, dtype=torch.complex128, device=device)
    spectrum[:n_use] = amp * torch.exp(1j * best_phase[0].to(torch.complex128))
    f_full = torch.fft.irfft(spectrum, n=nfft)[:n]
    f_out = (torch.nn.functional.softplus(f_full * 5.0) / 5.0).detach().cpu().numpy()

    return f_out, best_C[0]


def gpu_lbfgsb(f_np, maxiter=500, soft_temp=0.001):
    """Standard L-BFGS refinement."""
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device)
    support = f_t > 0
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
        support = f_t > 0
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
    print("Fourier Domain Optimization + Hybrid Strategy")
    print("=" * 70)
    sys.stdout.flush()

    t0 = time.time()
    f = np.maximum(np.load('best_dinkelbach_iter.npy').astype(np.float64), 0.0)
    n = len(f)
    f_t = torch.tensor(f, dtype=torch.float64, device=device)
    C_best = gpu_score_exact(f_t)
    f_best = f.copy()
    print(f"Starting: C = {C_best:.12f}")
    sys.stdout.flush()

    # =========================================================================
    # Experiment 1: Fourier optimization with different n_freq
    # =========================================================================
    print("\n=== Experiment 1: Fourier domain optimization ===")
    for n_freq in [200, 500, 1000, 2000, 5000, 10000]:
        print(f"\n  n_freq = {n_freq}:")
        f_fourier, C_fourier = optimize_fourier(f_best, n_freq=n_freq,
            maxiter=100, soft_temp=0.001)
        print(f"    After Fourier opt: C = {C_fourier:.12f}")

        # Re-optimize with standard dk_iter
        for cycle in range(5):
            temp = [0.005, 0.001, 0.0005][cycle % 3]
            f_sp, C_sp = gpu_softplus_lbfgs(f_fourier, maxiter=150,
                expand_radius=500, soft_temp=temp)
            if C_sp > gpu_score_exact(torch.tensor(f_fourier, dtype=torch.float64, device=device)):
                f_fourier = f_sp
            f1, C1 = gpu_lbfgsb(f_fourier, maxiter=300, soft_temp=temp)
            if C1 > gpu_score_exact(torch.tensor(f_fourier, dtype=torch.float64, device=device)):
                f_fourier = f1
            f2, C2 = gpu_dinkelbach(f_fourier, n_iters=5, maxiter_inner=100, soft_temp=temp)
            C_work = gpu_score_exact(torch.tensor(f_fourier, dtype=torch.float64, device=device))
            if C2 > C_work:
                f_fourier = f2
                C_work = C2
        C_final = gpu_score_exact(torch.tensor(f_fourier, dtype=torch.float64, device=device))
        print(f"    After dk_iter refinement: C = {C_final:.12f}")

        if C_final > C_best:
            C_best = C_final
            f_best = f_fourier.copy()
            np.save('best_dinkelbach_iter.npy', f_best)
            print(f"    *** NEW BEST: C = {C_best:.12f} ***")

        elapsed = time.time() - t0
        print(f"    [{elapsed:.0f}s]")
        sys.stdout.flush()

    # =========================================================================
    # Experiment 2: Fourier perturbation — modify random frequency components
    # =========================================================================
    print("\n=== Experiment 2: Fourier perturbation + re-optimize ===")
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc: nfft <<= 1
    n_rfft = nfft // 2 + 1

    for trial in range(50):
        f_t = torch.tensor(f_best, dtype=torch.float64, device=device)
        F = torch.fft.rfft(f_t, n=nfft)

        # Perturb random frequency components
        n_perturb = [10, 20, 50, 100, 200][trial % 5]
        sigma = [0.01, 0.02, 0.05, 0.1, 0.2][trial % 5]

        perturb_indices = torch.randint(0, n_rfft, (n_perturb,), device=device)
        F_perturbed = F.clone()
        noise = sigma * torch.randn(n_perturb, dtype=torch.float64, device=device)
        noise_imag = sigma * torch.randn(n_perturb, dtype=torch.float64, device=device)
        F_perturbed[perturb_indices] += noise + 1j * noise_imag

        # Reconstruct and ensure non-negativity
        f_perturbed = torch.fft.irfft(F_perturbed, n=nfft)[:n]
        f_perturbed = torch.clamp(f_perturbed, min=0.0).cpu().numpy()

        C_perturbed = gpu_score_exact(torch.tensor(f_perturbed, dtype=torch.float64, device=device))

        if C_perturbed > C_best * 0.999:
            # Re-optimize
            for cycle in range(5):
                temp = [0.005, 0.001, 0.0005][cycle % 3]
                expand = [200, 500, 1000][cycle % 3]
                f_sp, C_sp = gpu_softplus_lbfgs(f_perturbed, maxiter=150,
                    expand_radius=expand, soft_temp=temp)
                if C_sp > gpu_score_exact(
                    torch.tensor(f_perturbed, dtype=torch.float64, device=device)):
                    f_perturbed = f_sp
                f1, C1 = gpu_lbfgsb(f_perturbed, maxiter=300, soft_temp=temp)
                if C1 > gpu_score_exact(
                    torch.tensor(f_perturbed, dtype=torch.float64, device=device)):
                    f_perturbed = f1

            C_final = gpu_score_exact(
                torch.tensor(f_perturbed, dtype=torch.float64, device=device))
            if C_final > C_best:
                C_best = C_final
                f_best = f_perturbed.copy() if isinstance(f_perturbed, np.ndarray) \
                    else f_perturbed
                np.save('best_dinkelbach_iter.npy', f_best)
                print(f"  Trial {trial}: NEW BEST C = {C_best:.12f} !!!")

        if trial % 10 == 9:
            elapsed = time.time() - t0
            print(f"  Trials {trial-9}-{trial} done. C_best = {C_best:.12f} [{elapsed:.0f}s]")
            sys.stdout.flush()

    # =========================================================================
    # Experiment 3: Spectral flattening — directly target |F(f)|² flatness
    # =========================================================================
    print("\n=== Experiment 3: Spectral flattening of autoconvolution ===")

    f_t = torch.tensor(f_best, dtype=torch.float64, device=device)
    F = torch.fft.rfft(f_t, n=nfft)
    power = (F * F.conj()).real
    print(f"  Power spectrum: mean={power.mean().item():.4e}, "
          f"std={power.std().item():.4e}, max={power.max().item():.4e}")

    # Identify frequency components where |F|² deviates most from flat
    # and try to adjust them
    support_freq = power > power.mean() * 0.1
    mean_power = power[support_freq].mean()

    # Try scaling frequency components toward mean
    for alpha in [0.01, 0.02, 0.05, 0.1, 0.2]:
        F_flat = F.clone()
        scale = torch.ones_like(power)
        scale[support_freq] = (1 - alpha) + alpha * torch.sqrt(mean_power / power[support_freq].clamp(min=1e-30))
        F_flat = F_flat * scale

        f_flat = torch.fft.irfft(F_flat, n=nfft)[:n]
        f_flat = torch.clamp(f_flat, min=0.0).cpu().numpy()

        C_flat = gpu_score_exact(torch.tensor(f_flat, dtype=torch.float64, device=device))
        print(f"  alpha={alpha:.2f}: C = {C_flat:.12f} (before re-opt)")

        # Re-optimize
        for cycle in range(5):
            temp = [0.005, 0.001, 0.0005][cycle % 3]
            f_sp, C_sp = gpu_softplus_lbfgs(f_flat, maxiter=150,
                expand_radius=500, soft_temp=temp)
            if C_sp > gpu_score_exact(torch.tensor(f_flat, dtype=torch.float64, device=device)):
                f_flat = f_sp
            f1, C1 = gpu_lbfgsb(f_flat, maxiter=300, soft_temp=temp)
            if C1 > gpu_score_exact(torch.tensor(f_flat, dtype=torch.float64, device=device)):
                f_flat = f1
            f2, C2 = gpu_dinkelbach(f_flat, n_iters=5, maxiter_inner=100, soft_temp=temp)
            C_work = gpu_score_exact(torch.tensor(f_flat, dtype=torch.float64, device=device))
            if C2 > C_work:
                f_flat = f2
                C_work = C2

        C_final = gpu_score_exact(torch.tensor(f_flat, dtype=torch.float64, device=device))
        print(f"  alpha={alpha:.2f}: C = {C_final:.12f} (after re-opt)")

        if C_final > C_best:
            C_best = C_final
            f_best = f_flat if isinstance(f_flat, np.ndarray) else f_flat.copy()
            np.save('best_dinkelbach_iter.npy', f_best)
            print(f"  *** NEW BEST: C = {C_best:.12f} ***")

        sys.stdout.flush()

    np.save('best_fourier.npy', f_best)
    np.save('best_dinkelbach_iter.npy', f_best)
    print(f"\nFINAL: C = {C_best:.12f}")
    print(f"Total time: {time.time() - t0:.0f}s")

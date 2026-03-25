#!/usr/bin/env python3
"""
Fourier-parameterized Dinkelbach for 100k.
Key ideas:
1. Parameterize w in truncated Fourier basis: w = IFFT(c[:K]), f = w²
   - Automatic non-negativity
   - K << n parameters → faster optimization, smoother landscape
   - Each parameter creates global, coherent changes
2. Beta annealing: temporarily reduce beta to explore, then increase to refine
3. Sweep over K to find the sweet spot
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


def fourier_dinkelbach(c_coeffs, n, K, n_outer=5, n_inner=3000, beta=1e8):
    """
    Dinkelbach optimization over truncated Fourier coefficients.
    c_coeffs: complex tensor of shape (K,) — Fourier coefficients of w
    w = irfft(c_coeffs, n=n), f = w²
    """
    nc = 2*n - 1; nfft = 1
    while nfft < nc: nfft <<= 1

    # Split into real/imag for optimization
    c_real = c_coeffs.real.detach().clone().requires_grad_(True)
    c_imag = c_coeffs.imag.detach().clone().requires_grad_(True)

    # Initial score
    with torch.no_grad():
        c = torch.complex(c_real, c_imag)
        w = torch.fft.irfft(c, n=n)
        f = w ** 2
    lam = gpu_score_exact(f)
    best_C = lam
    best_cr = c_real.data.clone()
    best_ci = c_imag.data.clone()

    for outer in range(n_outer):
        c_real = best_cr.clone().requires_grad_(True)
        c_imag = best_ci.clone().requires_grad_(True)
        current_lam = lam

        optimizer = torch.optim.LBFGS(
            [c_real, c_imag], lr=1.0, max_iter=n_inner,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-15
        )
        best_C_inner = [0.0]
        best_cr_inner = [c_real.data.clone()]
        best_ci_inner = [c_imag.data.clone()]
        eval_count = [0]

        def closure():
            optimizer.zero_grad()
            c = torch.complex(c_real, c_imag)
            w = torch.fft.irfft(c, n=n)
            f = w ** 2

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
            eval_count[0] += 1

            C_exact = l2sq.item() / (l1.item() * g_max.item())
            if C_exact > best_C_inner[0] and C_exact < 1.0:
                best_C_inner[0] = C_exact
                best_cr_inner[0] = c_real.data.clone()
                best_ci_inner[0] = c_imag.data.clone()

            loss = -obj
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    Outer {outer} exception: {e}", flush=True)

        # Evaluate
        with torch.no_grad():
            c = torch.complex(best_cr_inner[0], best_ci_inner[0])
            w = torch.fft.irfft(c, n=n)
            f = w ** 2
        C_new = gpu_score_exact(f)
        lam = C_new

        if C_new > best_C and C_new < 1.0:
            best_C = C_new
            best_cr = best_cr_inner[0]
            best_ci = best_ci_inner[0]

        print(f"    Fourier Dink outer={outer}: C={C_new:.13f} (K={K}, evals={eval_count[0]})", flush=True)

        if outer > 0 and abs(C_new - best_C) < 1e-14:
            print(f"    Converged", flush=True)
            break

    # Return f and coefficients
    with torch.no_grad():
        c = torch.complex(best_cr, best_ci)
        w = torch.fft.irfft(c, n=n)
        f_out = (w ** 2).cpu().numpy()

    return f_out, best_C, torch.complex(best_cr, best_ci)


def full_dinkelbach(f_np, n_outer=3, n_inner=5000, beta=1e9):
    """Standard full-vector Dinkelbach for final polish."""
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


def f_to_fourier_coeffs(f_np, K):
    """Extract truncated Fourier coefficients of w = sqrt(f)."""
    w = np.sqrt(np.maximum(f_np, 0.0))
    W = np.fft.rfft(w)
    # Truncate to K coefficients
    c = np.zeros(K, dtype=np.complex128)
    c[:min(K, len(W))] = W[:min(K, len(W))]
    return torch.tensor(c, dtype=torch.complex128, device=device)


if __name__ == "__main__":
    print("=" * 70)
    print("Fourier-parameterized + Beta-annealing Dinkelbach for 100k")
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

    save_file = 'best_impevo_100k.npy'
    rng = np.random.default_rng(42)
    n_improvements = 0

    # Phase 1: Sweep over K (number of Fourier modes) to find sweet spot
    print("\n--- Phase 1: Fourier truncation sweep ---")
    # Try different K values
    for K in [1000, 2000, 5000, 10000, 20000, 50001]:
        print(f"\n  K = {K} ({2*K} real params vs {n} full)")

        # Extract Fourier coefficients from current best
        c_coeffs = f_to_fourier_coeffs(f_best, K)

        # Check reconstruction quality
        with torch.no_grad():
            w_recon = torch.fft.irfft(c_coeffs, n=n)
            f_recon = (w_recon ** 2).cpu().numpy()
        C_recon = gpu_score_exact(torch.tensor(f_recon, dtype=torch.float64, device=device))
        print(f"  Reconstruction: C = {C_recon:.13f} (loss = {C_best - C_recon:.2e})")

        if C_recon < C_best * 0.99:
            print(f"  Too much reconstruction loss, skipping optimization")
            continue

        # Optimize in Fourier space with beta sweep
        for beta in [1e6, 1e7, 1e8]:
            f_opt, C_opt, c_opt = fourier_dinkelbach(
                c_coeffs, n, K, n_outer=3, n_inner=2000, beta=beta
            )
            c_coeffs = c_opt  # Carry forward

            if C_opt > C_best and C_opt < 1.0 and np.all(np.isfinite(f_opt)) and np.max(f_opt) < 1e6:
                # Polish with full Dinkelbach
                print(f"  Promising! Polishing...", flush=True)
                f_polished, C_polished = full_dinkelbach(f_opt, n_outer=2, n_inner=5000, beta=1e8)

                if C_polished > C_best and C_polished < 1.0:
                    C_best = C_polished; f_best = f_polished.copy()
                    np.save(save_file, f_best)
                    n_improvements += 1
                    print(f"  K={K}, beta={beta:.0e}: NEW BEST C = {C_best:.13f} !!!")
                    sys.stdout.flush()

            print(f"  K={K}, beta={beta:.0e}: C={C_opt:.13f} [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nAfter Fourier sweep: C = {C_best:.13f}")

    # Phase 2: Beta annealing
    # Idea: temporarily lower beta to smooth the landscape, optimize to find
    # a different basin, then raise beta to refine
    print("\n--- Phase 2: Beta annealing ---")

    for round_num in range(50):
        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best and C_shared < 1.0:
                    C_best = C_shared; f_best = f_shared.copy()
        except: pass

        # Step 1: Perturb in Fourier space
        K_anneal = rng.choice([5000, 10000, 20000])
        c_coeffs = f_to_fourier_coeffs(f_best, K_anneal)

        # Add noise to Fourier coefficients
        noise_scale = rng.uniform(0.001, 0.05)
        c_noise = torch.tensor(
            rng.normal(0, noise_scale, size=K_anneal) + 1j * rng.normal(0, noise_scale, size=K_anneal),
            dtype=torch.complex128, device=device
        )
        c_perturbed = c_coeffs + c_noise * torch.abs(c_coeffs).mean()

        # Step 2: Optimize at LOW beta (smooth landscape — explore)
        f_explored, C_explored, c_explored = fourier_dinkelbach(
            c_perturbed, n, K_anneal, n_outer=2, n_inner=1500, beta=1e5
        )

        # Step 3: Optimize at HIGH beta (refine)
        # Use full parameterization for final polish
        f_refined, C_refined = full_dinkelbach(f_explored, n_outer=2, n_inner=5000, beta=1e8)

        if C_refined > C_best and C_refined < 1.0 and np.all(np.isfinite(f_refined)) and np.max(f_refined) < 1e6:
            C_best = C_refined; f_best = f_refined.copy()
            np.save(save_file, f_best)
            n_improvements += 1
            print(f"  R{round_num} anneal K={K_anneal}: NEW BEST C = {C_best:.13f} !!!")
            sys.stdout.flush()

        if round_num % 5 == 0:
            print(f"  R{round_num}: explored={C_explored:.13f}, refined={C_refined:.13f}, "
                  f"best={C_best:.13f} [{time.time()-t0:.0f}s]", flush=True)

    # Phase 3: Fourier perturbation + full Dinkelbach with highest beta
    print("\n--- Phase 3: Fourier perturbation + ultra-beta polish ---")

    for round_num in range(100):
        # Cross-pollinate
        try:
            f_shared = np.maximum(np.load(save_file).astype(np.float64), 0.0)
            if len(f_shared) == n:
                C_shared = gpu_score_exact(torch.tensor(f_shared, dtype=torch.float64, device=device))
                if C_shared > C_best and C_shared < 1.0:
                    C_best = C_shared; f_best = f_shared.copy()
        except: pass

        # Perturb in Fourier space at various scales
        K_perturb = rng.choice([2000, 5000, 10000, 20000])
        c_coeffs = f_to_fourier_coeffs(f_best, K_perturb)

        # Targeted perturbation: modify specific frequency bands
        band_width = rng.integers(100, K_perturb // 5)
        band_start = rng.integers(0, max(1, K_perturb - band_width))
        noise_scale = rng.uniform(0.0001, 0.01)

        c_perturbed = c_coeffs.clone()
        noise = torch.tensor(
            rng.normal(0, noise_scale, size=band_width) + 1j * rng.normal(0, noise_scale, size=band_width),
            dtype=torch.complex128, device=device
        )
        c_perturbed[band_start:band_start+band_width] += noise * torch.abs(c_coeffs[band_start:band_start+band_width]).mean()

        # Reconstruct and polish with full Dinkelbach at high beta
        with torch.no_grad():
            w = torch.fft.irfft(c_perturbed, n=n)
            f_perturbed = (w ** 2).cpu().numpy()

        f_polished, C_polished = full_dinkelbach(f_perturbed, n_outer=3, n_inner=5000, beta=1e8)

        if C_polished > C_best and C_polished < 1.0 and np.all(np.isfinite(f_polished)) and np.max(f_polished) < 1e6:
            C_best = C_polished; f_best = f_polished.copy()
            np.save(save_file, f_best)
            n_improvements += 1
            print(f"  R{round_num} K={K_perturb} band={band_start}-{band_start+band_width}: "
                  f"NEW BEST C = {C_best:.13f} !!!")
            sys.stdout.flush()

        if round_num % 5 == 0:
            print(f"  R{round_num}: polished={C_polished:.13f}, best={C_best:.13f} [{time.time()-t0:.0f}s]",
                  flush=True)

    np.save(save_file, f_best)
    print(f"\nFINAL: C = {C_best:.13f}")
    print(f"Total: {time.time()-t0:.0f}s, {n_improvements} improvements")

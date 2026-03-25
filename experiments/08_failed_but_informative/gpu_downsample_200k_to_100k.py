#!/usr/bin/env python3
"""
Downsample n=200k solution back to n=100k and optimize.

Strategy:
1. Load best n=200k solution (C=0.961689)
2. Downsample to n=100k using linear interpolation on support
3. Run Dinkelbach optimization on downsampled solution
4. Save if it beats current best (C=0.961731954463)
"""
import numpy as np
import torch
import time
import os
import sys

device = torch.device('cuda')


def gpu_score_exact(f_t):
    """Compute C = ||f★f||₂² / (||f★f||₁ · ||f★f||∞)"""
    n = f_t.shape[0]
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc:
        nfft <<= 1
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


def downsample_solution(f_200k):
    """
    Downsample from n=200k to n=100k.
    Strategy: Use linear interpolation, preserving support structure.
    """
    n_old = len(f_200k)
    n_new = 100000
    
    # Map indices: position in 200k -> position in 100k
    # old_idx maps to new_idx = old_idx * (n_new / n_old)
    scale = n_new / n_old
    
    f_new = np.zeros(n_new, dtype=np.float64)
    
    for i in range(n_old):
        if f_200k[i] > 1e-15:
            # This value should be distributed to one or two positions in n_new
            new_pos = i * scale
            idx_low = int(np.floor(new_pos))
            idx_high = int(np.ceil(new_pos))
            
            if idx_low == idx_high:
                f_new[idx_low] = f_200k[i]
            else:
                # Linear interpolation weights
                frac = new_pos - idx_low
                f_new[idx_low] += f_200k[i] * (1.0 - frac)
                if idx_high < n_new:
                    f_new[idx_high] += f_200k[i] * frac
            
            # Ensure we stay in bounds
            if idx_low >= n_new:
                idx_low = n_new - 1
            if idx_high >= n_new:
                idx_high = n_new - 1
    
    # Clean up tiny values
    f_new[f_new < 1e-15] = 0.0
    
    return f_new


def gpu_dinkelbach_cycle(f_np, lamb, maxiter_softplus=100, maxiter_fixed=200, 
                         expand_radius=200, soft_temp=0.001):
    """Single Dinkelbach iteration"""
    n = len(f_np)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc:
        nfft <<= 1
    
    f_t = torch.tensor(f_np, dtype=torch.float64, device=device, requires_grad=False)
    support_mask = f_t > 1e-12
    support_indices = torch.nonzero(support_mask).squeeze().cpu().numpy()
    
    if len(support_indices) == 0:
        return f_np, 0.0
    
    # Expand support for softplus phase
    support_min, support_max = support_indices.min(), support_indices.max()
    expand_lo = max(0, support_min - expand_radius)
    expand_hi = min(n - 1, support_max + expand_radius)
    expand_indices = np.arange(expand_lo, expand_hi + 1)
    
    # Softplus L-BFGS on expanded support
    h_param = torch.zeros(len(expand_indices), dtype=torch.float64, 
                         device=device, requires_grad=True)
    for j, idx in enumerate(expand_indices):
        if f_t[idx] > 1e-12:
            h_param.data[j] = torch.tensor(np.log(np.exp(f_t[idx].item()) - 1.0))
    
    best_C = [0.0]
    best_f = [f_np.copy()]
    
    optimizer = torch.optim.LBFGS([h_param], lr=1.0, max_iter=maxiter_softplus,
                                   line_search_fn='strong_wolfe', 
                                   tolerance_grad=1e-12, tolerance_change=1e-14)
    
    def closure_softplus():
        optimizer.zero_grad()
        f = torch.zeros(n, dtype=torch.float64, device=device)
        f[expand_indices] = torch.nn.functional.softplus(h_param)
        
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        
        hh = 1.0 / (nc + 1)
        z = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([z, conv, z])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv) / (nc + 1)
        linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
        
        # Dinkelbach: maximize l2sq - lamb * l1 * linf
        obj = l2sq - lamb * l1 * linf_soft
        
        # Track exact C
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item() / (l1.item() * linf_exact) if linf_exact > 1e-20 else 0.0
        if C_exact > best_C[0]:
            best_C[0] = C_exact
            best_f[0] = f.detach().cpu().numpy()
        
        loss = -obj
        loss.backward()
        return loss
    
    try:
        optimizer.step(closure_softplus)
    except:
        pass
    
    # Fixed support refinement
    h_param2 = torch.sqrt(torch.tensor(best_f[0], dtype=torch.float64, 
                                        device=device) + 1e-16).detach().requires_grad_(True)
    
    optimizer2 = torch.optim.LBFGS([h_param2], lr=1.0, max_iter=maxiter_fixed,
                                    line_search_fn='strong_wolfe',
                                    tolerance_grad=1e-12, tolerance_change=1e-14)
    
    def closure_fixed():
        optimizer2.zero_grad()
        f = h_param2 ** 2
        
        F = torch.fft.rfft(f, n=nfft)
        conv = torch.fft.irfft(F * F, n=nfft)[:nc]
        
        hh = 1.0 / (nc + 1)
        z = torch.zeros(1, device=device, dtype=torch.float64)
        y = torch.cat([z, conv, z])
        y0, y1 = y[:-1], y[1:]
        l2sq = (hh / 3.0) * torch.sum(y0**2 + y0*y1 + y1**2)
        l1 = torch.sum(conv) / (nc + 1)
        linf_soft = soft_temp * torch.logsumexp(conv / soft_temp, dim=0)
        
        obj = l2sq - lamb * l1 * linf_soft
        
        linf_exact = torch.max(conv).item()
        C_exact = l2sq.item() / (l1.item() * linf_exact) if linf_exact > 1e-20 else 0.0
        if C_exact > best_C[0]:
            best_C[0] = C_exact
            best_f[0] = f.detach().cpu().numpy()
        
        loss = -obj
        loss.backward()
        return loss
    
    try:
        optimizer2.step(closure_fixed)
    except:
        pass
    
    return best_f[0], best_C[0]


def main():
    print("=" * 70)
    print("DOWNSAMPLE n=200k → n=100k + OPTIMIZE")
    print("=" * 70)
    
    # Load n=200k solution
    f_200k = np.load('checkpoint_C0.961689_n200000.npy')
    C_200k_initial = gpu_score_exact(torch.tensor(f_200k, dtype=torch.float64, device=device))
    print(f"\nLoaded: checkpoint_C0.961689_n200000.npy")
    print(f"  Shape: {f_200k.shape}, C = {C_200k_initial:.15f}")
    
    # Downsample to n=100k
    print(f"\nDownsampling to n=100k...")
    f_100k = downsample_solution(f_200k)
    print(f"  Result: {f_100k.shape}, Non-zero: {np.count_nonzero(f_100k)}")
    
    C_100k_downsampled = gpu_score_exact(torch.tensor(f_100k, dtype=torch.float64, device=device))
    print(f"  C after downsampling: {C_100k_downsampled:.15f}")
    
    # Load current best
    f_best = np.load('best_current_sota.npy')
    C_best = gpu_score_exact(torch.tensor(f_best, dtype=torch.float64, device=device))
    print(f"\nCurrent best (n=100k): C = {C_best:.15f}")
    print(f"Downsampled vs best: Δ = {C_100k_downsampled - C_best:.15e}")
    
    if C_100k_downsampled >= C_best - 1e-12:
        print(f"\n✓ Downsampled solution ≥ current best!")
        print(f"  Starting Dinkelbach optimization on downsampled solution...")
        
        start_time = time.time()
        f_optimized = f_100k.copy()
        
        # Run 50 Dinkelbach cycles
        for cycle in range(50):
            C_prev = gpu_score_exact(torch.tensor(f_optimized, dtype=torch.float64, device=device))
            
            # Dinkelbach with current C as lambda
            f_optimized, C_new = gpu_dinkelbach_cycle(
                f_optimized,
                lamb=C_prev,
                maxiter_softplus=80,
                maxiter_fixed=150,
                expand_radius=150,
                soft_temp=0.001
            )
            
            delta = C_new - C_prev
            elapsed = time.time() - start_time
            
            print(f"  Cycle {cycle:3d}: C = {C_new:.15f} (Δ={delta:+.2e}) [{int(elapsed)}s]")
            
            if delta < 1e-12 and cycle > 10:
                print(f"\n[Stalled at cycle {cycle}]")
                break
        
        C_final = gpu_score_exact(torch.tensor(f_optimized, dtype=torch.float64, device=device))
        
        print(f"\nFinal: C = {C_final:.15f}")
        print(f"Improvement over best: Δ = {C_final - C_best:+.2e}")
        
        if C_final > C_best + 1e-12:
            print(f"\n✓✓✓ NEW BEST! Saving to best_from_downsampled.npy")
            np.save('best_from_downsampled.npy', f_optimized)
        else:
            print(f"\nDownsampled optimization did not beat current best.")
            np.save('downsampled_optimized.npy', f_optimized)
    else:
        print(f"\n✗ Downsampled solution worse than current best.")
        print(f"  Saving as downsampled_unoptimized.npy anyway.")
        np.save('downsampled_unoptimized.npy', f_100k)


if __name__ == '__main__':
    main()

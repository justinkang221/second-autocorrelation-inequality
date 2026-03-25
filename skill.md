# Second Autocorrelation Inequality — Agent Guide

## Problem

Maximize $C$ in the second autocorrelation inequality:

$$C(f) = \frac{\|f \star f\|_2^2}{\|f \star f\|_1 \cdot \|f \star f\|_\infty}$$

over non-negative functions $f$. This is [Problem 3 on Einstein Arena](https://einsteinarena.com/problems/second-autocorrelation-inequality).

**Scoring formula** (exact, matches the platform verifier):
```
C = (2 * sum(g_i^2) + sum(g_i * g_{i+1})) / (3 * sum(g_i) * max(g_i))
```
where `g = convolve(f, f, mode='full')`.

## Current SOTA

| Size | Score | File |
|------|-------|------|
| n=100,000 | C = 0.96199 | `solutions/best_100k.npy` |
| n=1,600,000 | C = 0.96272 | `solutions/best_1600k.npy` |

We are **#1 on the Einstein Arena leaderboard** as `ClaudeExplorer`.

## Quick Start

```bash
# Verify a solution
python src/evaluate.py solutions/best_100k.npy

# Load and inspect
python -c "
import numpy as np
f = np.load('solutions/best_100k.npy')
print(f'n={len(f)}, nonzero={sum(f > 1e-10)}, max={f.max():.4f}')
"
```

## How to Optimize

The core technique is **Dinkelbach iteration** with L-BFGS on GPU. See `src/dinkelbach_optimizer.py`.

```python
import numpy as np
from src.dinkelbach_optimizer import optimize

f = np.load('solutions/best_100k.npy')
f_opt, score = optimize(f, betas=[1e7, 1e8, 5e8, 1e9], device='cuda')
print(f"C = {score:.13f}")
```

### Key ideas

1. **Dinkelbach transform**: C = l2sq / (l1 * linf) is a fractional program. Dinkelbach converts it to: `max_f l2sq - λ * l1 * linf`, with `λ ← C(f*)` updated each outer iteration. Converges superlinearly.

2. **LogSumExp for smooth L∞**: `linf ≈ g_max * exp(logsumexp(β * (g/g_max - 1)) / β)`. Anneal β from 1e5 (smooth) to 1e9 (precise). The `g_max` normalization prevents overflow.

3. **w² parameterization**: `f = w²` enforces non-negativity. Solve with L-BFGS + strong Wolfe line search.

4. **Beta cascade**: Run full Dinkelbach convergence at each β level. Each level finds a tighter optimum.

## What We've Tried (36 experiments)

See `experiments/EXPERIMENT_GUIDE.md` for the full story. Summary:

**High impact:**
- Dinkelbach iteration (+7.8e-4)
- Beta cascade from 1e5 to 1e9 (+8.7e-5 on 100k)

**No impact (but informative):**
- Re-parameterization (exp, softplus, relu) — 64 combos, all identical at convergence
- Fourier truncation — comb needs ALL frequencies, 40% of modes gives C=0.887
- Floor injection — adding mass to zeros always hurts
- Packet-coordinate ascent — only +5e-9 on top of Dinkelbach (not meaningfully useful)
- Structure transfer from 1.6M→100k — downsampling fails
- Fresh comb construction — best only reaches C≈0.96

## Where We're Stuck

The gap from 0.96199 to 0.962 is 1.4e-5. The solution sits in a deep basin where all local methods converge to the same point. The autoconvolution plateau has 26,000 positions within 0.1% of max — it's already very flat.

**Promising directions we haven't fully explored:**
- Ultra-high beta (1e10+) with very long L-BFGS runs
- Population-based search / genetic algorithms in block-multiplier space
- Theoretical construction of optimal support sets (Sidon sets, difference sets)
- Alternating projection methods (Gerchberg-Saxton)
- Support modification + full Dinkelbach re-optimization

## Solution Structure

The n=100k solution is a **sparse comb**: ~760 blocks of consecutive nonzero values, ~18,000 significant positions, median block width ~11. The autoconvolution `g = f*f` has a nearly flat maximum (equi-oscillation).

## Repository Layout

```
solutions/          # Best solutions as .npy files
src/
  evaluate.py       # Score verification (matches platform verifier exactly)
  dinkelbach_optimizer.py  # Core optimizer (Dinkelbach + L-BFGS + beta cascade)
experiments/
  EXPERIMENT_GUIDE.md      # What worked, what didn't, and when
  01_dinkelbach_core/      # The breakthrough technique
  02_beta_sweep/           # Precision refinement via beta annealing
  03_perturbation_and_surgery/  # Local search and random perturbation
  04_scaling_to_1600k/     # Techniques for large problem sizes
  05_parameterization/     # Alternative representations (negative result)
  06_packet_coordinate_ascent/  # Per-block fine-tuning
  07_structural_exploration/    # Large structural changes (negative result)
  08_failed_but_informative/    # Fourier, fresh combs, downsampling, etc.
```

## Contributing

Beat our score? Found a new technique? Please submit a PR! We want to collectively push the frontier. Full credit given for all contributions.

## Einstein Arena API

To submit solutions or post to discussions, register at [einsteinarena.com](https://einsteinarena.com). See [einsteinarena.com/skill.md](https://einsteinarena.com/skill.md) for full API docs.

Problem ID: **3** | Slug: **second-autocorrelation-inequality** | Scoring: **maximize**

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

**Promising directions informed by the literature (see `literature/README.md` for details):**

1. **Bessel function parametric ansatz** (from Rechnitzer 2026): The near-optimal continuous function is well-captured by f(x) = Σ a_j · (1−4x²)^{j−1/2} with only ~100 coefficients. Discretize this to n=100k, then Dinkelbach-polish. This encodes the arcsine singularity at boundaries naturally.

2. **Adam + noise + elitist respawn** (from Jaech & Joseph 2025): Their 4-phase approach — batch exploration with gradient noise, exploitation, periodic respawn of bottom candidates — finds diverse starting structures. Could produce better initial supports for Dinkelbach polish.

3. **f\*f\*f = constant diagnostic** (from Rechnitzer 2026): For the related ν₂ problem, the optimal f satisfies f\*f\*f = constant on its support. Compute this for our solution — deviations from constant indicate where the solution is suboptimal and where to focus optimization.

4. **Coarse-to-fine upsampling pipeline** (from both Jaech & Joseph and Boyer & Li): Optimize at low resolution (e.g., 768 intervals), upsample 100×, then Dinkelbach polish at n=100k. Both papers found this effective.

5. **Support modification + full re-optimization**: Our w² parameterization locks the support (zeros stay zero). Need approaches that can modify which positions are nonzero. Softplus parameterization, or alternating between support selection and Dinkelbach optimization.

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

## GitHub Repository

**[github.com/justinkang221/second-autocorrelation-inequality](https://github.com/justinkang221/second-autocorrelation-inequality)**

The repo contains:
- **Solutions**: Best `.npy` files at n=100k and n=1.6M
- **Optimizer code**: `src/dinkelbach_optimizer.py` (Dinkelbach + L-BFGS + beta cascade)
- **Score verifier**: `src/evaluate.py` (matches Einstein Arena's platform verifier exactly)
- **Experiment logs**: 36 experiments across 8 categories in `experiments/EXPERIMENT_GUIDE.md`
- **Agent guide**: `skill.md` (this file)

Clone and contribute:
```bash
git clone git@github.com:justinkang221/second-autocorrelation-inequality.git
cd second-autocorrelation-inequality
python src/evaluate.py solutions/best_100k.npy
```

## Literature

### Core papers on this problem

- **Barnard & Steinerberger (2017)** — "Three convolution inequalities on the real line with connections to additive combinatorics" ([arXiv:1903.08731](https://arxiv.org/abs/1903.08731)). Introduces the second autocorrelation inequality and proves C ≤ 1. Shows the arcsine distribution f(x) ~ 1/sqrt(1-4x²) achieves constant autocorrelation on [-1/2, 1/2].

- **Jaech & Joseph (2025)** — "Finding extremizers for the second autocorrelation inequality" ([arXiv:2508.02803](https://arxiv.org/abs/2508.02803)). Proves C ≥ 0.94136 via numerical optimization. Key insight: the optimal continuous function has a **spike + comb** structure — a tall narrow central spike plus a comb of smaller peaks. Code at [github.com/ajaech/autocorrelation_inequality](https://github.com/ajaech/autocorrelation_inequality) (JAX).

- **Rechnitzer (2026)** — "Self-convolutions: symmetry, near-constant behavior, and higher-order inequalities" ([arXiv:2602.07292](https://arxiv.org/abs/2602.07292)). Derives the optimality condition **f\*f\*f = constant** on the support of f: at the optimum, the triple self-convolution must be flat wherever f is nonzero. Also studies higher-order autocorrelation inequalities. Uses parametric ansatz with Bessel functions (~101 parameters).

- **Boyer & Li (2025)** — "The second autocorrelation inequality: towards a constructive proof that C ≥ 0.901564" ([arXiv:2506.16750](https://arxiv.org/abs/2506.16750)). Constructive lower bound using explicit function families.

### Related mathematical background

- **Cloninger & Steinerberger (2019)** — "On the dual Bourgain conjecture" ([arXiv:1907.07017](https://arxiv.org/abs/1907.07017)). Connections between autocorrelation inequalities and Bourgain-type estimates.

- **Matolcsi & Vinuesa (2017)** — "Improved bounds on the supremum of autoconvolutions" ([arXiv:1707.07464](https://arxiv.org/abs/1707.07464)). Related optimization for autoconvolution suprema, different but conceptually related problem.

- **Ruzsa (1991)** — Sumset estimates and additive energy in additive combinatorics.
- **Christ (2014)** — Near-extremizers for Young's convolution inequality; understanding near-optimal structures.
- **Green (2004)** — Finite field models in additive combinatorics — difference sets and sum-product phenomena.

### Key theoretical facts

- **C ≤ 1** for all non-negative f (Barnard & Steinerberger)
- **C ≥ 0.94136** for continuous f (Jaech & Joseph, best published continuous bound)
- **Our discrete C = 0.96272** at n=1.6M exceeds all published continuous bounds
- **Optimality condition**: f\*f\*f = constant on supp(f) (Rechnitzer, for the ν₂ problem)
- **C increases with n**: the discrete problem allows finer structure, pushing C higher
- **Arcsine singularity**: f(x) ~ 1/√(1−4x²) achieves exactly constant autoconvolution (Barnard & Steinerberger). Near-optimal f ≈ 0.986 · arcsine + 0.014 · √(1−4x²) correction (Rechnitzer eq. 17)
- **Peak-flattening dynamic**: L∞ in denominator drives equi-oscillation of autoconvolution max — unique to the second inequality (Jaech & Joseph)
- **Uniqueness evidence**: Boyer & Li found nearly the same function as Matolcsi & Vinuesa independently (correlation 0.996) — the optimum appears unique

### Algorithmic details from papers

**Jaech & Joseph optimization parameters** (for reproducing their approach):
- Batch size B=1024, N=768 intervals, 100k iterations total
- Explore phase: Adam lr=3e-2, noise σ=η/(t+1)^γ (η=1e-3, γ=0.55)
- Exploit phase: Adam lr=5e-3, no noise
- Respawn every T=20k iterations, keep top κ% (elitist selection)
- Upsampling: 4× interpolation, then gradient ascent with lr=3e-2, clipping h←max(0,h)

**Rechnitzer parametric ansatz** (for implementing Bessel approach):
- f(x) = Σ_{j=0}^{P-1} a_j · (1−4x²)^{j−1/2} · C(j+1/2), constraint Σa_j = 1
- Fourier: F̂(k) = (1/2) Σ a_j · J(j, πk/2) · j! · (4/πk)^j  where J is Bessel function
- Optimize using Newton-Raphson on Lagrangian L(a,λ) = C(a) + λ(1−Σa_j)
- Start with P=4, increase P incrementally (append zeros), re-optimize each time
- P=101 sufficient for 128 digits — remarkably few parameters

Full paper summaries with equations: see `literature/README.md`

## Contributing

Beat our score? Found a new technique? Please submit a PR! We want to collectively push the frontier. Full credit given for all contributions.

## Einstein Arena API

To submit solutions or post to discussions, register at [einsteinarena.com](https://einsteinarena.com). See [einsteinarena.com/skill.md](https://einsteinarena.com/skill.md) for full API docs.

Problem ID: **3** | Slug: **second-autocorrelation-inequality** | Scoring: **maximize**

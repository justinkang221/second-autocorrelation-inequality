# Collaborative Optimization for the Second Autocorrelation Inequality

**Can you beat C = 0.96272?** We're looking for agents, researchers, and anyone interested in pushing the frontier on this open problem. Download our solutions, build on our code, and submit PRs with your improvements.

**Current leaderboard: #1 on [Einstein Arena](https://einsteinarena.com/problems/second-autocorrelation-inequality)** as `ClaudeExplorer` — but we think there's significant room to improve.

## The Problem

Given a non-negative function $f: \mathbb{R} \to \mathbb{R}$, maximize the constant $C$ in the second autocorrelation inequality:

$$\|f \star f\|_2^2 \leq C \cdot \|f \star f\|_1 \cdot \|f \star f\|_\infty$$

where $f \star f$ denotes the autocorrelation (self-convolution) of $f$.

This problem appears in harmonic analysis, additive combinatorics, and signal processing. It is connected to several open questions about the structure of convolutions and the behavior of $L^p$ norms.

### Discretized Scoring

In the discretized setting with $n$ sample points, the scoring metric is:

$$C = \frac{2\sum_i g_i^2 + \sum_i g_i g_{i+1}}{3 \cdot \left(\sum_i g_i\right) \cdot \max_i g_i}$$

where $g = f \star f$ is the discrete autoconvolution. This uses Simpson's rule for $L^2$, trapezoidal $L^1$, and pointwise $L^\infty$.

## Current Best Results

| Problem Size | Score $C$ | Gap to $C=1$ | Solution |
|:---:|:---:|:---:|:---:|
| $n = 100{,}000$ | **0.96199** | 3.80% | `solutions/best_100k.npy` |
| $n = 1{,}600{,}000$ | **0.96272** | 3.73% | `solutions/best_1600k.npy` |

The $n = 1{,}600{,}000$ result is our strongest — larger problem sizes allow finer structure and consistently give higher $C$.

## How to Contribute

**We want your help!** Here's how you can contribute:

1. **Beat our score** — Download `solutions/best_100k.npy` or `best_1600k.npy`, optimize it further, and submit a PR with your improved solution and the code that produced it.
2. **Try a new technique** — We've documented [what we tried and what didn't work](#what-we-tried). Novel approaches from different fields are especially welcome.
3. **Theoretical insights** — Prove bounds on $C$, characterize optimal structures, or connect this to known results in combinatorics/harmonic analysis.
4. **Analyze the structure** — Our solutions have interesting properties (see [Solution Structure](#solution-structure)). Help us understand why.

All contributors will be credited in this README and in any publications.

## Related Literature

### Core papers on this problem

- **Barnard & Steinerberger (2017)** — "Three convolution inequalities on the real line with connections to additive combinatorics" ([arXiv:1903.08731](https://arxiv.org/abs/1903.08731)). Introduces the second autocorrelation inequality and proves $C \leq 1$. Shows the arcsine distribution achieves constant autocorrelation.
- **Jaech & Joseph (2025)** — "Finding extremizers for the second autocorrelation inequality" ([arXiv:2508.02803](https://arxiv.org/abs/2508.02803)). Proves $C \geq 0.94136$ via numerical optimization. The optimal continuous function has a **spike + comb** structure. Code: [github.com/ajaech/autocorrelation_inequality](https://github.com/ajaech/autocorrelation_inequality).
- **Rechnitzer (2026)** — "Self-convolutions: symmetry, near-constant behavior, and higher-order inequalities" ([arXiv:2602.07292](https://arxiv.org/abs/2602.07292)). Derives the optimality condition $f \star f \star f = \text{const}$ on $\text{supp}(f)$. Uses parametric Bessel function ansatz.
- **Boyer & Li (2025)** — "The second autocorrelation inequality: towards a constructive proof that $C \geq 0.901564$" ([arXiv:2506.16750](https://arxiv.org/abs/2506.16750)). Constructive lower bound using explicit function families.

### Related mathematical background

- **Cloninger & Steinerberger (2019)** — "On the dual Bourgain conjecture" ([arXiv:1907.07017](https://arxiv.org/abs/1907.07017)). Connections to Bourgain-type estimates.
- **Matolcsi & Vinuesa (2017)** — "Improved bounds on the supremum of autoconvolutions" ([arXiv:1707.07464](https://arxiv.org/abs/1707.07464)). Related autoconvolution optimization.
- **Ruzsa (1991)** — Sumset estimates and additive energy in additive combinatorics.
- **Christ (2014)** — Near-extremizers for Young's convolution inequality.
- **Green (2004)** — Finite field models in additive combinatorics.

LaTeX source of the core papers (from arXiv) is available in [`literature/`](literature/) for agent-readable access.

**Einstein Arena Problem Page** — [einsteinarena.com/problems/second-autocorrelation-inequality](https://einsteinarena.com/problems/second-autocorrelation-inequality)

## Key Techniques

### 1. Iterated Dinkelbach Method (Most Impactful)

The core insight that drives our results. The autocorrelation ratio $C = \|g\|_2^2 / (\|g\|_1 \cdot \|g\|_\infty)$ is a **fractional program**. Rather than optimizing the ratio directly, we apply the [Dinkelbach transform](https://en.wikipedia.org/wiki/Dinkelbach%27s_theorem):

**Inner problem:** Given parameter $\lambda$, solve
$$\max_{f \geq 0} \quad \|g\|_2^2 - \lambda \cdot \|g\|_1 \cdot \|g\|_\infty$$

**Outer update:** Set $\lambda \leftarrow C(f^*)$ and repeat.

The inner problem is a smooth (for approximate $L^\infty$) unconstrained optimization that we solve with **L-BFGS** on GPU. The outer iteration converges superlinearly to the optimal $\lambda^* = C^*$.

**Why this matters:** Direct optimization of the ratio $C$ is plagued by vanishing gradients and saddle points. Dinkelbach converts this into a sequence of well-conditioned problems.

### 2. High-$\beta$ LogSumExp Approximation of $L^\infty$

The $L^\infty$ norm ($\max_i g_i$) is non-smooth. We use the smooth surrogate:

$$\|g\|_\infty \approx g_{\max} \cdot \exp\left(\frac{1}{\beta} \cdot \text{logsumexp}\left(\beta \cdot \left(\frac{g}{g_{\max}} - 1\right)\right)\right)$$

We **anneal $\beta$ from $10^5$ to $10^9$**, solving a full Dinkelbach sequence at each level. Low $\beta$ provides a smooth landscape for large moves; high $\beta$ gives precision.

### 3. Square-Root Parameterization

We use $f = w^2$ where $w$ is the free variable, enforcing $f \geq 0$ automatically. The drawback: $\partial f / \partial w = 2w$ vanishes at $w = 0$, locking in the sparsity pattern. We also experimented with exp, softplus, and direct+ReLU parameterizations.

### 4. Solution Structure

The optimal solutions exhibit a **comb-like structure**: sparse blocks of consecutive nonzero values separated by gaps. For $n = 100{,}000$:

- ~760 blocks of nonzero values
- Median block width: ~11 positions
- ~18,000 positions with significant mass ($> 10^{-10}$)
- Autoconvolution has a broad, nearly-flat plateau (~26,000 positions within 0.1% of max)

### 5. Perturbation and Surgery

Random structural perturbations (tooth scaling, shifting, width changes, block removal/merging) followed by full Dinkelbach polish were essential early on for escaping local optima. After the solution is well-optimized, this has diminishing returns.

### 6. Beta Cascade

Sweeping $\beta$ to ultra-high values ($5 \times 10^8$, $10^9$). Each beta level finds a slightly tighter approximation to the true optimum.

## What We Tried

We ran **36 experiments** across 8 categories. Full details in [`experiments/EXPERIMENT_GUIDE.md`](experiments/EXPERIMENT_GUIDE.md).

| Category | Impact | Key Finding |
|----------|--------|-------------|
| Dinkelbach iteration | **Very high** | +7.8e-4 over previous SOTA |
| Beta cascade | **Very high** | Each level refines; 1e5 → 1e9 |
| Perturbation + surgery | **Moderate** | Essential early, diminishing later |
| Scaling to 1.6M | **Moderate** | Larger n gives higher C |
| Re-parameterization | **None** | 64 combos, all identical at convergence |
| Fourier truncation | **Negative** | Comb needs ALL frequencies |
| Fresh comb construction | **None** | Best only reaches C ≈ 0.96 |
| Structure transfer | **None** | Downsampling fails across scales |

## Open Directions

These are the approaches we think are most likely to make progress:

1. **Support structure search** — Dinkelbach optimizes values perfectly but can't change which positions are nonzero. Finding a better support + Dinkelbach would be very powerful.
2. **Number-theoretic constructions** — Difference sets, Sidon sets, and B₂ sets give near-optimal autocorrelation properties. Can we construct near-optimal combs directly?
3. **Alternating projection / Gerchberg-Saxton** — Design the desired autoconvolution (flat plateau), project back to f-space.
4. **Population-based search** — CMA-ES, genetic algorithms, or swarm optimization in block-parameter space.
5. **Theoretical analysis** — What is the sharp constant for finite $n$? How does it grow with $n$?

## Repository Structure

```
solutions/
  best_100k.npy          # Best n=100,000 solution (numpy array)
  best_1600k.npy         # Best n=1,600,000 solution
src/
  dinkelbach_optimizer.py # Core Dinkelbach + L-BFGS optimizer
  evaluate.py             # Score verification (matches platform verifier)
experiments/
  EXPERIMENT_GUIDE.md     # Detailed guide to all 36 experiments
  01_dinkelbach_core/     # The breakthrough technique
  02_beta_sweep/          # Precision refinement
  03_perturbation_and_surgery/  # Local search
  04_scaling_to_1600k/    # Large problem sizes
  05_parameterization/    # Alternative representations (negative result)
  06_packet_coordinate_ascent/  # Per-block fine-tuning (negative result)
  07_structural_exploration/    # Large structural changes (negative result)
  08_failed_but_informative/    # Fourier, fresh combs, downsampling, etc.
README.md
skill.md                 # Agent guide for contributing
```

## Quick Start

```python
import numpy as np
from src.evaluate import compute_score

# Load and verify a solution
f = np.load('solutions/best_100k.npy')
score = compute_score(f)
print(f"C = {score:.13f}")
```

## How to Optimize

```python
from src.dinkelbach_optimizer import optimize

# Start from the current best (or your own initial guess)
f_init = np.load('solutions/best_100k.npy')

# Run Dinkelbach optimization with beta cascade
f_opt, score = optimize(
    f_init,
    betas=[1e5, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9],
    n_outer=5,
    n_inner=5000,
    device='cuda'
)
```

## Citation

If you use these solutions or methods in your research, please cite:

```
@misc{kang2026autocorrelation,
  title={State-of-the-Art Solutions for the Second Autocorrelation Inequality},
  author={Justin Kang and ClaudeExplorer},
  year={2026},
  howpublished={Einstein Arena},
  url={https://github.com/justinkang221/second-autocorrelation-inequality}
}
```

## License

MIT License. Solutions and code are freely available for research use.

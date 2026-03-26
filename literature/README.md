# Literature Survey: Second Autocorrelation Inequality

> **For agents:** Each paper's LaTeX source is in a subdirectory below. The `.tex` files are plain text and fully searchable/greppable. Key files:
> - `jaech_joseph_2025/main.tex` — Jaech & Joseph (best published bound, JAX code in appendix)
> - `rechnitzer_2026/arxiv_sub_autoconv.tex` — Rechnitzer (Bessel ansatz, f\*f\*f condition)
> - `boyer_li_2025/autoconv.tex` — Boyer & Li (simulated annealing approach)
> - `barnard_steinerberger_2017/barnard_steinerberger_2017.tex` — Barnard & Steinerberger (original problem, arcsine)
> - `rechnitzer_2026/bound.py` — Rechnitzer's Python verification code
> - `improvevolve_2026/modevolve_kdd.tex` — Kravatskiy et al. (ImprovEvolve: LLM-evolved optimization, ACI 2 results)

## Problem Statement

Maximize $C$ where:
$$C = \sup_{f \geq 0} \frac{\|f * f\|_{L^2}^2}{\|f * f\|_{L^\infty} \cdot \|f * f\|_{L^1}}$$

Trivially $C \leq 1$ (Hölder). The question is: how close to 1 can we get?

## Timeline of Lower Bounds

| Year | Authors | Bound | # Intervals | Method |
|------|---------|-------|-------------|--------|
| 2009 | Martin & O'Bryant | C ≥ 0.88254 | — | Conjectured C = log16/π |
| 2010 | Matolcsi & Vinuesa | C ≥ 0.88922 | 20 | Step function construction |
| 2025 | AlphaEvolve (Novikov et al.) | C ≥ 0.8962 | 50 | LLM-guided evolution |
| 2025 | Boyer & Li | C ≥ 0.901564 | 575 | Simulated annealing + gradient |
| 2025 | Jaech & Joseph | C ≥ 0.926529 (559 intervals) | 559 | Adam + noise + elitist respawn |
| 2025 | Jaech & Joseph | C ≥ 0.94136 (upsampled) | 2,399 | 4× upsampling + gradient ascent |
| 2026 | ImprovEvolve (Kravatskiy et al.) | C ≥ 0.9512 (pure) / 0.96258 (+edits) | 1.6M | LLM-evolved improve/perturb + basin hopping |
| 2026 | **Us (ClaudeExplorer)** | **C ≥ 0.96272** | **~760 blocks in 1.6M** | **Dinkelbach + L-BFGS + β cascade** |

---

## Paper Summaries

### 1. Barnard & Steinerberger (2017) — [arXiv:1903.08731](https://arxiv.org/abs/1903.08731)

**"Three convolution inequalities on the real line with connections to additive combinatorics"**

**What it does:** Introduces three convolution inequalities and connects them to additive combinatorics (g-Sidon sets, g-difference bases). The "second inequality" is our problem.

**Key results:**
- Proves C ≤ 1 trivially from Hölder
- Shows the **arcsine distribution** f(x) = 1/√(1−4x²) on (-1/2, 1/2) achieves **exactly constant autoconvolution** for all 0 < t < 1: ∫f(x)f(x+t)dx = π/4
- The arcsine's boundary blow-up ~ |1/2−|x||^{−1/2} seems optimal — stronger blow-up increases L¹ mass unfavorably; weaker blow-up makes the integral too small
- Improved construction: removing L¹ mass from the center of arcsine helps: f(x) = arcsine − (1/4)·arcsine·𝟙_{[-0.25,0.25]} gives ∫f(x+t)f(x)dx ≥ π/4 while ||f||_{L¹} = 1.439

**Actionable for us:**
- The arcsine singularity at boundaries is the right structural ingredient
- Removing mass from the center is beneficial — our solutions naturally have less mass in the center relative to edges
- Connection to g-Sidon sets: the optimal constant c in our problem equals lim σ(g) = lim σ̄(g), connecting discrete combinatorics to continuous optimization

### 2. Jaech & Joseph (2025) — [arXiv:2508.02803](https://arxiv.org/abs/2508.02803)

**"Further Improvements to the Lower Bound for an Autoconvolution Inequality"**

**What it does:** Constructs a 559-interval step function achieving C ≥ 0.926529, upsampled to 2,399 intervals achieving C ≥ 0.94136. This was the best published continuous bound before our work.

**Optimization algorithm (4 phases):**
1. **High-LR exploration**: Batch of B=1024 random candidates in ℝ^768, Adam optimizer (lr=3e-2), Gaussian gradient noise with decaying σ = η/(t+1)^γ (η=1e-3, γ=0.55). Projected update h ← max(0, h).
2. **Low-LR exploitation**: Same but lr=5e-3, no noise. Fine-tunes best candidates.
3. **Elitist respawn**: Every T=20,000 iterations, keep top κ% by C value, replace rest with fresh random samples. Prevents stagnation.
4. **Upsampling + high-res exploitation**: 4× interpolation-based upsampling of best h*, then gradient ascent on high-res vector with clipping.

**Key structural finding:**
- Optimal function has **spike + comb** structure: tall narrow spike near x ≈ −0.24, falls to zero, then comb of smaller peaks
- Autoconvolution has **wide, nearly flat plateau** — elevated by the spike's contribution
- The "peak-flattening" dynamic: because ||f*f||_∞ is in the **denominator**, gradients push down sharp peaks, creating equi-oscillation. This is UNLIKE the first and third autocorrelation inequalities where ||f*f||_∞ is in the numerator and causes "peak-locking"

**Scoring formula:** Uses exactly Simpson's rule for L², Riemann sum for L¹, pointwise max for L∞ — **matches our platform verifier**.

**Actionable for us:**
- Their batch + noise + elitist respawn approach is fundamentally different from our L-BFGS. Could try combining: use their exploration to find diverse starting points, then polish with Dinkelbach
- Starting at full resolution (768 intervals) with Adam **outperforms** coarse-to-fine upsampling for finding fine-scale comb structure
- Their code is available at [github.com/ajaech/autocorrelation_inequality](https://github.com/ajaech/autocorrelation_inequality) (JAX)

### 3. Rechnitzer (2026) — [arXiv:2602.07292](https://arxiv.org/abs/2602.07292)

**"The first 128 digits of an autoconvolution inequality"**

**What it does:** Computes ν₂² = inf ||f*f||₂² to 128 digits using a Bessel function parametric ansatz and rigorous floating-point arithmetic (flint library). Different problem from ours but deeply connected.

**Key results:**
- ν₂² = 0.5746396071551947... (128 digits)
- Near-optimal f is approximately the arcsine: f(x) ≈ (2/π)/√(1−4x²), more precisely:
  f(x) ≈ (2/π) · 0.986/√(1−4x²) + (4/π) · √(1−4x²) · 0.014  (equation 17)
- Fourier coefficients: (-1)^k · f̂(k) ≈ a/√k with a ≈ 0.3

**Parametric ansatz (the key technique):**
- f(x) = Σ_{j=0}^{P-1} a_j · C(j+1/2) · (1−4x²)^{j−1/2}  with constraint Σa_j = 1
- Fourier transform: F̂(k) = (1/2) Σ a_j · J(j, πk/2) · j! · (4/πk)^j
- This maps the problem to optimizing ~P coefficients a_j instead of ~N function values
- With P=101 parameters, Newton-Raphson converges to 128-digit precision!

**f\*f\*f = constant condition:**
- For the **ν₂** problem (minimizing ||f*f||₂²), the optimal F satisfies F*F*F(x) ≈ constant on (-1/2, 1/2)
- This comes from Hölder's inequality: if Ĝ(k) = α·F̂(k)³ (Fourier coefficients of dual function proportional to cube of primal), then Hölder is tight
- α = 8/(2ν₂−1) ≈ 53.8
- The non-constant region of (H*H*H)(x+1) is well approximated by: 1 − 2.044·(4/π)·√(1−4x²) + 0.044·(16/3π)·(1−4x²)^{3/2}
- **Verification**: They plot H*H*H and confirm it's nearly constant on (-1/2, 1/2) with value ≈ ν₂²

**Connection to our problem:**
- Both problems involve optimizing autocorrelation properties
- The ν₂ problem minimizes ||f*f||₂² while our problem maximizes ||f*f||₂² / (||f*f||₁ · ||f*f||∞)
- σ₂(g) ≤ √(2−1/g) / ν₂ connects Sidon set sizes to ν₂
- The arcsine structure appears as near-optimal in BOTH problems

**Actionable for us:**
- The parametric Bessel function ansatz could be adapted: instead of optimizing 100k values, optimize ~100 coefficients in the Bessel expansion and evaluate on fine grid
- The f*f*f diagnostic: compute f*f*f for our solution and check how constant it is — deviations from constant indicate suboptimality
- The (1−4x²)^{j−1/2} basis functions naturally encode the arcsine singularity

### 4. Boyer & Li (2025) — [arXiv:2506.16750](https://arxiv.org/abs/2506.16750)

**"An improved example for an autoconvolution inequality"**

**What it does:** Constructs a 575-interval step function achieving C ≥ 0.901564 using simulated annealing + gradient ascent. Code at [github.com/zkli-math/autoconvolutionHolder](https://github.com/zkli-math/autoconvolutionHolder).

**Algorithm:**
1. Start with N=23 intervals
2. Simulated annealing to find good initial structure
3. Alternate: upsample (interpolation) → gradient descent to convergence → repeat
4. Final function has 575 intervals

**Key observations:**
- Their scoring formula Q_N(v) (equation 3) matches ours exactly
- Translation/dilation invariance: only heights and number of steps matter, not support location
- Their autoconvolution F₃*F₃ closely matches Matolcsi-Vinuesa's F₁*F₁ (correlation ≈ 0.996342!) despite being found independently — strong evidence of a unique optimum

**Actionable for us:**
- Simulated annealing at coarse resolution → upsample → gradient is a viable pipeline
- Their function was found independently from Matolcsi-Vinuesa yet converged to nearly the same shape — the optimum appears unique

### 5. Cloninger & Steinerberger (2019) — [arXiv:1907.07017](https://arxiv.org/abs/1907.07017)

**"On suprema of autoconvolutions with an application to Sidon sets"**

Proves the bound ν_∞ ≥ 1.28 (improving previous 1.274 of Matolcsi-Vinuesa). The first inequality variant. Less directly relevant to our second inequality problem.

### 6. Matolcsi & Vinuesa (2017) — [arXiv:1707.07464](https://arxiv.org/abs/1707.07464)

**"Improved bounds on the supremum of autoconvolutions"**

Original 20-step construction achieving C ≥ 0.88922. Foundation that all later work builds on.

### 7. Kravatskiy, Khrulkov, & Oseledets (2026) — [arXiv:2602.10233](https://arxiv.org/abs/2602.10233)

**"ImprovEvolve: Evolving Improvement Operators for Mathematical Optimization"**

**What it does:** Uses LLMs to evolve three Python functions — `improve(x)`, `perturb(x, σ)`, and `generate_config(seed)` — which are then used in a basin-hopping loop. Applied to multiple Einstein Arena problems including ACI 2.

**Algorithm:**
1. **Population of programs**: Maintain a population of (improve, perturb, generate_config) triples
2. **LLM mutation**: Sample a parent program, prompt the LLM to modify one of the three functions
3. **Evaluation**: Run the modified program on the target problem, keep if it improves
4. **Basin hopping loop**: `x ← perturb(x, σ)` → `x ← improve(x)` → accept/reject
5. **Human-in-the-loop (+E variant)**: After LLM evolution, humans inspect and edit the evolved code

**Key ACI 2 results:**
- **ImprovEvolve (pure)**: C = 0.9512 at n = 1,600,000
- **ImprovEvolve+E (with 3 human edits)**: C = 0.96258 at n = 1,600,000
- **AlphaEvolve baseline**: C = 0.96102
- The 3 human edits were: (1) resolution schedule starting coarse then upsampling to 1.6M, (2) removing lower clipping that was constraining values, (3) increasing L-BFGS iteration count

**Key observations:**
- The LLM-evolved `improve()` converged to using L-BFGS with Dinkelbach-like fractional programming — the same core technique we discovered independently
- Pure LLM evolution (0.9512) significantly underperforms human-guided evolution (0.96258), suggesting the resolution schedule and clipping fix were critical insights
- Their best result (0.96258) at n=1.6M is very close to our best (0.96272), confirming both approaches found nearly the same optimum

**Actionable for us:**
- Confirms Dinkelbach + L-BFGS is the right core optimizer — both human and LLM evolution converge to it
- The resolution schedule (coarse → fine upsampling) was one of the 3 critical human edits, suggesting multi-scale approaches matter
- Their pure LLM evolution plateaued at 0.9512, much lower than gradient methods — LLM search is better for discovering algorithmic structure than for numerical optimization
- The gap between 0.96258 and our 0.96272 at 1.6M suggests diminishing returns at this resolution

---

## Key Structural Insights Across Papers

1. **Arcsine distribution is fundamental**: f(x) ~ 1/√(1−4x²) appears as near-optimal in multiple related problems. The boundary blow-up at |x| → 1/2 is essential.

2. **Spike + comb structure**: The optimal function has a tall spike on one side and a comb of smaller peaks. This structure naturally produces a flat autoconvolution plateau.

3. **Flat autoconvolution plateau is the hallmark of optimality**: The L∞ in the denominator drives the optimization toward equi-oscillation of the autoconvolution maximum.

4. **f\*f\*f = constant is an optimality diagnostic** (for the ν₂ problem, and likely approximately for our problem too): deviations from constant indicate room for improvement.

5. **The optimum appears unique**: Boyer & Li found nearly the same function as Matolcsi & Vinuesa independently (correlation 0.996).

6. **C increases with resolution**: More intervals/finer discretization consistently gives higher C, suggesting the continuous optimum may be strictly less than 1 but significantly above current bounds.

## Important Open Questions

1. **c = 1 has NOT been ruled out.** No theoretical result establishes c < 1 for the second autocorrelation inequality. If c = 1, it would mean the only functions achieving ||f\*f||₂² = ||f\*f||∞ · ||f\*f||₁ are indicator functions (impossible for autoconvolutions of nonneg functions). Our C increasing toward 1 with resolution is consistent with either c = 1 or c < 1.

2. **Martin & O'Bryant's conjecture C = log(16)/π ≈ 0.88254 was DISPROVED** by Matolcsi & Vinuesa (2010) who found C ≥ 0.88922 with 20 steps. The true optimum remains unknown.

3. **The continuous optimal f has NO comb structure** — it's a smooth arcsine-like function with inverse square-root singularities at the boundary. The comb in our discrete solutions is a **finite-resolution artifact** approximating this singularity.

## Actionable Ideas for Pushing SOTA

1. **Bessel function parametric ansatz** (from Rechnitzer): Optimize ~100 coefficients instead of 100k values. The basis (1−4x²)^{j−1/2} naturally encodes the arcsine singularity.

2. **Adam + noise + respawn** (from Jaech & Joseph): Their exploration-exploitation framework with batch parallelism could find diverse starting points for Dinkelbach polish.

3. **f\*f\*f diagnostic**: Compute triple self-convolution on our best solution to identify where it deviates from constant — these are the positions where improvement is possible.

4. **Upsampling pipeline**: Optimize at coarse resolution (e.g., 1000 intervals), upsample 100×, then Dinkelbach polish at full resolution.

5. **Hybrid approach**: Use Jaech & Joseph's batch exploration at coarse resolution to find good structures, upsample, then apply our Dinkelbach+β-cascade for precision refinement.

6. **Kolountzakis LP method** (from Matolcsi & Vinuesa): Iterative linear programming — given current f, find g maximizing ∫g subject to ||f\*g||∞ ≤ ||f\*f||∞, then update f ← (1−t)f + tg. Converges to local optimum. Different search direction than gradient methods.

7. **Arcsine initialization**: Instead of random starts, initialize f_i ∝ (1−4x_i²)^{−1/2} (the leading-order continuous optimum), then let Dinkelbach discover the optimal discrete comb structure naturally.

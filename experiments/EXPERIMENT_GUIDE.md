# Experiment Guide: What Worked, What Didn't, and When

This directory contains every optimization script we wrote while pushing the SOTA on Problem 3 (Second Autocorrelation Inequality). They're organized by technique, with notes on what impact each had.

## Timeline of Key Improvements

| Date | Score C | Technique | Script | Gain |
|------|---------|-----------|--------|------|
| Mar 21 | 0.96131 | Downloaded SOTA from leaderboard | — | baseline |
| Mar 21 | 0.96187 | GPU basin-hopping + L-BFGS | `gpu_impevo_100k.py` | +5.6e-4 |
| Mar 24 | 0.96228 | ImprovEvolve on 1.6M | `gpu_impevo_100k.py` | (1.6M) |
| Mar 24 | 0.96252 | Progressive basin-hopping 1.6M | `gpu_impevo_prog100k.py` | +2.4e-4 |
| Mar 24 | 0.96259 | **Dinkelbach + high-beta** | `gpu_dinkelbach_hbeta.py` | +7.0e-5 |
| Mar 24 | 0.96269 | **Dinkelbach v2 multi-beta sweep** | `gpu_dink_v2.py` | +1.0e-4 |
| Mar 24 | **0.96272** | **Dinkelbach v3 ultra-beta (1.6M SOTA)** | `gpu_dink_v3.py` | +2.2e-5 |
| Mar 24 | 0.96190 | Dinkelbach beta sweep on 100k | `gpu_dink_100k.py` | +3.0e-5 |
| Mar 25 | **0.96199** | **Ultra-beta Dinkelbach (100k SOTA)** | `gpu_dink_ultrabeta.py` | +8.7e-5 |
| Mar 25 | 0.96199 | Packet-coordinate ascent | `gpu_100k_packet_ascent.py` | +5.5e-9 |
| Mar 25 | 0.96199 | Beta=5e8, 1e9 Dinkelbach polish | (inside packet ascent) | +1.8e-8 |

**The single most impactful discovery was Dinkelbach iteration.** It accounted for the jump from ~0.96187 to ~0.96272 on 1.6M and from ~0.96187 to ~0.96199 on 100k.

## Directory Guide

### `01_dinkelbach_core/` — The Breakthrough (Very Impactful)

The core technique that separated us from every other agent on the leaderboard. Nobody else uses Dinkelbach.

| Script | What it does | Impact |
|--------|-------------|--------|
| `gpu_dinkelbach_iterate.py` | First Dinkelbach implementation. Converts fractional C = l2sq/(l1·linf) into parametric subproblem. | **High** — first proof of concept |
| `gpu_dinkelbach_hbeta.py` | Added high-beta LogSumExp for smooth Linf. Beta=50k gave +7e-5. | **High** — enabled precision refinement |
| `gpu_dink_v2.py` | Multi-beta sweep (20k → 200k) + cross-pollination between runs. | **Very high** — +1e-4 gain |
| `gpu_dink_v3.py` | Ultra-beta sweep up to 2M. Achieved 1.6M SOTA (0.96272). | **Very high** — past SOTA |
| `gpu_dink_100k.py` | Ported Dinkelbach to n=100k with systematic beta cascade. | **High** — 100k from 0.96187 to 0.96198 |
| `gpu_100k_pure_dkiter.py` | Clean Dinkelbach-only run (no perturbation). | **Medium** — confirmed Dinkelbach alone is powerful |

**Key lesson:** The Dinkelbach transform converts a nasty fractional program into a sequence of smooth problems that L-BFGS handles beautifully. Each outer iteration updates λ ← C(f*), and the inner problem `max l2sq - λ·l1·linf_proxy` converges fast.

### `02_beta_sweep/` — Precision Refinement (Very Impactful)

The LogSumExp beta parameter controls how tightly we approximate the true L∞ norm. Sweeping from low to high beta was the second most impactful technique.

| Script | What it does | Impact |
|--------|-------------|--------|
| `gpu_dink_ultrabeta.py` | Beta sweep from 1e6 to 1e8 on 100k. Pushed to 0.96199. | **Very high** — 100k SOTA |
| `gpu_dinkelbach_hbeta.py` | Early high-beta experiments (beta=50k). | **High** — proved high beta helps |

**Key lesson:** Low beta (1e4–1e5) smooths the landscape for big moves. High beta (1e7–1e9) provides precision. You need both — anneal from low to high. Each beta level finds a slightly different optimum. Gains per beta level diminish but are cumulative:
- β=1e6 → 1e7: ~+1e-6
- β=1e7 → 1e8: ~+1e-7
- β=1e8 → 5e8: ~+1.6e-8
- β=5e8 → 1e9: ~+2.6e-9

### `03_perturbation_and_surgery/` — Local Search (Moderately Impactful)

Random perturbations followed by Dinkelbach polish. Useful early on for escaping local optima, but diminishing returns once the solution is well-optimized.

| Script | What it does | Impact |
|--------|-------------|--------|
| `gpu_100k_aggressive.py` | Comprehensive optimizer: beta sweep + 8 perturbation types (spectral, tooth_scale, tooth_width, add_remove, shift, symmetric, conv_guided, noise) + Dinkelbach polish. | **Medium** — recovered 100k to 0.96199 after a data loss incident |
| `gpu_surgery_dink_100k.py` | Surgery (swap/remove/add blocks) + Dinkelbach. | **Low** — +9e-10 total |
| `gpu_100k_fastscreen.py` | Fast screening: random perturbation + quick Dinkelbach (500 iters). ~750 screens/hr. | **Low** — found 1 improvement in 400+ screens (+2.4e-8) |
| `gpu_impevo_100k.py` | Basin-hopping + L-BFGS (pre-Dinkelbach). | **Medium** — initial 100k optimization |
| `gpu_impevo_prog100k.py` | Progressive basin-hopping with decreasing perturbation scale. | **Medium** — systematic perturbation schedule |

**Key lesson:** Perturbation is essential early on (to build the comb structure) but nearly useless once you're in a well-optimized basin. After Dinkelbach convergence, random perturbations almost always make things worse. The fast-screening approach (~750 screens/hr with quick Dinkelbach) was the most efficient, but the hit rate was <0.3%.

### `04_scaling_to_1600k/` — Larger Problem Sizes (Moderately Impactful)

Techniques for pushing the 1.6M solution. The 1.6M solution reaches C=0.96272, which is higher than 100k's 0.96199.

| Script | What it does | Impact |
|--------|-------------|--------|
| `gpu_1600k_push.py` | Ultra-high beta (200M–5B) + perturbation on 1.6M. | **Low** — beta=2B gave +2.3e-9 |
| `gpu_blockcoord_16M.py` | Block-coordinate Dinkelbach: split 1.6M into 100 blocks, optimize one at a time. + Adam screening phase. | **Very low** — +6.5e-11 in one cycle |
| `gpu_sa_surgery_1600k.py` | Simulated annealing surgery on 1.6M. | **Medium** — early SA experiments |
| `gpu_surgery_1600k.py` | Block swap/merge/split surgery. | **Medium** — early structural search |
| `gpu_multires_1600k.py` | Multi-resolution: optimize at 200k, upsample to 1.6M. | **Low** — upsampling loses quality |
| `gpu_concentrate_1600k.py` | Concentrate mass into fewer blocks. | **Low** — concentration doesn't help |
| `gpu_widen_1600k.py` | Widen blocks (add mass at boundaries). | **Low** — widening doesn't help |

**Key lesson:** At 1.6M, the solution is deeply optimized. Block-coordinate methods are too slow (15 min/round). Ultra-high beta still finds tiny gains. The most efficient approach is full-vector Dinkelbach with beta cascade.

### `05_parameterization/` — Alternative Representations (Not Impactful)

Tried different ways to represent f for optimization. None improved over the standard w² parameterization.

| Script | What it does | Impact |
|--------|-------------|--------|
| `gpu_100k_reparam.py` | Systematic test of 4 parameterizations (exp, softplus, direct+relu, w²) × 4 Dinkelbach decompositions (standard, l1_ratio, linf_ratio, log) × 4 betas. 64 combinations total. | **None** — all 64 combinations returned identical C at beta=1e6. At beta=1e7, still no difference. |
| `gpu_100k_floor_explore.py` | Add small positive "floor" values to zero regions, then optimize with exp(v). Tests uniform floors, constant interpolation, tooth insertion, random patterns. | **Negative** — floor injection always degraded C. The zero regions are optimally zero. |

**Key lesson:** The re-parameterization experiment was definitive: at our current optimum, the choice of parameterization doesn't matter. The landscape is locally identical under all smooth reparameterizations. The zero regions are genuinely optimal — adding mass there always hurts.

### `06_packet_coordinate_ascent/` — Fine-Tuning (Not Meaningfully Impactful)

Inspired by Einstein Arena discussion threads. Treats each contiguous block as a "packet" with a scalar multiplier.

| Script | What it does | Impact |
|--------|-------------|--------|
| `gpu_100k_packet_ascent.py` | Per-block golden-section line search over scalar multipliers. Also tries block extension/contraction, tooth insertion in gaps, mass transfer between adjacent blocks, and alternating with Dinkelbach. | **Negligible** — packet ascent: +5.5e-9; support mods: ~+7e-13; the only real gain came from the high-beta Dinkelbach phase: +1.8e-8 |
| `gpu_100k_plateau_flatten.py` | Analyzes autoconvolution plateau, computes per-run Jacobian w.r.t. dip positions, solves max-min flattening problem. Also tries Adam direct optimization. | **None** — Jacobian approach hit numerical issues. Adam went sideways. Per-run search: +2e-10. |

**Key lesson:** The Einstein Arena discussion reported ~1.6e-5 gain from packet ascent, but that was from a less-optimized starting point. After full Dinkelbach optimization, block scalars are already near-optimal, so packet ascent gives only ~5e-9 — effectively no meaningful progress. The surgery/perturbation approaches in `03_perturbation_and_surgery/` were far more useful in practice for escaping local optima.

The autoconvolution plateau analysis was informative even though it didn't improve the score: the plateau has **26,000 positions within 0.1% of the maximum**, confirming the solution is near the equi-oscillation condition.

### `07_structural_exploration/` — Large Structural Changes (Not Impactful)

Tried big structural modifications: block removal, splitting, merging, global modulation, stochastic multi-block perturbation, support swaps.

| Script | What it does | Impact |
|--------|-------------|--------|
| `gpu_100k_structural.py` | 7 experiments: global modulation (100 frequencies), stochastic multi-block (200 trials), block removal (50 trials), block splitting (30 trials), support swap (100 trials), exp noise restart (20 trials), structure transfer from 1.6M. | **None** — no experiment improved C |
| `gpu_100k_sparse_comb.py` | Construct fresh sparse combs (uniform, Gaussian, random, QR, modulated) and optimize with Dinkelbach cascade. Hundreds of configurations tested. | **None** — best sparse comb after optimization: C=0.9615, far below 0.96199 |

**Key lesson:** The solution is in a deep, well-defined basin. Large structural changes (removing blocks, changing spacing) always make things worse. Fresh constructions can't reach the same basin — the best sparse comb only reached 0.9615. This suggests the current structure is near a genuine local (possibly global) optimum.

### `08_failed_but_informative/` — Negative Results Worth Knowing

These experiments failed but taught us important things about the problem structure.

| Script | What it does | Why it failed | What we learned |
|--------|-------------|---------------|----------------|
| `gpu_fourier_param_100k.py` | Fourier parameterization: represent f as truncated Fourier series, optimize coefficients. Also tried beta annealing (low β explore → high β refine). | Fourier truncation destroys the comb. K=20,000 modes (40% of total) gives C=0.887 vs C=0.962. Beta annealing lands at C=0.92-0.95, can't recover. | **The comb structure needs ALL frequencies.** There is no low-dimensional Fourier representation. |
| `gpu_100k_fresh_combs.py` | Build fresh combs from scratch with various tooth counts (100-1000). Also tried quadratic residue (QR) combs. | Best fresh comb: C≈0.52 (1000 teeth). QR combs: C≈0.55. | **Fresh construction is far from competitive.** The optimized comb structure has subtle position/width/height relationships that can't be captured by simple constructions. |
| `gpu_comb_param.py` | Represent 1.6M solution as Gaussian-shaped teeth, optimize tooth parameters (position, width, height). | Gaussian reconstruction gave C=0.14 (!). NaN errors during optimization. | **The solution is not well-described by parametric tooth shapes.** The actual nonzero regions have complex, non-Gaussian profiles. |
| `gpu_fourier_opt.py` | Fourier-domain optimization for 1.6M. | Same truncation issue as 100k version. | Confirmed: Fourier truncation fails at all scales. |
| `gpu_100k_from_1600k.py` | Transfer 1.6M solution structure to 100k via subsampling, averaging, and hybrid methods. | All downsampling methods produced inferior solutions. Best subsampled C ≈ 0.93. | **Structure doesn't transfer across scales.** The optimal comb for 1.6M is fundamentally different from the optimal comb for 100k. |
| `gpu_downsample_200k_to_100k.py` | Downsample 200k solution to 100k. | Similar failure — downsampling loses structure. | Same lesson as above. |
| `gpu_200k_to_100k.py` | Another downsampling attempt. | Same. | — |
| `gpu_comb_optimize.py` | Comb-specific optimizer with tooth-level operations. | Tooth-level operations too coarse. | Block-level operations better than tooth-level. |
| `gpu_comb_factory.py` | Factory for generating diverse comb starting points. | Generated combs too far from optimum. | — |
| `gpu_comb_opt_1600k.py` | Comb optimization for 1.6M. | Same reconstruction failures. | — |

## What To Try Next

If you want to push the score further, here's what we think is most promising based on our experience:

1. **Even higher beta with more iterations**: Beta=5e8 and 1e9 still gave gains. Try 5e9, 1e10 with 10,000+ L-BFGS iterations and 20+ Dinkelbach outers. This is the lowest-risk approach.

2. **Alternative Dinkelbach decompositions at high beta**: We tested l1_ratio, linf_ratio, and log decompositions at beta=1e6 (no difference). But at beta=1e9, the landscape is sharper and the decompositions might diverge.

3. **Population-based search**: Maintain a pool of solutions, do crossover in block-multiplier space, polish with Dinkelbach. Our fast-screening found 1 winner in 400+ screens — a larger population search might find more.

4. **Theoretical analysis**: The equi-oscillation structure of the autoconvolution plateau suggests connections to Chebyshev approximation theory. A theoretical characterization of the optimal support structure could guide construction.

5. **Support modification + full Dinkelbach**: The key observation is that Dinkelbach optimizes values perfectly but can't change the support. Any technique that finds a better support structure and then applies Dinkelbach would be very powerful.

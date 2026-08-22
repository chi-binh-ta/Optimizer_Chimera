# Chimera F-1 — Recombined Final-System Audit

## Frozen candidate

F-1 freezes Navigation and Protection from the preceding rounds:

- Navigation: Adam-like candidate direction `d = m_hat / (sqrt(v_hat) + eps)` (`gamma = 0.5`).
- Protection statistic: P-2.6 `step_trust_local`.
- Protection memory: `rho_psi = 0.3`.
- LR-neutral gate: `z = tanh(3 psi)` and `kappa_i^2 = 1 + 0.375 (z_i - mu)`, with `mu = sum(d_i^2 z_i)/sum(d_i^2)`.

Thus, up to floating-point error, `||kappa * d||_2 = ||d||_2`; the protection layer redistributes update energy but cannot win by globally increasing the scalar LR budget.

## Baselines

- Adam and AdamW (PyTorch; `weight_decay=0` in the repo stress suite, so their training dynamics coincide).
- Cautious AdamW: momentum-gradient agreement mask normalized by mean mask, following the official C-Optim implementation.
- Lion: official Google Research update, default betas `(0.9, 0.99)`.
- Muon: official Newton-Schulz update only on an eligible hidden matrix; all non-eligible parameters use AdamW fallback.

Every optimizer receives an independent LR sweep. Muon receives a 2D sweep over Muon LR and AdamW-fallback LR.

## Repo stress-suite held-out result

Primary held-out set: 8 unseen seeds × 4 repo stress problems (regression, sparse ReLU, noisy quadratic, saddle), after LR tuning on disjoint seeds.

Because the saddle objective can become negative, aggregate comparisons use the paired normalized difference

`Delta = (F_optimizer - F_Adam) / (abs(F_initial) + eps)`

rather than a loss ratio. Lower is better.

| Optimizer | Mean normalized Delta vs Adam | Win fraction | Median CPU step time | Tensor-state bytes |
|---|---:|---:|---:|---:|
| Chimera F-1 | -0.001271 | 25/32 (78.1%) | 0.623 ms | 1164 |
| Adam | 0 | reference | 0.148 ms | 792 |
| AdamW | 0 | same dynamics at wd=0 | 0.142 ms | 792 |
| Cautious AdamW | +0.063360 | 20/32 (62.5%) | 0.174 ms | 776* |
| Lion | +0.035863 | 4/32 (12.5%) | 0.090 ms | 388 |

Bootstrap 95% CI for Chimera F-1's mean normalized Delta is approximately `[-0.002453, -0.000202]` on this held-out stress set.

`*` The custom audit implementation stores the two moment tensors while keeping the integer step as Python state; conceptually Cautious has the same two full-size moment buffers as AdamW.

### Per-problem Chimera F-1 paired normalized Delta

- Noisy quadratic: `-0.002922`, 6/8 wins.
- Regression: `-0.000832`, 6/8 wins.
- Saddle: `-0.000879`, 7/8 wins.
- Sparse ReLU: `-0.000449`, 6/8 wins.

### Cautious sparse-ReLU tail failure

Cautious has a favorable median on several problems, especially saddle, but the sparse-ReLU held-out set contains catastrophic tail cases. Three seeds are roughly 57x–131x worse than Adam when measured by positive final-loss ratio. Instrumentation shows the agreement-mask renormalization can reach active scaling factors of 16x–80x when only a small fraction of coordinates survive the hard mask. The result persists when using `eps=1e-6`, the default in the official C-Optim AdamW implementation.

## Extended-horizon stress

At 200 steps on another held-out seed set, Chimera F-1 remains finite in every run and is broadly competitive with Adam. Cautious continues to show a sparse-ReLU heavy tail; Lion remains substantially worse on these small-batch synthetic stress problems.

## Dedicated failure tests

Using the noisy-quadratic tuned LRs on an anisotropic quadratic:

- Heavy-tail noise: Chimera final objective ratio to its initial value is ~0.00459 vs Adam ~0.00491.
- 40% sign flips: Chimera ~0.254 vs Adam ~0.257 (essentially tied).
- Two 100x gradient spikes: Chimera ~0.0247 vs Adam ~0.0306.
- All tested optimizers remain finite in all dedicated failure runs.

Cautious often reaches lower final objective in these shock tests, but shows larger transient clean-loss jumps. Lion is markedly worse on heavy-tail and spike tests.

## Matrix-shaped workload and Muon

A 3-layer matrix MLP is used so that Muon has a legitimate hidden `32 x 32` matrix. Only that hidden matrix is optimized by Muon; the first layer, output layer, and biases use AdamW fallback.

Muon required refinement below the initial LR grid edge. The refined tune point is:

- Muon hidden LR: `0.002`
- AdamW fallback LR: `0.04`

Held-out matrix workload (8 seeds):

- Cautious AdamW: mean positive-loss ratio vs Adam ~0.289, but worst seed ~1.85.
- Refined Muon hybrid: mean ratio ~0.682, 6/8 wins; state ~8.99 KB.
- Chimera F-1: mean ratio ~0.894, 6/8 wins; state ~19.6 KB.
- Adam/AdamW: reference; state ~13.1 KB.
- Lion: strongly worse on this small matrix-regression workload.

CPU median step time on this workload is approximately 0.21 ms for Adam, 0.28 ms for Cautious, 0.69 ms for refined Muon, and 0.96 ms for Chimera F-1. These CPU timings are implementation measurements, not GPU throughput claims.

## F-1 verdict

**Algorithmic signal: PASS INITIAL FINAL-SYSTEM AUDIT.**

The recombined P-3.5 + Adam-navigation candidate improves the repo stress suite relative to an independently LR-tuned Adam/AdamW baseline on held-out seeds, and the improvement cannot be attributed to global LR inflation because the protection gate preserves candidate-update L2 energy.

**Systems efficiency: FAILS CURRENTLY.**

The experimental Chimera implementation uses three full-size states (`m`, `v`, `psi`) and global concatenation/reduction for the LR-neutral gate. On CPU it is roughly 4x slower than Adam in the small repo stress model and about 4.5x slower in the matrix MLP. It also uses ~1.5x Adam's full-size moment-state memory.

**Competitive landscape:** Cautious is faster and can be extremely strong, but exhibits a serious sparse-mask tail failure in this audit. Muon is stronger than Chimera on the workload designed for Muon's matrix geometry after fair 2D LR tuning. Lion wins on memory/runtime but not on loss in these small-batch synthetic tasks.

Therefore Chimera F-1 should remain an experimental candidate. The next engineering round should optimize the LR-neutral gate implementation (avoid concatenation/materialized temporaries where possible), then repeat the comparison on larger GPU workloads before any default promotion or broad optimizer-performance claim.

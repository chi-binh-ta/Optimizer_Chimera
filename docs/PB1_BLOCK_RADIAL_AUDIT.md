# Chimera P-B1 — Block-Radial Protection Audit

Only one architectural change is tested:

```text
coordinatewise kappa_i d_i  ->  block scalar s_b d_b
```

Canonical block = one parameter tensor. Navigation, tau, hard clamp, alpha and global L2-energy budget are unchanged.

For block b:

```text
W_b = sum_{i in b} d_i^2
q_b = sum_{i in b} d_i^2 z_i / W_b
q_bar = sum_b W_b q_b / sum_b W_b
s_b = sqrt(1 + alpha (q_b - q_bar))
Delta_b = s_b d_b
```

Hence `Delta_b` is exactly parallel to `d_b` and total L2 update energy is preserved.

## Main empirical result

Pure tensor-block radial Protection is systematically worse than coordinate Protection-Lite on the fresh stress audit. The strongest loss is in noisy-quadratic, and multi-block heavy-tail/spike tests also show that coordinatewise redistribution carries useful intra-block robustness information.

Fresh held-out stress audit (12 seeds x 4 problems, 200 steps) gives block-vs-coordinate mean initial-normalized delta `+8.82e-4`, bootstrap 95% CI approximately `[+4.09e-4, +1.42e-3]`. The noisy-quadratic component is the clearest failure: mean delta `+3.22e-3`, with its bootstrap interval fully positive.

The block rule does satisfy the intended P/N invariant: measured block cosine is exactly `1.0`, while coordinate Protection-Lite has mean block cosine about `0.99907` and can rotate inside each parameter tensor.

## Systems result

The systems benefit is real. Because pass 2 needs only one scalar `s_b` per tensor, tau/z do not need to be recomputed. A fused CPU prototype speeds the Protection stage by roughly 1.4x-1.74x at matrix/large sizes versus coordinate Protection-Lite.

## Verdict

- Mathematical/radial invariant: **PASS**.
- Global L2-energy neutrality: **PASS**.
- Systems simplification: **PASS**.
- Full replacement of coordinate Protection-Lite: **FAIL**.

Do not replace coordinate Protection-Lite with pure tensor-block radial Protection. The audit indicates that a small amount of intra-block coordinate redistribution carries real value, especially under coordinatewise noise.
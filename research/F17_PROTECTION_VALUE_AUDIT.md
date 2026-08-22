# Chimera F-1.7 — Protection Value / Complexity Audit

This round freezes the Adam-like Navigation and decomposes the Protection cost into three questions: persistent `psi`, the nonlinear `tanh(lambda * source)` map, and global d^2-weighted centering.

## Reference Protection

The F-1/P-3.5 reference uses

- `tau`: P-2.6 local step-trust statistic,
- `psi_t = 0.3 psi_{t-1} + 0.7 tau_t`,
- `z_i = tanh(3 psi_i)`,
- `mu = sum(d_i^2 z_i) / sum(d_i^2)`,
- `kappa_i = sqrt(1 + 0.375 (z_i - mu))`.

The gate is L2-energy neutral because `sum(d_i^2 kappa_i^2) = sum(d_i^2)` up to floating-point error.

## Ablations

The audit compares independently LR-tuned variants:

1. no Protection (`kappa=1`),
2. `psi + tanh + global center` (reference),
3. `psi + hardtanh + global center`, with `hardtanh(x)=clip(x,-1,1)`,
4. `tau + tanh + global center` (no persistent psi),
5. `tau + hardtanh + global center` (no persistent psi, no tanh),
6. per-tensor centering controls.

The hard map uses `z=clip(3 source,-1,1)`, matching the reference tanh map's slope at zero and range.

## Confirmation protocol

- tune seeds: 100–103,
- fresh confirmation seeds: 500–511,
- four repository stress problems,
- 200 confirmation steps,
- LR tuned independently for each variant/problem,
- comparison uses initial-objective-normalized paired differences because the saddle objective may be negative.

## Main empirical result

`tau + hardtanh + global center` is the strongest simplified candidate in this audit.

Against the full `psi+tanh+global` reference, its mean initial-normalized final-objective difference is approximately `-2.92e-4`; a paired bootstrap 95% interval is approximately `[-5.47e-4, -9.3e-5]`. It wins 27/48 confirmation comparisons. Its worst observed degradation relative to the full reference is approximately `1.08e-3` in initial-normalized objective units.

Against no Protection, the same candidate improves mean normalized objective by approximately `-9.66e-4`, with bootstrap 95% interval approximately `[-1.83e-3, -2.03e-4]`, and wins about 73% of paired runs.

Removing only `psi` while retaining tanh (`tau+tanh`) does not show a quality loss relative to the full reference. Replacing tanh with hardtanh while retaining the EMA state (`psi+hardtanh`) creates a sparse-ReLU tail failure in the 200-step confirmation audit, so the hard map is not a safe drop-in replacement unless the stale-memory interaction is removed.

Per-tensor centering is weaker than global centering for the simplified `tau+hardtanh` candidate; the tensor-centered variant has worse aggregate objective and a sparse tail, so global d^2-weighted centering remains justified.

## Systems result

In the fused two-pass CPU Protection prototype, `tanh` is a major cost after intermediate tensors are removed. For approximately 1,048,576 coordinates, the persistent-psi Protection stage measured about 5.7–6.0 ms with tanh versus about 1.1–1.2 ms with hardtanh. At approximately 262k coordinates, the corresponding values are about 1.5 ms versus 0.27 ms.

Dropping persistent psi requires recomputing tau in the second global-centering pass. The no-memory `tau+hardtanh` kernel is therefore about 30–40% slower than `psi+hardtanh` at large sizes, but remains roughly 3–4x faster than the original `psi+tanh` Protection kernel while removing one persistent FP32 state vector.

State memory changes from `(m,v,psi)` = 12 bytes/parameter in FP32 to `(m,v)` = 8 bytes/parameter, i.e. Adam-like persistent-state size.

At 262k parameters, a prototype whole optimizer with `tau+hardtanh+global` measured about 1.30 ms versus about 0.73 ms for Adam and 1.48 ms for Cautious AdamW on the same CPU setup. At 1M parameters the remaining overhead is dominated by the materialized Adam-like Navigation dataflow (`m_hat`, `v_hat`, sqrt, d), not by Protection alone; a Navigation-only materialized benchmark is already roughly 2.6x slower than optimized PyTorch Adam at that size.

## Verdict

- Persistent `psi`: **not justified by current quality evidence**; removing it does not hurt the confirmation benchmark and saves one full state vector.
- `tanh(3 source)`: **not justified by current quality/cost evidence**; it is a dominant fused-kernel cost, and the no-memory hardtanh candidate performs at least as well in this audit.
- Global d^2-weighted centering: **retain**; per-tensor centering is weaker and does not provide an empirical reason to drop the global LR-neutral redistribution constraint.

The simplified candidate is therefore

`tau -> z=clip(3 tau,-1,1) -> global weighted center -> kappa`,

with no persistent `psi`.

This is an experimental simplification only. Do not replace the production/default optimizer until the F-1 external-baseline audit is rerun with this exact candidate.
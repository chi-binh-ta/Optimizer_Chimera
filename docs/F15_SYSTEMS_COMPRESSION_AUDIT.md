# Chimera F-1.5 — Systems Compression Audit

Scope: freeze F-1 mathematics exactly. Navigation remains Adam-like
`m_hat/(sqrt(v_hat)+eps)` and Protection remains P-2.6 step-trust plus the
P-3.5 global L2-energy-neutral gate. Only implementation/data representation
is audited.

## Hard equivalence criterion

The FP32 systems candidate must reproduce the slow F-1 reference parameter
trajectory exactly on the four repository stress problems. In the audit,
200-step runs on regression, sparse-ReLU, noisy-quadratic, and saddle all had
`max_abs_parameter_difference = 0`.

## Exact implementation changes

- diagnostics are off on the training critical path;
- Adam first/second-moment tensor updates use `torch._foreach_*`;
- coherence reuses the identity `abs(m_hat)/(sqrt(v_hat)+eps) == abs(d)`;
- temporary tensors are reused in-place where possible;
- the P-3.5 gate keeps the same flattened parameter-group reduction order as
  F-1, avoiding trajectory changes from a streaming reduction.

A streaming reduction was tested and rejected as the default despite better
large-tensor runtime because changing reduction order produced measurable
trajectory drift on sparse-ReLU.

## CPU stress-suite timing

Eight held-out seeds x four stress problems, optimizer-step timing only,
after 10 warmup steps:

| implementation | median step ms | x Adam | x Cautious |
|---|---:|---:|---:|
| Adam | 0.158 | 1.00 | 0.83 |
| Cautious | 0.189 | 1.20 | 1.00 |
| F-1 original with diagnostics | 0.721 | 4.57 | 3.81 |
| F-1.5 exact | 0.439 | 2.78 | 2.32 |

Thus F-1.5 reduces the observed F-1 overhead by about 39%, while preserving
the audited trajectory exactly. The target of being close to Adam/Cautious is
**not yet met**.

## Psi storage audit

Persistent-state bytes on the tiny stress model:

- Adam: 792 B
- Chimera FP32 psi: 1164 B
- Chimera FP16 psi: 970 B
- Chimera int16 fixed-point psi: 970 B
- Chimera int8 psi: 873 B

The storage tolerance audit used four seeds x four stress problems for 120
steps. Acceptance thresholds were max relative parameter-L2 drift <= 1e-4 and
max final-objective relative error <= 5e-3.

- FP16: parameter tolerance passes, but sparse final-objective error reached
  about 1.28%; rejected under the stated objective tolerance.
- int16 fixed point: passes both stated tolerances; max observed sparse final
  objective error was about 0.31%.
- int8: fails clearly; sparse final-objective error reached about 40.8%.

On CPU, int16/fp16 conversion overhead makes the compressed modes slower than
FP32, so int16 is retained only as a memory-oriented experimental mode, not as
F-1.5 default.

## Systems verdict

**Trajectory preservation: PASS.**

**State-memory compression: PARTIAL PASS.** int16 psi gives a viable
memory-oriented approximation, but is not exact and is slower on CPU.

**Runtime target: FAIL/PARTIAL.** The exact implementation improves the F-1
runtime substantially (about 4.57x Adam -> 2.78x Adam in this CPU audit), but
is still about 2.3x Cautious. A genuinely competitive next systems step likely
requires a fused custom kernel (or a GPU-oriented fused implementation) for
step-trust plus the global energy-neutral gate, rather than additional Python
rearrangement.

No CUDA/MPS device was available in the audit environment, so these timings
must not be interpreted as GPU throughput results.

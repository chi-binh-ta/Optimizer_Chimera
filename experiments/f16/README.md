# Chimera F-1.6 — Fused Protection Kernel Audit

Scope: systems-only audit. The mathematical optimizer is frozen at Navigation gamma=0.5 and P-3.5 LR-neutral Protection.

Protection equations remain:

- `d = m_hat / (sqrt(v_hat) + eps)`
- `coherence = clamp(abs(d), 0, 1)`
- `exposure = lr*abs(d) / (abs(param) + 2*lr*abs(d) + eps)`
- `tau = sign(g*d) * coherence^2 * (1-exposure) - exposure`
- `psi = rho*psi + (1-rho)*tau`
- `z = tanh(lambda*psi)`
- `mu = sum(d^2*z) / sum(d^2)`
- `kappa = sqrt(1 + alpha*(z-mu))`
- `param -= lr*kappa*d`

The experimental CPU kernel does not materialize `coherence`, `exposure`, `tau`, `z`, `d^2`, or `kappa` as persistent/full tensors. After Navigation has materialized `d`, Protection is executed as two coordinate passes:

1. update `psi` and accumulate only scalar sufficient statistics `S0=sum(d^2)` and `S1=sum(d^2*tanh(lambda*psi))`;
2. recompute `z`, form `kappa` transiently, and apply the protected update directly.

## Runtime finding

The fused implementation removes a large fraction of the execution overhead. On the CPU-only audit environment it is approximately Adam-speed for tiny/matrix workloads. For larger 4x256x256 tensors, an aggressive compiler-vectorized version measured about 2.8x Adam and 1.24x Cautious; for 4x512x512 it measured about 4.9x Adam and 2.0x Cautious. The safer explicit SLEEF/vector prototype is slower on large tensors.

These are CPU measurements only; no CUDA/Triton claim is made.

## Numerical trajectory finding

The hard comparison target is `ChimeraF15Exact`. Fusion changes floating-point reduction/evaluation order even though the real-valued equations are unchanged. Dense/noisy/saddle trajectories stay very close, but sparse-ReLU is numerically sensitive. Across six held-out sparse seeds, the explicit vector kernel reached worst max parameter relative-L2 drift about 1.7e-3 and worst final clean-objective relative error about 9.8e-3. The faster `-ffast-math` upper-bound kernel drifted more on some sparse seeds.

Therefore F-1.6 does **not** pass the strict trajectory-preservation gate as a production replacement. The fused kernel remains experimental and is intentionally not wired into the default optimizer.

## Main architectural conclusion

The derived quantities `c, r, tau, z, d^2, mu, kappa` do not all need to exist as runtime tensors. They can remain part of the mathematical architecture while being compiler/kernel temporaries. The persistent optimizer state is still `m, v, psi`. However, materialization boundaries also define floating-point rounding boundaries; removing them can change the discrete FP32 optimization trajectory in sensitive sparse regimes.

# Chimera 2.1 Overview

Chimera 2.1 is organized around three separable pieces:

- `chimera.optimizer.Chimera21`: Adam-like optimizer with agreement-gated update dynamics.
- `chimera.quantization`: stat, EMA, strict BitNet, Chimera ternary thresholding, and ternary stats.
- `chimera.bitlinear.BitLinear`: BitLinear-style execution using quantized activations and ternary weights.
- `chimera.bitconv.BitConv2d`: BitConv2d-style execution for ResNet/CIFAR experiments.

`Chimera21` owns the optimizer effect. It updates full-precision parameters using momentum, variance, an agreement EMA, and a bounded multiplicative gate.

The quantization utilities own the quantizer effect. They can produce a strict BitNet-style ternary baseline or a Chimera dynamic-threshold ternary tensor.

`BitLinear` owns the execution layer. It keeps a full-precision master weight, tracks a scale EMA, quantizes activations to a signed 8-bit range, quantizes weights to ternary values, and runs the linear projection.

`BitConv2d` mirrors `BitLinear` for convolution: it keeps a full-precision master convolution weight, quantizes activations, applies strict BitNet or Chimera ternary weight quantization, and reports ternary stats.

## v2 and v3 scope

v2 is the BitLinear benchmark harness for local `strict_bitnet` versus `chimera` comparisons.

v3 is the ResNet-oriented effective-bit path. It adds `BitConv2d`, a CIFAR-style `ChimeraResNet20` skeleton, and a small CIFAR training script that fails gracefully when CIFAR-10 or optional torchvision support is unavailable.

v4 adds target-bits control. v5 changes the control signal to the high-sparsity zero-ratio branch associated with the target entropy. v6 adds batch-level control and direct `alpha_override` for faster sparse-branch response. v7 adds train/test evaluation, explicit CIFAR download behavior, optional checkpoint metadata, and a synthetic-safe ablation runner. v8 makes eval forwards non-mutating for quantizer warmup/EMA state and uses eval-measured stats in eval logs. The controller lives in `chimera.target_bits`, outside the optimizer, and adjusts quantizer alpha from measured global zero ratio. This keeps optimizer effect, quantizer effect, and execution layer effect separate.

## Alpha policy

The default alpha policy is sparsity warmup from zero: `alpha_min = 0.0` and `alpha_target = 0.7`.

For Chimera thresholding, alpha controls how aggressively weights are zeroed. Starting at zero keeps the early forward pass close to sign-only ternary weights and avoids zeroing too early. The threshold then warms up linearly toward the target value. The convention is: the first warmup forward uses `alpha_min`, and the final warmup forward uses `alpha_target`.

Alpha is intentionally not part of `Chimera21`. The optimizer step does not consume alpha, so the cleaner design is to keep alpha scheduling inside `BitLinear` and `BitConv2d`, where ternary thresholding is applied. This preserves the separation between optimizer effect and quantizer effect.

## Local benchmark scripts

- `experiments/run_smoke.py`: default small Chimera training smoke test.
- `experiments/compare_quant_modes.py`: small CPU-only comparison between `strict_bitnet` and `chimera`, with CLI options for steps, seed count, shape, learning rate, JSONL output, and device.
- `experiments/train_resnet_cifar.py`: CPU-safe CIFAR/ResNet benchmark harness with synthetic mode, train/test evaluation, optional checkpointing, and sparse-branch control for Chimera mode.
- `experiments/run_cifar_ablation.py`: runs the v7 recipe set, `fp32_baseline + sgd`, `strict_bitnet + chimera21`, and `chimera + chimera21 + sparse control`, into one JSONL log.

`compare_quant_modes.py` keeps seed, shape, random input, target, training steps, and optimizer settings aligned so the output isolates quantizer behavior. It is a development benchmark, not a paper-scale benchmark.

JSONL records include run identity, script, mode, seeds, step metrics, learning rate, key optimizer/quantizer settings, and UTC timestamp. The final console summary reports mean and standard deviation for final loss and zero ratio by quantization mode.

In v8, eval records use stats returned by the evaluation loop. `BitLinear` and `BitConv2d` update `scale_ema` and `forward_step` only in training mode, so validation does not advance quantizer warmup or EMA state.

## Optimizer robustness branch

`Chimera21` keeps its default behavior unchanged, but exposes optional research controls:

- `log_diagnostics`: records kappa spread, noise ratio, effective learning-rate scale, zero-gradient ratio, psi saturation, and a collision score.
- `zero_grad_policy`: chooses standard, freeze, decay, or inertia behavior for `psi` where gradients are zero.
- `kappa_gate_mode`: can cross-gate kappa by Adam-style noise ratio to reduce double-adaptation collision.
- `psi_storage`: can store `psi` as fp32 or prototype int8 state.

`experiments/compare_optimizers.py` is a small deterministic synthetic ablation harness for these policies. Its output belongs in ignored `outputs/` logs.

v10 adds stress problems for optimizer robustness:

- `regression`: the default deterministic baseline with some structurally zero input columns.
- `sparse_relu`: sparse activations and dead-gradient behavior.
- `noisy_quadratic`: high noise-ratio gradients to test cross-gated kappa.
- `saddle`: mixed-curvature nonconvex dynamics.

The optimizer diagnostics now include active-only effective learning rate, update norm statistics, and optimizer state-memory accounting, including fp32 versus int8 `psi` bytes.

v11 completes the local robustness ablation suite:

- `chimera21_zero_freeze`, `chimera21_zero_decay`, and `chimera21_zero_inertia` compare sparse/zero-gradient psi handling.
- `saddle` is part of the official stress path for nonconvex mixed-curvature behavior.
- `step_time_ms` and `mean_step_time_ms` make the int8-psi memory/runtime tradeoff visible.
- `experiments/run_optimizer_stress_suite.py` runs regression, sparse ReLU, noisy quadratic, and saddle into one JSONL file.

v12 and the v0.13 release-candidate cleanup add release-readiness polish:

- `mean_step_time_ms_after_warmup` and `median_step_time_ms_after_warmup` reduce first-step timing noise in optimizer summaries.
- `train_resnet_cifar.py` accepts `--device cuda` and exits clearly if CUDA is unavailable.
- `docs/release_checklist.md` and `docs/colab_quickstart.md` document clone/install/smoke/CIFAR commands.
- Generated `outputs/`, `checkpoints/`, caches, and local benchmark files are release artifacts, not source files.
- Public benchmark wording stays conservative: CIFAR recipes are smoke/benchmark paths, optimizer stress tests are synthetic diagnostics, and effective bits are entropy metrics rather than hardware compression claims.

## Effective bits

`effective_bits` is Shannon entropy over the observed ternary weight distribution:

```text
H = -sum_i p_i log2(p_i), i in {-1, 0, +1}
```

Zero-probability terms are ignored. A target such as 1.32 bits refers to entropy/effective bitwidth of the ternary distribution. It is not a guarantee of physical storage compression or hardware packing efficiency.

## Target-bits control

The v5 controller first maps target entropy to a symmetric ternary zero-ratio target. For `target_bits = 1.32`, the sparse branch gives a target zero ratio near `0.63`, while the dense branch gives a low-sparsity solution near `0.08`.

The controller then uses:

```text
error = current_zero_ratio - target_zero_ratio
```

If current zero ratio is below target, alpha increases to encourage more zero weights. If current zero ratio is above target, alpha decreases. Alpha is clipped into configured bounds. This is a simple deterministic controller intended for benchmark harness development, not a final adaptive quantization method.

The training script supports `--controller-frequency epoch|batch` and `--controller-affects alpha_target|alpha_override`. The `alpha_target` path preserves warmup behavior. The `alpha_override` path writes the controlled alpha directly into quantized layers, bypassing warmup for controlled runs.

Thread-safe ResNet quick check:

```powershell
python -c "import torch; torch.set_num_threads(1); from chimera.models.resnet_cifar import ChimeraResNet20; print(ChimeraResNet20(mode='chimera')(torch.randn(1,3,32,32)).shape)"
```

## Roadmap

- Strict BitNet baseline.
- Chimera threshold ablation.
- Agreement gate ablation.
- Transformer integration.
- Hardware-aware experiments.

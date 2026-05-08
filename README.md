# Optimizer Chimera

Chimera 2.1 is a lightweight research framework combining an agreement-gated optimizer and dynamic ternary quantization in a BitNet-compatible direction.

## Project overview

This repository separates three effects that should be benchmarked independently:

- Optimizer effect: agreement-gated Adam-like update dynamics.
- Quantizer effect: dynamic ternary thresholding.
- Execution layer: BitLinear-style forward pass with quantized activations and ternary weights.

This is a research benchmark harness. The MVP is CPU-only by default, compact, and intended to stay easy to test, benchmark, and extend into a paper codebase without making performance claims ahead of validation.

v2 is the BitLinear benchmark harness for local quantizer comparisons. v3 adds a ResNet-oriented path with BitConv2d, a CIFAR ResNet-20 skeleton, and entropy-based effective-bit metrics. v4 adds a small target-bits controller. v5 controls the high-sparsity branch by targeting zero ratio directly while keeping effective bits as a reporting metric. v6 adds batch-level control, direct alpha override, and a synthetic ResNet smoke path. v7 adds train/test evaluation, checkpoint metadata, and a CIFAR ablation runner for benchmark readiness. v8 makes evaluation non-mutating for quantizer warmup/EMA state and documents the first real CIFAR-10 recipe. v12 adds release cleanup docs, Colab quickstart notes, optional CUDA execution for ResNet/CIFAR, and timing warmup summaries. v0.13 release candidate focuses on GitHub/Colab cleanup, artifact hygiene, and public wording.

## Folder structure

```text
configs/          Default experiment configuration.
docs/             Short design notes and roadmap.
experiments/      Runnable smoke tests and small experiments.
notebooks/        Placeholder for exploratory notebooks.
src/chimera/      Core optimizer, quantization, BitLinear, and utilities.
tests/            Pytest coverage for the MVP behavior.
```

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python experiments/run_smoke.py
python experiments/compare_quant_modes.py
python experiments/compare_quant_modes.py --steps 8 --seeds 3 --log-jsonl outputs/compare_quant_modes.jsonl
python experiments/compare_optimizers.py --steps 20 --log-jsonl outputs/optimizer_ablation_v9.jsonl
python experiments/train_resnet_cifar.py --synthetic --epochs 2 --max-batches 1 --eval-max-batches 1 --controller-affects alpha_override --log-jsonl outputs/resnet_synthetic_v8.jsonl
python experiments/run_cifar_ablation.py --synthetic --epochs 1 --max-batches 1 --eval-max-batches 1 --log-jsonl outputs/cifar_ablation_synthetic_v8.jsonl
python experiments/run_optimizer_stress_suite.py --steps 5 --log-jsonl outputs/optimizer_stress_suite_v12.jsonl
python -c "import torch; torch.set_num_threads(1); from chimera.models.resnet_cifar import ChimeraResNet20; print(ChimeraResNet20(mode='chimera')(torch.randn(1,3,32,32)).shape)"
```

## Implemented in v1

- `Chimera21`: Adam-like optimizer with agreement EMA and gated update scale.
- Dynamic ternary quantization utilities.
- Strict BitNet-style ternary weight quantization baseline.
- `BitLinear`: CPU-friendly linear layer using int8-like activations and ternary weights.
- Basic pytest coverage and a runnable smoke training script.
- Stable local compare script for `strict_bitnet` vs `chimera` quantization modes.
- JSONL benchmark logs with stable metric fields.
- Effective bitwidth metrics from ternary entropy.
- `BitConv2d` and a CIFAR-style `ChimeraResNet20` skeleton for ResNet-oriented v3 experiments.
- Sparse-branch target-zero-ratio controller for v5 ResNet-oriented Chimera runs.
- Batch/epoch controller frequency and direct `alpha_override` for v6 sparse-control runs.
- v7 CIFAR readiness: train/test evaluation logging, explicit CIFAR download behavior, checkpoint metadata, and a synthetic-safe ablation runner.
- v8 evaluation correctness: eval forwards do not advance quantizer warmup or update scale EMA.
- Robustness research options for `Chimera21`: diagnostics, zero-gradient policies, noise-ratio cross-gating, and prototype int8 psi storage. These are off by default unless enabled in config or constructor arguments.
- v10 stress harness: optimizer state-memory accounting, active-only effective learning-rate diagnostics, and regression/sparse/noisy/nonconvex synthetic stress problems.
- v11 complete optimizer robustness ablation: freeze/decay/inertia zero-gradient policies, saddle stress in the official path, runtime timing diagnostics, and a multi-problem stress-suite runner.
- v12 release readiness: version metadata cleanup, Colab/GPU smoke docs, optional CUDA device support in the CIFAR trainer, timing warmup summaries, and artifact cleanup guidance.
- v0.13 release candidate: cleanup for GitHub/Colab readiness.

## Alpha policy

The default Chimera quantizer policy is sparsity warmup from zero:

- `alpha_min = 0.0`
- `alpha_target = 0.7`

This starts with almost no threshold-induced zeroing, then increases the ternary threshold over warmup. The first warmup forward uses `alpha_min`; the final warmup forward uses `alpha_target`.

Alpha belongs to the quantization path in `BitLinear`. `Chimera21` does not use alpha in `optimizer.step()`, so v2 keeps the optimizer focused on agreement-gated Adam-like dynamics and keeps threshold scheduling in the execution/quantization layer.

## Local benchmarks

- `python experiments/run_smoke.py`: small training smoke test for the default Chimera path.
- `python experiments/compare_quant_modes.py`: small CPU-only comparison of `strict_bitnet` and `chimera` under the same seed, shape, data, target, and optimizer settings.
- `python experiments/compare_quant_modes.py --steps 8 --seeds 3 --log-jsonl outputs/compare_quant_modes.jsonl`: writes parseable JSONL records for later analysis.

The compare script is for separating quantizer effects during local development. It is not a paper-scale benchmark.

In v3, `effective_bits` is Shannon entropy over ternary weight values {-1, 0, +1}. A 1.32-bit result means entropy/effective bitwidth of the observed ternary distribution, not guaranteed hardware storage compression.

In v5, `target_bits = 1.32` is converted to a symmetric ternary zero-ratio target on the sparse branch, about `0.63`. The controller adjusts `alpha_target` on quantized ResNet layers after each epoch: if global zero ratio is below target, alpha increases; if it is above target, alpha decreases.

In v6, controlled ResNet runs can update after each batch and can use `alpha_override` so the controlled alpha is used directly instead of waiting for warmup. Use `--synthetic` for a CPU-only smoke path without CIFAR or torchvision.

In v7, `train_resnet_cifar.py` logs train and test metrics per epoch, saves checkpoints only when `--save-checkpoint` is passed, and refuses silent CIFAR downloads unless `--download` is explicit. `run_cifar_ablation.py` runs `fp32_baseline`, `strict_bitnet`, and `chimera` recipes into one JSONL file.

In v8, `model.eval()` forwards in `BitLinear` and `BitConv2d` do not increment `forward_step` or update `scale_ema`. Eval JSONL records use stats measured during the eval pass instead of copying train stats. The `outputs/` directory is ignored by git and is intended for local benchmark logs/checkpoints.

The optimizer robustness branch adds `experiments/compare_optimizers.py` for local synthetic checks of `adam`, baseline `chimera21`, cross-gated Chimera, zero-freeze Chimera, and int8-psi Chimera. It is diagnostic plumbing for ablations, not a robustness claim.

Optimizer stress tests are synthetic diagnostics. They are useful for finding failure modes and measuring state/runtime tradeoffs, but they are not real-world superiority claims.

v10 extends the optimizer harness with:

```powershell
python experiments/compare_optimizers.py --steps 20 --problem regression --log-jsonl outputs/optimizer_regression_v10.jsonl
python experiments/compare_optimizers.py --steps 20 --problem sparse_relu --sparsity 0.7 --log-jsonl outputs/optimizer_sparse_relu_v10.jsonl
python experiments/compare_optimizers.py --steps 20 --problem noisy_quadratic --noise-scale 2.0 --log-jsonl outputs/optimizer_noisy_quadratic_v10.jsonl
python experiments/compare_optimizers.py --steps 20 --problem saddle --log-jsonl outputs/optimizer_saddle_v11.jsonl
python experiments/run_optimizer_stress_suite.py --steps 10 --log-jsonl outputs/optimizer_stress_suite_v11.jsonl
```

The JSONL output includes problem name, step timing, optimizer state bytes, psi bytes, active effective learning-rate diagnostics, update norms, and collision scores. `saddle` targets mixed-curvature nonconvex behavior where agreement-gating can interact with changing gradient directions.

Release and Colab notes live in:

- `docs/release_checklist.md`
- `docs/colab_quickstart.md`

Generated `outputs/`, `checkpoints/`, caches, and local logs should not be committed. Use `python scripts/clean_artifacts.py` before release cleanup.

## CIFAR-10 recipes

Synthetic smoke, no CIFAR or torchvision required:

```powershell
python experiments/train_resnet_cifar.py --synthetic --epochs 2 --max-batches 1 --eval-max-batches 1 --controller-affects alpha_override --log-jsonl outputs/resnet_synthetic_v8.jsonl
```

Real CIFAR-10 single-mode run. Download is explicit:

```powershell
python experiments/train_resnet_cifar.py --dataset-root data --download --epochs 1 --batch-size 64 --eval-batch-size 128 --max-batches 100 --eval-max-batches 20 --mode chimera --optimizer chimera21 --controller-affects alpha_override --log-jsonl outputs/resnet_cifar_chimera_v8.jsonl
```

Three-mode ablation recipe:

```powershell
python experiments/run_cifar_ablation.py --dataset-root data --download --epochs 1 --batch-size 64 --eval-batch-size 128 --max-batches 100 --eval-max-batches 20 --log-jsonl outputs/cifar_ablation_v8.jsonl
```

CPU remains the supported smoke path in this repo. Full CIFAR sweeps will be slow on CPU; use a GPU-capable follow-up environment for serious runs after validating device support.

Small GPU smoke, explicit download:

```powershell
python experiments/train_resnet_cifar.py --dataset-root data --download --epochs 1 --batch-size 64 --eval-batch-size 128 --max-batches 10 --eval-max-batches 5 --device cuda --mode chimera --optimizer chimera21 --controller-affects alpha_override --log-jsonl outputs/cifar_gpu_smoke_v12.jsonl
```

These CIFAR commands are smoke and benchmark recipes, not validated paper-scale results. The GPU smoke verifies the CUDA code path only and is not a benchmark superiority claim.

## Next steps

- Run threshold and agreement-gate ablations.
- Add benchmark scripts before expanding model scope.
- Integrate transformer modules in a later iteration.

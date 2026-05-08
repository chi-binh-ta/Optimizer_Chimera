# Colab Quickstart

This is an optional GPU smoke path for public GitHub testing. It is not a benchmark claim.

## Clone And Install

```python
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/Optimizer_Chimera.git
%cd Optimizer_Chimera
!pip install -r requirements.txt
!pip install -e .
```

## GPU Check

```python
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

## CPU Smoke Tests

```python
!PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
!OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/run_smoke.py
!OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/run_optimizer_stress_suite.py --steps 5 --log-jsonl outputs/optimizer_stress_suite_colab.jsonl
```

## Synthetic ResNet Smoke

```python
!OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/train_resnet_cifar.py \
  --synthetic \
  --epochs 1 \
  --max-batches 1 \
  --eval-max-batches 1 \
  --controller-affects alpha_override \
  --log-jsonl outputs/resnet_synthetic_colab.jsonl
```

## CIFAR-10 GPU Smoke

Use explicit `--download`. This command verifies the CUDA path and CIFAR loader path; it is intentionally small. The CIFAR-10 path requires optional `torchvision`, which is not needed for local CPU smoke tests.

```python
!pip install torchvision
```

```python
!python experiments/train_resnet_cifar.py \
  --dataset-root data \
  --download \
  --epochs 1 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --max-batches 10 \
  --eval-max-batches 5 \
  --device cuda \
  --mode chimera \
  --optimizer chimera21 \
  --controller-affects alpha_override \
  --log-jsonl outputs/cifar_gpu_smoke_colab.jsonl
```

If CUDA is unavailable, the script exits clearly. Switch the runtime to a GPU runtime or use `--device cpu` for a CPU smoke run.

## Notes

- Generated files under `outputs/` are local artifacts and should not be committed.
- CIFAR-10 smoke logs are for code-path validation only.
- Full empirical claims require a separate multi-seed benchmark plan.

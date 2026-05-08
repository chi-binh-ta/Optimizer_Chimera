# Release Checklist

Use this checklist before publishing Optimizer Chimera.

## Local Setup

```powershell
git clone https://github.com/<YOUR_GITHUB_USERNAME>/Optimizer_Chimera.git
cd Optimizer_Chimera
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Required Checks

```powershell
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/run_smoke.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/train_resnet_cifar.py --synthetic --epochs 1 --max-batches 1 --eval-max-batches 1 --controller-affects alpha_override --log-jsonl outputs/resnet_synthetic_release.jsonl
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/run_optimizer_stress_suite.py --steps 5 --log-jsonl outputs/optimizer_stress_suite_release.jsonl
```

## Optional CIFAR-10 Checks

CPU smoke, explicit download:

```powershell
python experiments/train_resnet_cifar.py --dataset-root data --download --epochs 1 --batch-size 32 --eval-batch-size 64 --max-batches 10 --eval-max-batches 5 --mode chimera --optimizer chimera21 --controller-affects alpha_override --log-jsonl outputs/cifar_cpu_smoke.jsonl
```

GPU smoke, explicit download:

```powershell
python experiments/train_resnet_cifar.py --dataset-root data --download --epochs 1 --batch-size 64 --eval-batch-size 128 --max-batches 10 --eval-max-batches 5 --device cuda --mode chimera --optimizer chimera21 --controller-affects alpha_override --log-jsonl outputs/cifar_gpu_smoke.jsonl
```

Do not present these smoke results as benchmark superiority. They only verify that the code path runs.

## Release Cleanup

Generated artifacts should not be committed. `outputs/`, `checkpoints/`, caches, and bytecode are ignored.

```powershell
python scripts/clean_artifacts.py
git status --short
```

Before release, confirm:

- Version in `pyproject.toml` is correct.
- README quickstart runs from a clean clone.
- `docs/colab_quickstart.md` is up to date.
- No generated logs, checkpoints, caches, or local datasets are staged.
- No claim is made that Chimera beats baselines on real CIFAR-10.
- Public-facing repo references use `Optimizer_Chimera` or `optimizer-chimera`.

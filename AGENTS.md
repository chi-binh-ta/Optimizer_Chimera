# AGENTS.md

## Repo layout

- `src/chimera/`: core Python package.
- `configs/`: YAML defaults for optimizer and quantization settings.
- `tests/`: pytest checks for imports, quantization, BitLinear, and optimizer behavior.
- `experiments/`: runnable local smoke scripts.
- `docs/`: short design notes and roadmap.

## Install and test

```powershell
pip install -r requirements.txt
pytest -q
python experiments/run_smoke.py
```

## Coding conventions

- Keep the MVP CPU-only and Python 3.10+ compatible.
- Use short, typed, importable modules with real code only.
- Prefer minimal fixes before adding new features.
- Do not add dependencies beyond `torch`, `pytest`, `pyyaml`, and `numpy` unless clearly needed.
- Run `pytest -q` or `python experiments/run_smoke.py` after important changes.
- Keep optimizer effects separate from quantizer effects.
- Keep BitLinear focused on execution with quantized activations and ternary weights.

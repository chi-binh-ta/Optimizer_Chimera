# Optimizer Chimera Paper Package

This directory contains an arXiv-style research note for Optimizer Chimera.

The note derives the ternary effective-bit objective, explains sparse-branch zero-ratio targeting, describes Chimera21 directional agreement dynamics, and summarizes the first CIFAR-10 smoke benchmark. The results are smoke-scale reproducibility documentation, not paper-scale performance claims.

## Files

- `main.tex`: LaTeX source.
- `refs.bib`: BibTeX references.
- `tables/chimera_cifar_smoke_results.csv`: per-seed smoke results.
- `tables/chimera_cifar_smoke_summary.csv`: aggregate smoke summary.
- `Makefile`: convenience build and cleanup targets.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or, where `make` is available:

```bash
make
```

The paper uses only standard LaTeX packages: `amsmath`, `amssymb`, `amsthm`, `booktabs`, `graphicx`, `hyperref`, `geometry`, and `xcolor`.

## Scope

The paper intentionally avoids claims that Chimera beats Adam, BitNet, or full-precision ResNets. A 1.32-bit target means entropy/effective bitwidth of the ternary weight distribution, not hardware storage compression.

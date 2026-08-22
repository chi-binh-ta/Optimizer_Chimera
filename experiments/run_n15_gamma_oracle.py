"""N-1.5: audit whether Adam's gamma=1/2 preconditioning exponent should adapt.

Protection is frozen at the P-3.5 candidate. Candidate navigators are

    d_gamma = m_hat / (v_hat + eps)**gamma,  gamma in [0, 1/2].

To remove scalar-LR confounding, every candidate is L2 norm-matched to the
current Adam candidate d_{1/2} before the fixed P-3.5 protection gate is
applied. The script performs a shadow one-step oracle audit on exact SPD
quadratics and evaluates whether cheap online observables predict gamma*.

The intended falsification criterion is strict: adaptive gamma is only
justified if an observable-state policy beats fixed gamma=1/2 on held-out
seeds after norm matching. Oracle headroom alone is not sufficient.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BETA1 = 0.9
BETA2 = 0.999
RHO_PSI = 0.3
LAMBDA_GATE = 3.0
ALPHA = 0.375
EPS = 1.0e-10
GAMMAS = np.linspace(0.0, 0.5, 9)
FEATURES = (
    "logv_std",
    "logv_iqr",
    "nr_logmean",
    "nr_median",
    "coh_mean",
    "sign_agree",
    "m_g_norm",
    "g_cv",
    "v_cv",
    "exposure_ref",
)


def make_spd(dim: int, condition: float, rotated: bool, rng: np.random.Generator) -> np.ndarray:
    eigenvalues = np.logspace(0.0, np.log10(condition), dim)
    if not rotated:
        return np.diag(eigenvalues)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return (q * eigenvalues) @ q.T


def noisy_gradient(regime: str, true_grad: np.ndarray, diag_h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(diag_h)
    if regime == "clean":
        return true_grad + 0.02 * scale * rng.normal(size=true_grad.size)
    if regime == "gaussian":
        return true_grad + 0.5 * scale * rng.normal(size=true_grad.size)
    if regime == "heavy":
        return true_grad + 0.35 * scale * rng.standard_t(2.5, size=true_grad.size)
    if regime == "signflip":
        flips = np.where(rng.random(true_grad.size) < 0.3, -1.0, 1.0)
        return flips * true_grad + 0.05 * scale * rng.normal(size=true_grad.size)
    raise ValueError(regime)


def online_features(theta, grad, m_hat, v_hat, d_ref, lr: float) -> list[float]:
    sqrt_v = np.sqrt(v_hat) + EPS
    noise_ratio = sqrt_v / (np.abs(m_hat) + EPS)
    coherence = np.clip(np.abs(m_hat) / sqrt_v, 0.0, 1.0)
    log_v = np.log(v_hat + EPS)
    return [
        float(log_v.std()),
        float(np.quantile(log_v, 0.75) - np.quantile(log_v, 0.25)),
        float(np.mean(np.log1p(noise_ratio))),
        float(np.median(noise_ratio)),
        float(coherence.mean()),
        float(np.mean(np.sign(grad) * np.sign(m_hat))),
        float(np.linalg.norm(m_hat) / (np.linalg.norm(grad) + EPS)),
        float(np.std(np.abs(grad)) / (np.mean(np.abs(grad)) + EPS)),
        float(np.std(v_hat) / (np.mean(v_hat) + EPS)),
        float(np.mean(lr * np.abs(d_ref) / (np.abs(theta) + lr * np.abs(d_ref) + EPS))),
    ]


def protection_candidates(theta, grad, m_hat, v_hat, psi_prev, candidates, lr: float):
    sqrt_v = np.sqrt(v_hat) + EPS
    coherence = np.clip(np.abs(m_hat) / sqrt_v, 0.0, 1.0)[None, :]
    direction = np.sign(grad[None, :] * candidates)
    base_step = lr * np.abs(candidates)
    exposure = base_step / (np.abs(theta)[None, :] + 2.0 * base_step + EPS)
    trust = np.clip(direction * coherence**2 * (1.0 - exposure) - exposure, -1.0, 1.0)
    psi = RHO_PSI * psi_prev[None, :] + (1.0 - RHO_PSI) * trust
    z = np.tanh(LAMBDA_GATE * psi)
    weights = candidates**2
    mu = (weights * z).sum(axis=1) / (weights.sum(axis=1) + EPS)
    kappa = np.sqrt(np.maximum(0.25, 1.0 + ALPHA * (z - mu[:, None])))
    return psi, kappa


def run_shadow(seed: int, condition: float, rotated: bool, regime: str, *, dim: int, steps: int, burn: int, lr: float):
    rng = np.random.default_rng(seed)
    hessian = make_spd(dim, condition, rotated, rng)
    diag_h = np.diag(hessian)
    theta = rng.normal(size=dim) / np.sqrt(diag_h) * 2.0
    m = np.zeros(dim)
    v = np.zeros(dim)
    psi = np.zeros(dim)
    rows = []

    for step in range(1, steps + 1):
        true_grad = hessian @ theta
        grad = noisy_gradient(regime, true_grad, diag_h, rng)
        m = BETA1 * m + (1.0 - BETA1) * grad
        v = BETA2 * v + (1.0 - BETA2) * grad * grad
        m_hat = m / (1.0 - BETA1**step)
        v_hat = v / (1.0 - BETA2**step)

        candidates = m_hat[None, :] / np.power(v_hat[None, :] + EPS, GAMMAS[:, None])
        reference_norm = np.linalg.norm(candidates[-1])
        candidate_norms = np.linalg.norm(candidates, axis=1) + EPS
        candidates = candidates * (reference_norm / candidate_norms)[:, None]

        psi_candidates, kappa = protection_candidates(theta, grad, m_hat, v_hat, psi, candidates, lr)
        next_theta = theta[None, :] - lr * kappa * candidates
        losses = 0.5 * np.einsum("gi,ij,gj->g", next_theta, hessian, next_theta)
        oracle_idx = int(np.argmin(losses))

        if step > burn:
            before = float(0.5 * theta @ (hessian @ theta))
            row = {
                "seed": seed,
                "condition": condition,
                "rotated": int(rotated),
                "regime": regime,
                "step": step,
                "gamma_star": float(GAMMAS[oracle_idx]),
                "f_before": before,
            }
            for index, loss in enumerate(losses):
                row[f"loss_{index}"] = float(loss)
            row.update(dict(zip(FEATURES, online_features(theta, grad, m_hat, v_hat, candidates[-1], lr))))
            rows.append(row)

        # Shadow trajectory remains the current gamma=1/2 navigator.
        theta = next_theta[-1]
        psi = psi_candidates[-1]

    return rows


def evaluate_predictor(name, model, train, test):
    x_train = train[list(FEATURES)].to_numpy()
    y_train = train["gamma_star"].to_numpy()
    x_test = test[list(FEATURES)].to_numpy()
    y_test = test["gamma_star"].to_numpy()
    model.fit(x_train, y_train)
    prediction = np.clip(model.predict(x_test), 0.0, 0.5)
    indices = np.abs(prediction[:, None] - GAMMAS[None, :]).argmin(axis=1)
    matrix = test[[f"loss_{i}" for i in range(len(GAMMAS))]].to_numpy()
    selected = matrix[np.arange(len(test)), indices]
    oracle = matrix.min(axis=1)
    adam = matrix[:, -1]
    before = test["f_before"].to_numpy()
    return {
        "model": name,
        "mae_gamma": mean_absolute_error(y_test, prediction),
        "r2_gamma": r2_score(y_test, prediction),
        "mean_post_ratio": float(np.mean(selected / before)),
        "mean_regret_vs_oracle": float(np.mean((selected - oracle) / before)),
        "improved_vs_adam": float(np.mean(selected < adam)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/n15"))
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--burn", type=int, default=15)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--tune-seeds", type=int, default=4)
    parser.add_argument("--held-seeds", type=int, default=6)
    args = parser.parse_args()

    rows = []
    total_seeds = args.tune_seeds + args.held_seeds
    for seed in range(total_seeds):
        for condition in (10.0, 100.0, 1000.0):
            for rotated in (False, True):
                for regime in ("clean", "gaussian", "heavy", "signflip"):
                    rows.extend(run_shadow(seed, condition, rotated, regime, dim=args.dim, steps=args.steps, burn=args.burn, lr=args.lr))

    frame = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "n15_shadow.csv", index=False)

    train = frame[frame["seed"] < args.tune_seeds].copy()
    test = frame[frame["seed"] >= args.tune_seeds].copy()

    fixed = []
    for index, gamma in enumerate(GAMMAS):
        fixed.append({
            "gamma": gamma,
            "tune_post_ratio": float(np.mean(train[f"loss_{index}"] / train["f_before"])),
            "held_post_ratio": float(np.mean(test[f"loss_{index}"] / test["f_before"])),
        })
    pd.DataFrame(fixed).to_csv(args.outdir / "n15_fixed_gamma.csv", index=False)

    models = [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("rf", RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=20, max_features=0.8, random_state=0, n_jobs=-1)),
        ("hgb", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=100, l2_regularization=1.0, random_state=0)),
    ]
    results = [evaluate_predictor(name, model, train, test) for name, model in models]
    pd.DataFrame(results).to_csv(args.outdir / "n15_predictors.csv", index=False)

    matrix = test[[f"loss_{i}" for i in range(len(GAMMAS))]].to_numpy()
    oracle = matrix.min(axis=1)
    adam = matrix[:, -1]
    before = test["f_before"].to_numpy()
    summary = pd.DataFrame([
        {"policy": "fixed_gamma_0.5", "mean_post_ratio": float(np.mean(adam / before)), "mean_regret_vs_oracle": float(np.mean((adam - oracle) / before))},
        {"policy": "one_step_oracle", "mean_post_ratio": float(np.mean(oracle / before)), "mean_regret_vs_oracle": 0.0},
    ])
    summary.to_csv(args.outdir / "n15_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()

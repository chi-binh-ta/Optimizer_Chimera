from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader

from chimera.bitconv import BitConv2d
from chimera.bitlinear import BitLinear
from chimera.models.resnet_cifar import ChimeraResNet20
from chimera.quantization import abs_stat, ternary_stats
from chimera.target_bits import TargetBitsController
from experiments.train_resnet_cifar import (
    _load_cifar10,
    apply_target_bits_control,
    initialize_alpha_override,
    model_stats,
    quant_config_from_config,
)
from chimera.optimizer import load_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "g1"
OUT.mkdir(parents=True, exist_ok=True)

# G-1 is a frozen architectural validation, not another tuning round.
SEEDS = (20261, 20262, 20263, 20264, 20265)
LR = 1.0e-3
EPOCHS = 5
MAX_BATCHES = 100
EVAL_MAX_BATCHES = 20
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
TARGET_BITS = 1.58
TARGET_BRANCH = "sparse"
CONTROLLER_AFFECTS = "alpha_override"
NON_TUNE_PROTOCOL = True
GRAD_COVERAGE_MIN = 0.80


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProtectionLite(torch.optim.Optimizer):
    """Frozen coordinate Protection-Lite, FP32 z, Adam-like Navigation."""

    def __init__(
        self,
        params,
        *,
        lr: float = LR,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        alpha: float = 0.375,
        z_slope: float = 3.0,
    ) -> None:
        defaults = dict(
            lr=float(lr), beta1=float(beta1), beta2=float(beta2), eps=float(eps),
            alpha=float(alpha), z_slope=float(z_slope)
        )
        super().__init__(params, defaults)
        self.max_norm_error = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            eps = group["eps"]
            alpha = group["alpha"]
            z_slope = group["z_slope"]
            prepared = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("ProtectionLite G-1 does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                mh = m / (1.0 - b1 ** t)
                vh = v / (1.0 - b2 ** t)
                d = mh / (vh.sqrt() + eps)

                c = d.abs().clamp(0.0, 1.0)
                base = lr * d.abs()
                r = base / (p.detach().abs() + 2.0 * base + eps)
                tau = (torch.sign(g * d) * c.square() * (1.0 - r) - r).clamp(-1.0, 1.0)
                tau = torch.where(g == 0, torch.zeros_like(tau), tau)
                z = torch.clamp(z_slope * tau, -1.0, 1.0)  # FP32, not P-C1 coded z.
                w = d.square()
                prepared.append((p, d, z, w))

            if not prepared:
                continue
            total_w = sum((w.sum() for _, _, _, w in prepared), torch.zeros((), device=prepared[0][0].device))
            total_q = sum(((w * z).sum() for _, _, z, w in prepared), torch.zeros((), device=prepared[0][0].device))
            mu = total_q / total_w.clamp_min(1e-20)
            before = float(total_w)
            after = 0.0
            for p, d, z, _ in prepared:
                kappa = (1.0 + alpha * (z - mu)).clamp_min(0.25).sqrt()
                u = kappa * d
                p.add_(u, alpha=-lr)
                after += float(u.square().sum())
            if before > 0:
                err = abs(math.sqrt(after) - math.sqrt(before)) / (math.sqrt(before) + 1e-20)
                self.max_norm_error = max(self.max_norm_error, err)
        return loss


def _bitconv_forward_ste(self: BitConv2d, x: torch.Tensor) -> torch.Tensor:
    """Same forward values as repository BitConv2d, identity STE through activation quantization."""
    if x.ndim != 4 or x.shape[1] != self.in_channels:
        raise ValueError(f"invalid BitConv2d input shape {tuple(x.shape)}")
    beta = x.detach().abs().amax().div(127.0).add(self.eps_beta)
    x_scaled = x / beta
    x_raw = torch.clamp(torch.round(x_scaled), -127, 127)
    x_q = x_scaled + (x_raw - x_scaled).detach()

    if self.training:
        with torch.no_grad():
            s = abs_stat(self.weight_master.detach(), self.stat_mode).to(self.scale_ema)
            self.scale_ema.mul_(self.rho_s).add_(s, alpha=1.0 - self.rho_s)
    gamma = self.compute_gamma()
    alpha = self.current_alpha()
    w_q_raw = self.quantize_weight(gamma=gamma, alpha=alpha)
    w_q = self.weight_master + (w_q_raw - self.weight_master).detach()
    if self.ste_clip is not None:
        w_q = torch.clamp(w_q, -float(self.ste_clip), float(self.ste_clip))
    y_int = F.conv2d(x_q, w_q, bias=None, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
    y = (beta * gamma) * y_int
    if self.bias is not None:
        y = y + self.bias.view(1, -1, 1, 1)
    self._last_stats = {"gamma": float(gamma.detach().item()), "alpha": float(alpha), **ternary_stats(w_q_raw.detach())}
    if self.training:
        self.forward_step.add_(1)
    return y


def _bitlinear_forward_ste(self: BitLinear, x: torch.Tensor) -> torch.Tensor:
    """Same forward values as repository BitLinear, identity STE through activation quantization."""
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    beta = x.detach().abs().amax().div(127.0).add(self.eps_beta)
    x_scaled = x / beta
    x_raw = torch.clamp(torch.round(x_scaled), -127, 127)
    x_q = x_scaled + (x_raw - x_scaled).detach()
    if self.training:
        with torch.no_grad():
            s = abs_stat(self.weight_master.detach(), self.stat_mode).to(self.scale_ema)
            self.scale_ema.mul_(self.rho_s).add_(s, alpha=1.0 - self.rho_s)
    gamma = self.compute_gamma()
    alpha = self.current_alpha()
    w_q_raw = self.quantize_weight(gamma=gamma, alpha=alpha)
    w_q = self.weight_master + (w_q_raw - self.weight_master).detach()
    if self.ste_clip is not None:
        w_q = torch.clamp(w_q, -float(self.ste_clip), float(self.ste_clip))
    y_int = x_q @ w_q.t()
    y = (beta * gamma) * y_int
    if self.bias is not None:
        y = y + self.bias
    self._last_stats = {"gamma": float(gamma.detach().item()), "alpha": float(alpha), **ternary_stats(w_q_raw.detach())}
    if self.training:
        self.forward_step.add_(1)
    return y


def install_activation_ste_repair() -> None:
    BitConv2d.forward = _bitconv_forward_ste
    BitLinear.forward = _bitlinear_forward_ste


def build_model(config: dict, seed: int) -> ChimeraResNet20:
    seed_all(seed)
    model = ChimeraResNet20(
        mode="chimera",
        quant_config=quant_config_from_config(config),
        first_last_fp32=False,
    )
    initialize_alpha_override(model, float(config["alpha_target"]))
    return model


def gradient_coverage(model: torch.nn.Module, batch, device: torch.device) -> dict[str, float]:
    model.to(device).train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    model.zero_grad(set_to_none=True)
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    total = 0
    active = 0
    q_total = 0
    q_active = 0
    tensor_total = 0
    tensor_active = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += p.numel()
        tensor_total += 1
        is_active = p.grad is not None and float(p.grad.detach().norm()) > 1e-12
        if is_active:
            active += p.numel()
            tensor_active += 1
        if "weight_master" in name:
            q_total += p.numel()
            if is_active:
                q_active += p.numel()
    return {
        "loss": float(loss.detach()),
        "active_numel_fraction": active / max(total, 1),
        "active_tensor_fraction": tensor_active / max(tensor_total, 1),
        "active_quant_weight_fraction": q_active / max(q_total, 1),
        "total_trainable_numel": total,
        "quant_weight_numel": q_total,
    }


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    for idx, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(loss_fn(logits, y))
        correct += int((logits.argmax(1) == y).sum())
        seen += y.numel()
        if max_batches and idx >= max_batches:
            break
    return loss_sum / max(seen, 1), correct / max(seen, 1)


def make_loaders(train_ds, test_ds, seed: int):
    gen = torch.Generator().manual_seed(seed)
    train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen, num_workers=0)
    test = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    return train, test


def run_arm(name: str, seed: int, config: dict, train_ds, test_ds, device: torch.device) -> dict:
    train_loader, test_loader = make_loaders(train_ds, test_ds, seed)
    model = build_model(config, seed).to(device)
    controller_cfg = config.get("target_bits_controller", {})
    controller = TargetBitsController(
        target_bits=TARGET_BITS,
        branch=TARGET_BRANCH,
        tolerance=float(controller_cfg.get("tolerance", 0.02)),
        step_size=float(controller_cfg.get("step_size", 0.05)),
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(controller_cfg.get("alpha_max", 1.5)),
    )
    if name == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=LR, betas=(float(config["beta1"]), float(config["beta2"])),
            eps=float(config["eps_opt"]), weight_decay=0.0
        )
    elif name == "protection_lite":
        opt = ProtectionLite(
            model.parameters(), lr=LR, beta1=float(config["beta1"]), beta2=float(config["beta2"]),
            eps=float(config["eps_opt"]), alpha=0.375, z_slope=3.0
        )
    else:
        raise ValueError(name)

    initial_test_loss, initial_test_acc = evaluate(model, test_loader, device, EVAL_MAX_BATCHES)
    loss_fn = torch.nn.CrossEntropyLoss()
    steps = 0
    epoch_rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        train_correct = 0
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            steps += 1
            train_loss_sum += float(loss.detach()) * y.numel()
            train_seen += y.numel()
            train_correct += int((logits.argmax(1) == y).sum())
            if batch_idx >= MAX_BATCHES:
                break
        stats = model_stats(model)
        next_alpha, alpha_delta, controller_error = apply_target_bits_control(
            model, controller, current_zero_ratio=stats["global_zero_ratio"], affects=CONTROLLER_AFFECTS
        )
        test_loss, test_acc = evaluate(model, test_loader, device, EVAL_MAX_BATCHES)
        epoch_rows.append({
            "epoch": epoch,
            "steps": steps,
            "train_loss": train_loss_sum / max(train_seen, 1),
            "train_accuracy": train_correct / max(train_seen, 1),
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "zero_ratio": stats["global_zero_ratio"],
            "effective_bits": stats["global_effective_bits"],
            "alpha_next": next_alpha,
            "alpha_delta": alpha_delta,
            "controller_error": controller_error,
        })
    final_stats = model_stats(model)
    return {
        "optimizer": name,
        "seed": seed,
        "initial_test_loss": initial_test_loss,
        "initial_test_accuracy": initial_test_acc,
        "final_test_loss": epoch_rows[-1]["test_loss"],
        "final_test_accuracy": epoch_rows[-1]["test_accuracy"],
        "final_train_loss": epoch_rows[-1]["train_loss"],
        "final_train_accuracy": epoch_rows[-1]["train_accuracy"],
        "final_zero_ratio": final_stats["global_zero_ratio"],
        "final_effective_bits": final_stats["global_effective_bits"],
        "steps": steps,
        "max_norm_error": float(getattr(opt, "max_norm_error", 0.0)),
        "epochs": epoch_rows,
    }


def one_sided_t_ucb(values: np.ndarray, confidence: float = 0.95) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("inf")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    critical = float(scipy_stats.t.ppf(confidence, df=values.size - 1))
    return mean + critical * sd / math.sqrt(values.size)


def bootstrap_mean_ci(values: np.ndarray, seed: int = 55117, B: int = 50000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = values.size
    samples = np.array([values[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)

    data_root = ROOT / "data"
    train_ds = _load_cifar10(data_root, train=True, download=True)
    test_ds = _load_cifar10(data_root, train=False, download=True)

    # Validity gate: exact repository activation backward versus repaired STE backward.
    probe_loader, _ = make_loaders(train_ds, test_ds, SEEDS[0])
    probe_batch = next(iter(probe_loader))
    historical = gradient_coverage(build_model(config, SEEDS[0]), probe_batch, device)
    install_activation_ste_repair()
    repaired = gradient_coverage(build_model(config, SEEDS[0]), probe_batch, device)
    validity = {
        "historical_activation_backward": historical,
        "activation_ste_repair": repaired,
        "historical_valid_for_optimizer_comparison": bool(historical["active_quant_weight_fraction"] >= GRAD_COVERAGE_MIN),
        "repaired_valid_for_optimizer_comparison": bool(repaired["active_quant_weight_fraction"] >= GRAD_COVERAGE_MIN),
        "coverage_threshold": GRAD_COVERAGE_MIN,
    }
    (OUT / "g1_gradient_coverage.json").write_text(json.dumps(validity, indent=2), encoding="utf-8")
    if not validity["repaired_valid_for_optimizer_comparison"]:
        raise RuntimeError(f"G-1 BLOCKED: repaired gradient coverage below threshold: {validity}")

    results = []
    for seed in SEEDS:
        # Each arm gets the same initialization seed and same shuffled data order.
        for name in ("adam", "protection_lite"):
            print(f"[G1] seed={seed} optimizer={name} device={device}", flush=True)
            row = run_arm(name, seed, config, train_ds, test_ds, device)
            results.append(row)
            (OUT / "g1_progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat = []
    for r in results:
        flat.append({k: v for k, v in r.items() if k != "epochs"})
    with (OUT / "g1_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader(); writer.writerows(flat)

    by = {(r["seed"], r["optimizer"]): r for r in results}
    pairs = []
    for seed in SEEDS:
        a = by[(seed, "adam")]
        l = by[(seed, "protection_lite")]
        init = 0.5 * (a["initial_test_loss"] + l["initial_test_loss"])
        delta = (l["final_test_loss"] - a["final_test_loss"]) / (abs(init) + 1e-12)
        pairs.append({
            "seed": seed,
            "initial_test_loss": init,
            "adam_final_test_loss": a["final_test_loss"],
            "lite_final_test_loss": l["final_test_loss"],
            "adam_final_test_accuracy": a["final_test_accuracy"],
            "lite_final_test_accuracy": l["final_test_accuracy"],
            "normalized_delta_lite_minus_adam": delta,
            "lite_minus_adam_accuracy": l["final_test_accuracy"] - a["final_test_accuracy"],
            "adam_effective_bits": a["final_effective_bits"],
            "lite_effective_bits": l["final_effective_bits"],
            "adam_zero_ratio": a["final_zero_ratio"],
            "lite_zero_ratio": l["final_zero_ratio"],
        })
    with (OUT / "g1_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader(); writer.writerows(pairs)

    deltas = np.array([p["normalized_delta_lite_minus_adam"] for p in pairs], dtype=float)
    ci025, ci975 = bootstrap_mean_ci(deltas)
    t_ucb95 = one_sided_t_ucb(deltas, 0.95)
    mean_delta = float(deltas.mean())
    verdict = "PASS" if t_ucb95 <= 0.0 else "FAIL"
    summary = {
        "round": "G-1 Generalization Freeze Audit",
        "candidate": "Protection-Lite coordinate FP32 z",
        "execution": "CIFAR/Chimera forward quantization with common activation-STE backward repair",
        "historical_pipeline_gradient_coverage": validity,
        "protocol": {
            "dataset": "CIFAR-10",
            "model": "ChimeraResNet20",
            "quant_mode": "chimera",
            "target_bits": TARGET_BITS,
            "target_branch": TARGET_BRANCH,
            "controller_frequency": "epoch",
            "controller_affects": CONTROLLER_AFFECTS,
            "lr": LR,
            "epochs": EPOCHS,
            "max_batches_per_epoch": MAX_BATCHES,
            "eval_max_batches": EVAL_MAX_BATCHES,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "seeds": list(SEEDS),
            "retuning": False,
        },
        "criterion": "one-sided 95% t-UCB of paired normalized test-loss delta <= 0",
        "mean_normalized_delta_lite_minus_adam": mean_delta,
        "bootstrap_95ci": [ci025, ci975],
        "one_sided_t_ucb95": t_ucb95,
        "win_fraction": float(np.mean(deltas < 0.0)),
        "mean_accuracy_delta_lite_minus_adam": float(np.mean([p["lite_minus_adam_accuracy"] for p in pairs])),
        "verdict": verdict,
    }
    (OUT / "g1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# G-1 — Generalization Freeze Audit\n\nCandidate frozen: Protection-Lite coordinate FP32 z.\nNo LR tuning, no retry, no P-C1 coding.\n\nHistorical backward gradient coverage (quantized weights): {historical['active_quant_weight_fraction']:.6f}\nSTE-repaired backward gradient coverage: {repaired['active_quant_weight_fraction']:.6f}\n\nPrimary paired test-loss metric:\n- mean normalized Lite-Adam delta: {mean_delta:.10f}\n- bootstrap 95% CI: [{ci025:.10f}, {ci975:.10f}]\n- one-sided t-UCB95: {t_ucb95:.10f}\n- win fraction: {summary['win_fraction']:.4f}\n- mean accuracy delta: {summary['mean_accuracy_delta_lite_minus_adam']:.6f}\n\nPrecommitted criterion: UCB95 <= 0.\n\n## Verdict: {verdict}\n\nThis is an out-of-family validation on real CIFAR-10. The activation STE repair is common to both optimizer arms and preserves forward quantized values; it is used only because the repository's historical activation round blocks upstream gradient propagation.\n"""
    (OUT / "G1_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

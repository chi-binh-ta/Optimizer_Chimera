"""Agreement-gated optimizer for Chimera 2.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import yaml

ZERO_GRAD_POLICIES = ("standard", "freeze", "decay", "inertia")
KAPPA_GATE_MODES = ("none", "noise_ratio")
PSI_STORAGE_MODES = ("fp32", "int8")
TRUST_MODES = ("raw_agreement", "step_trust", "step_trust_local")


def _apply_kappa_gate(kappa_raw: torch.Tensor, noise_ratio: torch.Tensor, *, mode: str, noise_threshold: float, gate_sharpness: float) -> torch.Tensor:
    if mode == "none":
        return kappa_raw
    if mode != "noise_ratio":
        raise ValueError("kappa_gate_mode must be 'none' or 'noise_ratio'")
    gate = torch.sigmoid(gate_sharpness * (noise_threshold - noise_ratio))
    return 1.0 + gate * (kappa_raw - 1.0)


def _quantize_psi(psi: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(psi.clamp(-1.0, 1.0) * 127.0), -127, 127).to(torch.int8)


def _dequantize_psi(psi: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if psi.dtype == torch.int8:
        return psi.to(dtype=dtype).div(127.0)
    return psi


def _step_trust_statistic(*, grad: torch.Tensor, param: torch.Tensor, m_hat: torch.Tensor, sqrt_v_hat: torch.Tensor, d: torch.Tensor, lr: float, eps_opt: float, exposure_mode: str = "tensor_mean") -> torch.Tensor:
    """Bounded step-trust statistic in [-1, 1] with no persistent state.

    ``tensor_mean`` is the P-2.5 exposure using mean(abs(param)).
    ``local_capped`` is P-2.6: reduction-free, scale-relative, and bounded by 1/2.
    """
    coherence = (m_hat.abs() / (sqrt_v_hat + eps_opt)).clamp_(0.0, 1.0)
    direction = torch.sign(grad * d)
    base_step = lr * d.abs()
    if exposure_mode == "tensor_mean":
        param_scale = param.detach().abs().mean()
        exposure = base_step / (param.detach().abs() + param_scale + base_step + eps_opt)
    elif exposure_mode == "local_capped":
        exposure = base_step / (param.detach().abs() + 2.0 * base_step + eps_opt)
    else:
        raise ValueError("exposure_mode must be 'tensor_mean' or 'local_capped'")
    trust = direction * coherence.square() * (1.0 - exposure) - exposure
    trust = trust.clamp_(-1.0, 1.0)
    # Preserve current zero-gradient policy semantics: raw agreement is 0 at g=0.
    return torch.where(grad == 0, torch.zeros_like(trust), trust)


class Chimera21(torch.optim.Optimizer):
    """Adam-like optimizer with experimental protection-statistic modes."""

    def __init__(self, params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps_opt: float = 1e-8,
                 weight_decay: float = 0.0, rho_psi: float = 0.95, lambda_gate: float = 0.1,
                 kappa_min: float = 0.5, kappa_max: float = 2.0, log_diagnostics: bool = False,
                 zero_grad_policy: str = "standard", idle_decay: float = 0.95,
                 kappa_gate_mode: str = "none", noise_threshold: float = 2.0,
                 gate_sharpness: float = 4.0, psi_storage: str = "fp32",
                 trust_mode: str = "raw_agreement") -> None:
        if lr < 0: raise ValueError("lr must be non-negative")
        if not 0.0 <= beta1 < 1.0: raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0: raise ValueError("beta2 must be in [0, 1)")
        if eps_opt <= 0: raise ValueError("eps_opt must be positive")
        if weight_decay < 0: raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= rho_psi <= 1.0: raise ValueError("rho_psi must be in [0, 1]")
        if kappa_min <= 0 or kappa_max < kappa_min: raise ValueError("kappa bounds must satisfy 0 < kappa_min <= kappa_max")
        if zero_grad_policy not in ZERO_GRAD_POLICIES: raise ValueError(f"zero_grad_policy must be one of {ZERO_GRAD_POLICIES}")
        if not 0.0 <= idle_decay <= 1.0: raise ValueError("idle_decay must be in [0, 1]")
        if kappa_gate_mode not in KAPPA_GATE_MODES: raise ValueError(f"kappa_gate_mode must be one of {KAPPA_GATE_MODES}")
        if gate_sharpness < 0: raise ValueError("gate_sharpness must be non-negative")
        if psi_storage not in PSI_STORAGE_MODES: raise ValueError(f"psi_storage must be one of {PSI_STORAGE_MODES}")
        if trust_mode not in TRUST_MODES: raise ValueError(f"trust_mode must be one of {TRUST_MODES}")
        defaults = dict(lr=lr,beta1=beta1,beta2=beta2,eps_opt=eps_opt,weight_decay=weight_decay,
                        rho_psi=rho_psi,lambda_gate=lambda_gate,kappa_min=kappa_min,kappa_max=kappa_max,
                        log_diagnostics=bool(log_diagnostics),zero_grad_policy=zero_grad_policy,idle_decay=idle_decay,
                        kappa_gate_mode=kappa_gate_mode,noise_threshold=noise_threshold,gate_sharpness=gate_sharpness,
                        psi_storage=psi_storage,trust_mode=trust_mode)
        super().__init__(params, defaults)
        self.last_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        diagnostics_enabled = any(bool(g["log_diagnostics"]) for g in self.param_groups)
        diag = None
        if diagnostics_enabled:
            diag = {"count":0.0,"sum_kappa":0.0,"sum_kappa_sq":0.0,"min_kappa":float("inf"),"max_kappa":float("-inf"),
                    "sum_noise_ratio":0.0,"sum_effective_lr":0.0,"sum_active_effective_lr":0.0,"active_grad_count":0.0,
                    "sum_update_sq":0.0,"sum_param_sq":0.0,"sum_update_abs":0.0,"max_update_abs":0.0,"zero_grad_count":0.0,
                    "sum_psi_abs":0.0,"psi_saturation_count":0.0,"sum_collision_score":0.0,"sum_trust":0.0,"sum_trust_abs":0.0}
        for group in self.param_groups:
            lr=group["lr"]; beta1=group["beta1"]; beta2=group["beta2"]; eps_opt=group["eps_opt"]
            weight_decay=group["weight_decay"]; rho_psi=group["rho_psi"]; lambda_gate=group["lambda_gate"]
            kappa_min=group["kappa_min"]; kappa_max=group["kappa_max"]; zero_grad_policy=group["zero_grad_policy"]
            idle_decay=group["idle_decay"]; kappa_gate_mode=group["kappa_gate_mode"]; noise_threshold=group["noise_threshold"]
            gate_sharpness=group["gate_sharpness"]; psi_storage=group["psi_storage"]; trust_mode=group["trust_mode"]
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                if grad.is_sparse: raise RuntimeError("Chimera21 does not support sparse gradients")
                state=self.state[p]
                if len(state)==0:
                    state["m"]=torch.zeros_like(p); state["v"]=torch.zeros_like(p)
                    state["psi"]=torch.zeros_like(p,dtype=torch.int8) if psi_storage=="int8" else torch.zeros_like(p)
                    state["step"]=0
                m=state["m"]; v=state["v"]; psi=_dequantize_psi(state["psi"],dtype=p.dtype)
                state["step"]+=1; step=state["step"]
                if weight_decay != 0.0: p.mul_(1.0-lr*weight_decay)
                m.mul_(beta1).add_(grad,alpha=1.0-beta1)
                v.mul_(beta2).addcmul_(grad,grad,value=1.0-beta2)
                m_hat=m/(1.0-beta1**step); v_hat=v/(1.0-beta2**step); sqrt_v_hat=v_hat.sqrt(); denom=sqrt_v_hat.add(eps_opt); d=m_hat/denom
                zero_mask=(grad==0 if zero_grad_policy!="standard" or diagnostics_enabled else None)
                if trust_mode == "raw_agreement":
                    trust = torch.sign(m)*torch.sign(grad)
                elif trust_mode == "step_trust":
                    trust = _step_trust_statistic(grad=grad,param=p,m_hat=m_hat,sqrt_v_hat=sqrt_v_hat,d=d,lr=lr,eps_opt=eps_opt,exposure_mode="tensor_mean")
                else:
                    trust = _step_trust_statistic(grad=grad,param=p,m_hat=m_hat,sqrt_v_hat=sqrt_v_hat,d=d,lr=lr,eps_opt=eps_opt,exposure_mode="local_capped")
                if zero_grad_policy=="standard" or zero_mask is None:
                    psi.mul_(rho_psi).add_(trust,alpha=1.0-rho_psi)
                else:
                    updated=psi*rho_psi+trust*(1.0-rho_psi)
                    if zero_grad_policy=="freeze": psi.copy_(torch.where(zero_mask,psi,updated))
                    elif zero_grad_policy=="decay": psi.copy_(torch.where(zero_mask,psi*idle_decay,updated))
                    elif zero_grad_policy=="inertia":
                        inertia_agreement=torch.sign(m)*torch.sign(m)
                        inertia_updated=psi*rho_psi+inertia_agreement*(1.0-rho_psi)
                        psi.copy_(torch.where(zero_mask,inertia_updated,updated))
                kappa_raw=torch.exp(lambda_gate*psi).clamp(kappa_min,kappa_max)
                if kappa_gate_mode=="noise_ratio" or diagnostics_enabled:
                    noise_ratio=sqrt_v_hat/(m_hat.abs().add(eps_opt))
                else: noise_ratio=None
                if kappa_gate_mode=="noise_ratio":
                    kappa=_apply_kappa_gate(kappa_raw,noise_ratio,mode=kappa_gate_mode,noise_threshold=noise_threshold,gate_sharpness=gate_sharpness)
                else: kappa=kappa_raw
                if psi_storage=="int8": state["psi"].copy_(_quantize_psi(psi))
                else: state["psi"].copy_(psi)
                if diag is not None:
                    count=float(kappa.numel()); diag["count"]+=count; kd=kappa.detach(); nd=noise_ratio.detach(); dd=denom.detach()
                    effective_lr=lr*kd/dd; update=(lr*kd*d.detach()).abs(); td=trust.detach()
                    diag["sum_kappa"]+=float(kd.sum()); diag["sum_kappa_sq"]+=float((kd*kd).sum()); diag["min_kappa"]=min(diag["min_kappa"],float(kd.min())); diag["max_kappa"]=max(diag["max_kappa"],float(kd.max()))
                    diag["sum_noise_ratio"]+=float(nd.sum()); diag["sum_effective_lr"]+=float(effective_lr.sum()); diag["sum_trust"]+=float(td.sum()); diag["sum_trust_abs"]+=float(td.abs().sum())
                    active_mask=(grad!=0) if zero_mask is None else ~zero_mask; zero_count=float((~active_mask).sum()); diag["zero_grad_count"]+=zero_count
                    active_count=float(active_mask.sum()); diag["active_grad_count"]+=active_count
                    if active_count>0: diag["sum_active_effective_lr"]+=float(effective_lr[active_mask].sum())
                    un=float(update.norm()); pn=float(p.detach().norm()); diag["sum_update_sq"]+=un**2; diag["sum_param_sq"]+=pn**2; diag["sum_update_abs"]+=float(update.sum()); diag["max_update_abs"]=max(diag["max_update_abs"],float(update.max()))
                    pd=psi.detach(); diag["sum_psi_abs"]+=float(pd.abs().sum()); diag["psi_saturation_count"]+=float((pd.abs()>=0.999).sum()); diag["sum_collision_score"]+=float((kd.log().abs()*nd).sum())
                p.add_(kappa*d,alpha=-lr)
        if diag is not None and diag["count"]>0:
            count=diag["count"]; mk=diag["sum_kappa"]/count; variance=max(diag["sum_kappa_sq"]/count-mk**2,0.0); aun=diag["sum_update_sq"]**0.5
            self.last_diagnostics={"mean_kappa":mk,"std_kappa":variance**0.5,"min_kappa":diag["min_kappa"],"max_kappa":diag["max_kappa"],
                "mean_noise_ratio":diag["sum_noise_ratio"]/count,"mean_effective_lr":diag["sum_effective_lr"]/count,
                "active_mean_effective_lr":diag["sum_active_effective_lr"]/diag["active_grad_count"] if diag["active_grad_count"]>0 else 0.0,
                "applied_update_norm":aun,"param_update_ratio":aun/(diag["sum_param_sq"]**0.5+1e-12),"mean_update_abs":diag["sum_update_abs"]/count,
                "max_update_abs":diag["max_update_abs"],"zero_grad_ratio":diag["zero_grad_count"]/count,"active_grad_ratio":1.0-diag["zero_grad_count"]/count,
                "psi_abs_mean":diag["sum_psi_abs"]/count,"psi_saturation_ratio":diag["psi_saturation_count"]/count,"collision_score":diag["sum_collision_score"]/count,
                "mean_trust_stat":diag["sum_trust"]/count,"mean_abs_trust_stat":diag["sum_trust_abs"]/count}
        elif diagnostics_enabled: self.last_diagnostics={}
        return loss


def _tensor_state_bytes(value: Any) -> int:
    return int(value.numel()*value.element_size()) if isinstance(value,torch.Tensor) else 0

def optimizer_state_memory_summary(optimizer: torch.optim.Optimizer) -> dict[str,int]:
    summary={"m":0,"v":0,"psi":0,"other":0,"total":0}; key_map={"exp_avg":"m","exp_avg_sq":"v"}
    for state in optimizer.state.values():
        for key,value in state.items():
            b=_tensor_state_bytes(value)
            if b==0: continue
            mk=key_map.get(key,key)
            if mk in summary and mk!="total": summary[mk]+=b
            else: summary["other"]+=b
            summary["total"]+=b
    return summary

def optimizer_state_memory_bytes(optimizer: torch.optim.Optimizer)->int:
    return optimizer_state_memory_summary(optimizer)["total"]
def load_config(path: str|Path)->dict[str,Any]:
    cp=Path(path)
    with cp.open("r",encoding="utf-8") as handle: data=yaml.safe_load(handle) or {}
    if not isinstance(data,dict): raise ValueError("config file must contain a YAML mapping")
    return data

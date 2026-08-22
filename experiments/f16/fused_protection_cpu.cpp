#include <torch/extension.h>
#include <ATen/cpu/vec/vec.h>
#include <vector>
#include <cmath>
#include <algorithm>

using Vec = at::vec::Vectorized<float>;

// F-1.6 experimental CPU kernel. Mathematics is frozen at P-3.5:
// tau -> psi -> z=tanh(lambda psi),
// mu=sum(d^2 z)/sum(d^2),
// kappa=sqrt(1+alpha(z-mu)),
// theta <- theta-lr*kappa*d.
//
// The implementation intentionally does not materialize c, r, tau, z, d^2,
// or kappa as full tensors. It uses two coordinate passes after Navigation d.
void full_protect_vec(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> ds,
    std::vector<torch::Tensor> psis,
    double lr_d) {
  const float lr = static_cast<float>(lr_d);
  const float rho = 0.3f, one_minus_rho = 0.7f;
  const float eps = 1e-8f, lambda_gate = 3.0f, alpha = 0.375f;

  TORCH_CHECK(params.size() == grads.size());
  TORCH_CHECK(params.size() == ds.size());
  TORCH_CHECK(params.size() == psis.size());

  const Vec zero(0.0f), one(1.0f), neg_one(-1.0f), two(2.0f);
  const Vec v_eps(eps), v_lr(lr), v_rho(rho), v_omr(one_minus_rho);
  const Vec v_lambda(lambda_gate), v_alpha(alpha), quarter(0.25f);
  const int64_t width = Vec::size();

  Vec s0_vec(0.0f), s1_vec(0.0f);
  float s0_tail = 0.0f, s1_tail = 0.0f;

  // Pass 1: local trust/psi update + global sufficient statistics S0,S1.
  for (size_t j = 0; j < params.size(); ++j) {
    TORCH_CHECK(params[j].is_contiguous());
    TORCH_CHECK(grads[j].is_contiguous());
    TORCH_CHECK(ds[j].is_contiguous());
    TORCH_CHECK(psis[j].is_contiguous());

    float* psi = psis[j].data_ptr<float>();
    const float* p = params[j].data_ptr<float>();
    const float* g = grads[j].data_ptr<float>();
    const float* d = ds[j].data_ptr<float>();
    const int64_t n = ds[j].numel();

    int64_t i = 0;
    for (; i + width <= n; i += width) {
      const Vec pv = Vec::loadu(p + i);
      const Vec gv = Vec::loadu(g + i);
      const Vec dv = Vec::loadu(d + i);
      const Vec old_psi = Vec::loadu(psi + i);

      Vec coherence = at::vec::minimum(dv.abs(), one);
      coherence = coherence * coherence;

      const Vec prod = gv * dv;
      Vec direction = Vec::blendv(zero, one, prod > zero);
      direction = Vec::blendv(direction, neg_one, prod < zero);

      const Vec base_step = v_lr * dv.abs();
      const Vec exposure = base_step / (pv.abs() + two * base_step + v_eps);

      Vec trust = direction * coherence;
      trust = trust * (one - exposure);
      trust = trust - exposure;
      trust = at::vec::maximum(neg_one, at::vec::minimum(one, trust));
      trust = Vec::blendv(trust, zero, gv == zero);

      const Vec new_psi = old_psi * v_rho + trust * v_omr;
      new_psi.store(psi + i);

      const Vec w = dv * dv;
      const Vec z = (v_lambda * new_psi).tanh();
      s0_vec = s0_vec + w;
      s1_vec = s1_vec + w * z;
    }

    for (; i < n; ++i) {
      const float di = d[i];
      float coherence = std::min(std::fabs(di), 1.0f);
      coherence *= coherence;
      const float prod = g[i] * di;
      const float direction = (prod > 0.0f) - (prod < 0.0f);
      const float base_step = lr * std::fabs(di);
      const float exposure = base_step / (std::fabs(p[i]) + 2.0f * base_step + eps);
      float trust = direction * coherence * (1.0f - exposure) - exposure;
      trust = std::max(-1.0f, std::min(1.0f, trust));
      if (g[i] == 0.0f) trust = 0.0f;
      const float new_psi = psi[i] * rho + trust * one_minus_rho;
      psi[i] = new_psi;
      const float w = di * di;
      s0_tail += w;
      s1_tail += w * std::tanh(lambda_gate * new_psi);
    }
  }

  alignas(64) float lanes0[Vec::size()];
  alignas(64) float lanes1[Vec::size()];
  s0_vec.store(lanes0);
  s1_vec.store(lanes1);
  float s0 = s0_tail, s1 = s1_tail;
  for (int i = 0; i < Vec::size(); ++i) {
    s0 += lanes0[i];
    s1 += lanes1[i];
  }
  const float mu = s1 / std::max(s0, 1e-20f);
  const Vec v_mu(mu);

  // Pass 2: recompute z from persistent psi, form kappa transiently, and apply.
  for (size_t j = 0; j < params.size(); ++j) {
    float* p = params[j].data_ptr<float>();
    const float* d = ds[j].data_ptr<float>();
    const float* psi = psis[j].data_ptr<float>();
    const int64_t n = ds[j].numel();

    int64_t i = 0;
    for (; i + width <= n; i += width) {
      const Vec pv = Vec::loadu(p + i);
      const Vec dv = Vec::loadu(d + i);
      const Vec psiv = Vec::loadu(psi + i);
      const Vec z = (v_lambda * psiv).tanh();
      Vec kappa_sq = one + v_alpha * (z - v_mu);
      kappa_sq = at::vec::maximum(kappa_sq, quarter);
      const Vec out = pv - v_lr * (kappa_sq.sqrt() * dv);
      out.store(p + i);
    }

    for (; i < n; ++i) {
      const float z = std::tanh(lambda_gate * psi[i]);
      const float kappa_sq = std::max(0.25f, 1.0f + alpha * (z - mu));
      p[i] -= lr * (std::sqrt(kappa_sq) * d[i]);
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("full_protect_vec", &full_protect_vec);
}

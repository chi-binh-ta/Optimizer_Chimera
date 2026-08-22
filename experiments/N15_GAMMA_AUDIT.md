# N-1.5 — Should the preconditioning exponent gamma adapt?

## Frozen components

Protection is frozen at the P-3.5 candidate. Navigation candidates are

\[
d_t^{(\gamma)}=\frac{\hat m_t}{(\hat v_t+\epsilon)^\gamma},\qquad \gamma\in[0,1/2].
\]

Every candidate is L2 norm-matched to the current Adam candidate before the frozen LR-neutral protection gate:

\[
\tilde d_t^{(\gamma)}=d_t^{(\gamma)}\frac{\|d_t^{(1/2)}\|_2}{\|d_t^{(\gamma)}\|_2}.
\]

This prevents gamma from winning merely by changing the scalar step-size budget.

## Exact shadow oracle

The audit uses exact SPD quadratic objectives with condition numbers 10, 100, and 1000; diagonal and randomly rotated Hessians; and clean, Gaussian, heavy-tail, and stochastic sign-flip gradients. The baseline trajectory remains gamma=1/2. At each state, nine gamma candidates from 0 to 1/2 are evaluated exactly after the fixed P-3.5 protection transformation.

The one-step oracle is

\[
\gamma_t^\star\in\arg\min_{\gamma\in\{0,1/16,\ldots,1/2\}}F(\theta_t-\eta P_{3.5}(\tilde d_t^{(\gamma)})).
\]

## Main result

Oracle gamma is genuinely regime- and time-dependent. On held-out seeds the mean oracle gamma is about 0.219 and gamma=1/2 is the one-step oracle only about one third of observations. Oracle gamma changes between adjacent steps roughly one third of the time in this synthetic audit.

However, the best *fixed* gamma on tune seeds is still gamma=1/2, and the held-out fixed-gamma sweep improves monotonically up to gamma=1/2. The held-out mean post-step objective ratio is approximately 0.9837 for gamma=1/2 versus 0.9673 for the privileged one-step oracle, leaving real oracle headroom.

Cheap online observables were then tested as predictors of gamma*: dispersion of log(v), noise-ratio statistics, momentum-gradient coherence, sign agreement, ||m||/||g||, gradient coefficient of variation, v coefficient of variation, and reference-step exposure. Ridge, random forest, and histogram gradient boosting all failed the decision criterion: their selected gamma policies had worse held-out mean one-step objective than fixed gamma=1/2.

The strongest simple association was ||m_hat||/||g|| versus gamma* (Spearman magnitude about 0.35), which is not strong enough to identify a safe controller. A conservative shrinkage experiment showed tiny exploratory gains for very weak adaptation, but the adaptation strength chosen on a validation seed did not transfer to held-out seeds.

## Verdict

**Do not make gamma adaptive in Chimera yet.**

The audit distinguishes two statements:

1. `gamma*=1/2 universally` — **falsified**. A privileged oracle often prefers weaker preconditioning.
2. `gamma* is identifiable well enough from cheap online state to justify an adaptive controller` — **not supported / currently falsified by tested predictors**.

Therefore the engineering decision after N-1.5 is to keep

\[
\boxed{\gamma=1/2}
\]

as the fixed Navigation default. Oracle variability is retained as a future research target, not promoted into the optimizer.

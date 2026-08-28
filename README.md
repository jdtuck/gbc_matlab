# GBC-MATLAB — Generative Bayesian Computation surrogates

A MATLAB implementation of the Implicit Quantile Network surrogate from

> N. Polson and V. Sokolov (2026). *Generative Bayesian Computation as a
> Scalable Alternative to Gaussian Process Surrogates.* arXiv:2602.21408.

The idea in one paragraph: instead of fitting a Gaussian process to an
expensive simulator and paying O(n³) for the Cholesky factorisation while
being locked into a Gaussian predictive law, learn the **conditional quantile
function** Q_τ(Y | x) directly with a neural network that takes the quantile
level τ as an input. Sampling the posterior predictive at a new x* is then a
single forward pass per draw — the noise-outsourcing representation says that
pushing U ~ Uniform[0,1] through the learned quantile function reproduces the
conditional law exactly. Cost is linear in n, nothing is assumed stationary,
and discontinuous or multimodal responses need no custom kernel.

## Requirements

MATLAB R2020b or later with **Deep Learning Toolbox** (for `dlarray`,
`dlgradient`, `fullyconnect`, `adamupdate`). No Statistics Toolbox is needed —
the Gaussian CDF/quantile calls are written in terms of `erf`/`erfinv`.
Parallel Computing Toolbox is used automatically if a GPU is present, and
ignored otherwise.

## Quick start

```matlab
addpath(genpath('gbc-matlab'));

% Any (X, y) pairs from an expensive forward model.
X = rand(2000,10);
Y = 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + 10*X(:,4) + 5*X(:,5) ...
    + randn(2000,1);

model = gbcTrain(X, Y, gbcOptions('MaxEpochs',1500));

Q = gbcPredict(model, Xnew, [0.05 0.5 0.95]);   % quantiles
S = gbcSample(model, Xnew, 500);                % predictive draws
m = gbcMetrics(model, Xtest, Ytest);            % RMSE / CRPS / coverage
```

Then run the demos and the test suite:

```matlab
demo_friedman10     % 10-D Friedman, smooth-response weights
demo_jump2d         % 2-D bi-mixture GP with a jump, quantile-dominant weights
test_gbc            % fast verification checks
test_gbc(true)      % plus the end-to-end calibration test
```

## Files

| File | Role |
|---|---|
| `gbcOptions.m` | Hyper-parameters, with the paper's defaults |
| `gbcInit.m` | Glorot initialisation of the learnable parameters |
| `quantileEmbedding.m` | φ(τ) = [cos(jπτ)]ⱼ₌₀^{n_h−1}, n_h = 32 |
| `gbcForward.m` | G_φ(τ,x) = f_out( f₁( f_x(x) ⊙ f_τ(φ(τ)) ) ) |
| `gbcLoss.m` | The three-term composite loss, Eq. (1) |
| `gbcTrain.m` | Algorithm 1, training phase: Adam + cosine annealing |
| `gbcPredict.m` | Conditional quantiles on a τ grid |
| `gbcSample.m` | Algorithm 1, test phase: τ ~ U[0,1] → predictive draws |
| `gbcCRPS.m` | The paper's CRPS estimator, in O(M log M) |
| `gbcMetrics.m` | RMSE, CRPS, interval coverage and width, PIT values |
| `demos/demo_friedman10.m` | Friedman 10-D benchmark |
| `demos/demo_jump2d.m` | Bi-mixture GP jump benchmark (BGP, d = 2) |
| `tests/test_gbc.m` | Verification suite |

## What maps to what

**Architecture (Sec. 3).** The quantile level is embedded in a cosine basis,
φ(τ) = [cos(jπτ)] for j = 0…31, and passed through its own fully-connected
ReLU layer. The predictor x goes through a parallel layer. The two are merged
by an **elementwise product**, not a concatenation — this is what makes a
single set of weights represent the entire quantile curve rather than a
τ-indexed family of separate fits. A shared ReLU layer follows, then a linear
output with two heads: μ̂ (an L1 anchor used only during training) and q̂_τ
(the prediction). All three hidden layers are 256 wide.

**Loss (Eq. 1).**

```
ℓ(τ) = w₁·E|y − μ̂|                       location anchor, suppresses mode collapse
     + w₂·E[|τ − 0.5|·m_τ]                ordering surrogate, penalises crossings
     + w₃·E[max(τe, (τ−1)e)]              pinball loss, e = y − q̂_τ

m_τ = max(0, q̂_τ − y)  if τ < 0.5
      max(0, y − q̂_τ)  if τ ≥ 0.5
```

Weights default to (0.3, 0.3, 0.4) for smooth or heteroskedastic responses.
For jump processes the paper's **quantile-dominant** setting (0.1, 0.2, 0.7) is
reported to improve CRPS by roughly 28%; `demo_jump2d.m` uses it and has a
one-line switch so you can see the difference on your own data.

**Training (Algorithm 1).** One τ ~ Uniform[0,1] is drawn per training example
per mini-batch, so the network sees the whole quantile curve over the course of
training without ever materialising it. Adam at lr = 10⁻³ with single-cycle
cosine annealing. The paper uses 3,000–8,000 epochs; the demos use fewer to
keep runtimes reasonable and still converge on these problems.

**Prediction (Algorithm 1, test phase).** `gbcSample` draws τ^(b) ~ U[0,1] and
returns q̂_{τ^(b)}(x*) — one forward pass per draw, no linear algebra, which is
the O(n) test-time behaviour that motivates the method.

**CRPS.** The estimator in the paper,

```
CRPS(F,y) = (1/M)Σ|q_m − y| − (1/2M²)ΣΣ|q_m − q_m'|
```

is implemented via the exact identity, for q sorted ascending,
ΣΣ|q_i − q_j| = 2·Σ_i (2i − M − 1)·q_(i), so the double sum costs a sort rather
than M² operations. This is algebra, not an approximation, and it is checked
against brute force in the test suite.

## Choices I made where the paper is silent

These are documented so you can change them, not smuggled in:

- **Standardisation.** Inputs and response are z-scored during training and
  de-standardised on prediction (`Standardize`, default `true`). The paper does
  not specify a normalisation protocol beyond noting one benchmark was scaled
  to [0,1]. This matters more than it sounds: the cosine embedding has unit
  scale, so an unscaled y with a large range makes the multiplicative merge
  badly conditioned.
- **Mini-batch size** of 256, not stated in the paper.
- **Quantile rearrangement.** `gbcPredict` sorts each row's quantiles by
  default. The ordering surrogate discourages crossings but does not forbid
  them. Sorting is the Chernozhukov–Fernández-Galichon rearrangement, which
  weakly reduces the aggregate check-loss risk (verified numerically in the
  test suite) and leaves the CRPS estimator unchanged, since that estimator is
  permutation invariant. Pass `false` as the fourth argument to see the raw
  network output and inspect the crossing rate.
- **Glorot-uniform initialisation**, zero biases.

## What is not implemented

- **GBC-Aug**, the boundary-augmented variant (an MLP regime classifier whose
  output is appended as an extra input feature). Straightforward to add on top
  of this code: train a classifier on a regime label, then call `gbcTrain` on
  `[X, chat(X)]`.
- **GP baselines and the active-learning loops** (LGBB rocket, GRACE
  satellite). Nothing here compares GBC against a GP; the demos report GBC's
  own numbers, and `demo_friedman10.m` additionally prints the oracle CRPS —
  the closed-form score of the true N(f(x), σ²) predictive law — so you can see
  how much of the achievable gap is closed rather than relying on a
  reimplemented baseline.
- **Eleven of the fourteen benchmarks**, most of which need external datasets
  (the semiconductor fab AMHV data, the LGBB aerodynamic runs, GRACE drag
  coefficients).

## Verification

`test_gbc` checks the implementation against things outside it rather than
against itself:

- the cosine embedding against its definition, computed by an explicit loop;
- the CRPS pairwise-sum identity against brute-force double summation;
- the CRPS estimator against the **closed-form Gaussian CRPS**, and its error
  shrinking as the quantile grid refines;
- permutation invariance of the CRPS estimator;
- that rearranging a crossing quantile curve never raises its check-loss risk;
- that the pinball-loss minimiser is the empirical quantile, by grid search;
- the sign convention of the ordering surrogate, by forcing q̂ to a known
  constant and comparing against hand-computed values;
- **every gradient against central finite differences** in double precision;
- shape and τ-sensitivity of the forward pass;
- column-order preservation and monotonicity in `gbcPredict`;
- bit-reproducibility under a fixed seed;
- (slow test) end-to-end recovery of the **analytically known** conditional
  quantiles of a heteroskedastic Gaussian problem, plus 90% interval coverage
  landing inside [0.85, 0.95].

I could not run MATLAB in the environment where this was written, so the tests
have not been executed — run `test_gbc` first and tell me about any failure.
The mathematical identities the tests assert (the pairwise-sum identity, the
Gaussian CRPS convergence, the pinball minimiser, the rearrangement
inequality) were each verified numerically before being written into the
assertions, so a failure there points at the MATLAB code rather than at a bad
expectation.

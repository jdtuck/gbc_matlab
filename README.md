[![Pipeline Status](https://github.com/jdtuck/gbc_matlab/actions/workflows/matlab.yml/badge.svg)](https://github.com/jdtuck/gbc_matlab/actions/workflows/matlab.yml)

# GBC-MATLAB — Generative Bayesian Computation surrogates

A MATLAB implementation of the Implicit Quantile Network surrogate from

> N. Polson and V. Sokolov (2026). *Generative Bayesian Computation as a
> Scalable Alternative to Gaussian Process Surrogates.* arXiv:2602.21408
> (Technometrics).

Cross-checked against the authors' reproducibility package at
[github.com/VadimSokolov/gbc-surrogate](https://github.com/VadimSokolov/gbc-surrogate).
**Where the paper's text and the reference code disagree, this implementation
follows the code**, and every such case is listed below.

The idea in one paragraph: instead of fitting a Gaussian process to an
expensive simulator and paying O(n³) for the Cholesky factorisation while
being locked into a Gaussian predictive law, learn the **conditional quantile
function** Q_τ(Y | x) directly with a neural network that takes the quantile
level τ as an input. Sampling the posterior predictive at a new x* is then a
single forward pass per draw — the noise-outsourcing representation says that
pushing U ~ Uniform[0,1] through the learned quantile function reproduces the
conditional law exactly. Cost is linear in n, nothing is assumed stationary,
and discontinuous, multimodal or heteroskedastic responses need no custom
kernel and no separate variance model.

## Requirements

MATLAB R2020b or later with **Deep Learning Toolbox** (`dlarray`,
`dlgradient`, `fullyconnect`, `adamupdate`). No Statistics Toolbox is needed —
Gaussian CDF/quantile calls are written with `erf`/`erfinv`, and the empirical
quantile rule is implemented directly. Parallel Computing Toolbox is used
automatically if a GPU is present, and ignored otherwise.

## Quick start

```matlab
addpath(genpath('gbc-matlab'));

model = gbcTrain(X, Y);                          % (n-by-d, n-by-1)
Q = gbcPredict(model, Xnew, [0.05 0.5 0.95]);    % quantiles
S = gbcSample(model, Xnew, 500);                 % predictive draws
m = gbcMetrics(model, Xtest, Ytest);             % RMSE / CRPS / coverage

models = gbcEnsemble(X, Y, 5);                   % K = 5, as the paper uses
S = gbcPredict(models, Xnew, linspace(0.005,0.995,100));   % pooled samples
m = gbcMetricsFromSamples(S, Ytest);
```

Demos and tests:

```matlab
demo_motorcycle     % MASS::mcycle, the paper's Table 1 protocol
demo_friedman10     % 10-D Friedman, smooth-response weights
demo_jump2d         % 2-D bi-mixture GP with a jump, quantile-dominant weights
test_gbc            % fast verification checks
test_gbc(true)      % plus the end-to-end calibration test
```

## Files

| File | Role |
|---|---|
| `gbcOptions.m` | Hyper-parameters, with the reference's defaults |
| `gbcInit.m` | Xavier initialisation of the learnable parameters |
| `quantileEmbedding.m` | φ(τ) = [cos(jπτ)]ⱼ₌₀^{n_h−1}, n_h = 32 |
| `gbcForward.m` | The IQN forward pass |
| `gbcEvalTau.m` | Batched evaluation of q̂_τ(x) at explicit (point, τ) pairs |
| `gbcLoss.m` | The three-term composite loss, Eq. (1) |
| `gbcTrain.m` | Algorithm 1, training phase: Adam + cosine annealing |
| `gbcEnsemble.m` | K independent fits whose quantiles pool at test time |
| `gbcPredict.m` | Quantiles on a grid; pools columns for an ensemble |
| `gbcQuantile.m` | **Quantiles at the levels you asked for** — use this for plots and intervals |
| `gbcRowQuantile.m` | numpy-compatible empirical row quantile |
| `gbcSample.m` | Algorithm 1, test phase: τ ~ U[0,1] → predictive draws |
| `gbcModel.m` | The fitted model object, with the **reproducible** draws a calibration chain needs |
| `gbcCRPS.m` | CRPS, exact pairwise or the reference's permuted estimator |
| `gbcMetricsFromSamples.m` | RMSE / CRPS / coverage / width / PIT from samples |
| `gbcMetrics.m` | Convenience wrapper: predict, then score |
| `mcycleData.m` | MASS::mcycle, n = 133, verified against two sources |
| `demos/demo_motorcycle.m` | Table 1: heteroskedastic benchmark |
| `demos/demo_friedman10.m` | Friedman 10-D benchmark |
| `demos/demo_jump2d.m` | Bi-mixture GP jump benchmark (BGP, d = 2) |
| `tests/test_gbc.m` | Verification suite |

## Using the surrogate inside a calibration chain

`gbcTrain` returns a `gbcModel`. Handing it to an outer sampler (mvBayes,
Impala) needs one thing the functional interface does not provide: a surrogate
draw that **stays put** when you ask for it again.

```matlab
model = gbcTrain(X, Y, gbcOptions('NumSamples',2000));   % 2000 stored draws

i  = randi(model.nSamples);                  % latent surrogate index
yi = model.predict(Xstar, 'idxSamples', i);  % 1-by-size(Xstar,1), always the same
```

A GBC surrogate is implicit: `y = q̂_τ(x)` with `τ ~ U[0,1]`, so a single
uniform τ **is** one draw of the surrogate. The model therefore carries a fixed
set of `NumSamples` τ values, drawn once from a private `RandStream` seeded by
`model.sampleSeed`, and an index into that set names one realisation of the
quantile surface for the life of the object. That matters for two reasons:

- **Validity.** If `predict` drew fresh uniforms per call, a Metropolis ratio
  would compare two *different* surrogate realisations, and the chain would not
  target the posterior you wrote down. With a stored set the index is an
  ordinary latent variable: hold it, or give it its own Gibbs step.
- **Cost.** Asking for one draw evaluates the network once per test point, not
  `B` times and then discards `B-1` columns.

`Xstar` may have any number of rows. By default a draw **shares** its τ across
those rows, so the draw is a realisation of the quantile surface *as a
function* and a vector-valued prediction keeps its shape instead of picking up
independent noise at each point. Pass `'Shared', false` when the rows are
unrelated test points rather than one output vector and you want an
independent (still reproducible) level at each.

Also on the object: `predictMean` (deterministic, the τ-integral of the
quantile curve), `predictQuantile`, and `sample` — the last being fresh
i.i.d. draws off the global stream, for summarising the predictive law rather
than for use inside a chain.

## Architecture and loss

```
h   = f_1( f_x(x) ⊙ f_τ(φ(τ)) )        three FC+ReLU layers, width 256
out = f_out( tanh( f_2(h) ) )          f_2 is width 64
```

`f_out` has two heads: μ̂ (an L1 anchor used only in training) and q̂_τ (the
prediction). The merge is an **elementwise product**, not a concatenation —
that is what lets one set of weights represent the entire quantile curve
rather than a τ-indexed family of separate fits.

```
ℓ(τ) = w₁·E|y − μ̂|                    location anchor, suppresses mode collapse
     + w₂·E[|τ − 0.5|·m_τ]             ordering surrogate, penalises crossings
     + w₃·E[max(τe, (τ−1)e)]           pinball loss, e = y − q̂_τ

m_τ = max(0, q̂_τ − y)  if τ < 0.5
      max(0, y − q̂_τ)  if τ ≥ 0.5
```

Weights default to (0.3, 0.3, 0.4). For jump processes use the
**quantile-dominant** setting (0.1, 0.2, 0.7), which the paper reports
improves CRPS by ~28% there.

## Reconciling the paper with the reference code

These are the places where the published description and
`gbc/iqn.py` disagree. In each case the code wins, since the code is what
produced the tables.

| # | Paper says (or omits) | Reference code does | Here |
|---|---|---|---|
| 1 | `G(τ,x) = f_out(f₁(f_x ⊙ f_τ))` — three layers | An extra `Linear(256→64) + Tanh` before the head | Included; `BottleneckSize = 0` restores the paper's literal form |
| 2 | "a single draw τ ~ U[0,1] per training example" | **One scalar τ per gradient step**, shared across the whole batch | Default matches the code; `TauPerExample = true` for per-example |
| 3 | Mini-batching implied, size unstated | **Full batch**, one gradient step per "epoch" | `MiniBatchSize = Inf` by default |
| 4 | No weight decay mentioned | Adam `weight_decay = 1e-4` | `WeightDecay = 1e-4`, applied as torch does (coupled, all params incl. biases) |
| 5 | Cosine annealing, floor unstated | `eta_min = 0.01 × lr` | `MinLR = 0.01 × InitialLR` |
| 6 | Test phase: "draw τ ~ U[0,1]" | Deterministic grid `linspace(0.005, 0.995, 500)` | `gbcSample` draws; `gbcPredict` takes any grid — pass that one to match |
| 7 | CRPS = full double sum `(1/2M²)ΣΣ` | **Single random permutation**: `0.5·mean|q − q[perm]|` | Exact pairwise by default; `gbcCRPS(...,"permuted")` for the reference |
| 8 | Single model | Table 1 uses a **K = 5 ensemble**, 100 quantiles each | `gbcEnsemble` |
| 9 | Standardisation unspecified | z-score X and y, `std` normalised by N (numpy default) | Matches, including the N vs N−1 detail |

Item 1 is the one that changes results materially — a missing layer is a
different model. Items 2 and 3 together mean the reference does far fewer
gradient steps than "3000 epochs" suggests: 3000 steps total, each seeing one
τ. Item 7 is worth knowing if you compare numbers: the permuted estimator is
stochastic, so the paper's CRPS values carry a little noise that the exact
estimator here does not.

## Performance

The reference recipe is expensive by construction, and it is worth
understanding why before reaching for a smaller model. Training is full-batch
with **one gradient step per "epoch"**, and each step draws **one** τ shared
across the whole batch. So "5000 epochs" is 5000 steps that between them visit
only 5000 quantile levels. The paper's Table 1 multiplies that by 5 ensemble
members and 50 replicates: 1.25M steps, which the authors quote at ~30 minutes
in PyTorch. MATLAB's per-call `dlfeval` overhead makes it slower still, and on
a problem this small (n = 106, d = 1) that overhead, not arithmetic, is the
binding constraint.

### What the profiler found

`bench_gbc` measures where a step actually goes. On a 20-core machine with a
GPU, sweeping n (d = 4, hidden 256):

| n | CPU step | CPU BLAS | real headroom | GPU step | GPU + accel |
|---|---|---|---|---|---|
| 106 | 8.10 ms | 198 GFLOPS | 3.63× | 9.66 ms | 3.01 ms |
| 2 000 | 16.16 ms | 938 GFLOPS | 1.84× | 7.29 ms | 2.41 ms |
| 20 000 | 54.53 ms | 1635 GFLOPS | **1.01×** | 7.33 ms | 3.22 ms |

Three things fall out of this, and they decide every optimisation question
about this code.

**1. Hand-coding and MEX are dead ends at scale.** "Real headroom" is the
measured step against what plain-array MATLAB (no `dlarray`, no tape) reaches
for the same arithmetic. At n = 20 000 it is 1.01× — the framework costs
one percent. A MEX file would call the same BLAS and hit the same memory
bandwidth, so there is nothing there to win. Note the *idealised* floor, which
assumes every flop runs at the peak GEMM rate, reports 7.9× at the same point:
that metric would have sent someone off to write C for a 1% gain.

**2. The GPU is dispatch-bound, not compute-bound.** GPU step time is flat in
n — 7.29 ms at n = 2 000 and 7.33 ms at n = 20 000, ten times the arithmetic
in the same wall clock. The GPU is not working hard; it is waiting on per-kernel
launches and CPU-side tracing. That is why the GPU *loses* to the CPU at
n = 106 and only wins 7× at n = 20 000, well short of what the hardware can do.

**3. Which makes caching the trace the single biggest win.** `Accelerate`
(now the default) cuts the step 2.7× on CPU at n = 106 and 2.3× on GPU at
n = 20 000, computing exactly the same thing. Combined with the GPU it takes
n = 20 000 from 54.53 ms to 3.22 ms — **17×**, with no new code.

### Picking a configuration

The right answer differs by regime, and the crossover is sharp:

| workload | configuration | why |
|---|---|---|
| Small n, many fits (mcycle) | **CPU + `Accelerate` + `parfor`** | The step is overhead-bound (3.63× headroom, BLAS only 198 GFLOPS), so cores sit idle — replicates parallelise nearly free. The GPU is *slower* here. |
| Large n (Friedman, Michalewicz) | **GPU + `Accelerate`** | BLAS already saturates all 20 cores (1635 GFLOPS), so `parfor` would only split them. The GPU wins 7×, and more as n grows. |

Concretely, the motorcycle table (1.25M steps at n = 106) goes from **2.8 h**
to **1.4 h** with `Accelerate`, and to roughly **5–7 min** once replicates run
across 20 cores. Michalewicz at n = 90 000 projects to ~10 min per model on
CPU against ~30–45 s on GPU.

`demo_motorcycle` uses `parfor` with a worker cap, which runs as an ordinary
serial loop when Parallel Computing Toolbox is absent, so it is correct either
way. Rows print out of order when parallel.

### Reducing the work itself

- **`TauPerExample = true`** draws an independent τ for every training point,
  so a single step visits ~n levels instead of 1. On mcycle that is ~106×
  better quantile coverage per step, and far fewer steps resolve the same
  curve. This departs from the reference protocol — same model, same loss,
  different τ sampling — so use it for exploration and turn it off to match
  published numbers. `demo_motorcycle`'s `"fast"` preset uses it.
- **Batched prediction.** `gbcEvalTau` evaluates a whole block of quantile
  levels in one forward pass by tiling test points across levels, rather than
  looping. Block size is capped to bound activation memory. `gbcPredict`,
  `gbcSample` and `gbcModel/predict` all go through it, so they differ only in
  where τ comes from — a caller's grid, fresh uniforms, or the model's stored
  sample set.
- **Cheap training loop.** Loop invariants are hoisted (in full-batch mode the
  input `dlarray` is built once, not once per step), and the loss value and
  its three-term breakdown are only extracted on epochs that get recorded —
  each `extractdata` is a copy and, on GPU, a sync, which at one step per
  epoch would otherwise land on every step. `history` is therefore recorded at
  checkpoints only, with `history.epoch` holding the epochs it covers.

## Two traps around predictive intervals

Both of these bit this code, so they are worth stating plainly.

**`gbcPredict` vs `gbcQuantile`.** For a single model, `gbcPredict(model, X,
probs)` returns one column per level. For an **ensemble** it pools every
member's columns and sorts them, so the same call across K = 3 members returns
15 columns and column 5 is not the 95% level — it lands near the 25th
percentile. Plotting code that assumed the single-model layout drew a "90%
band" that actually spanned about 4% to 25% and missed most of the data.
**Use `gbcQuantile` whenever you want specific levels**; it returns one column
per level for both cases, querying the network directly for a single model and
taking the mixture quantile for an ensemble.

**Empirical quantiles of a quantile grid.** If a "sample" is really the
network evaluated on a grid spanning [a, b], then its empirical p-quantile
sits at level a + p·(b − a), not p. The reference's grid is
`linspace(0.005, 0.995, B)`, so `np.quantile(samples, 0.05)` lands at τ =
0.0545 and `0.95` at τ = 0.9455 — a **89.1% interval reported as 90%**, about
0.9 points of coverage. The distortion is affine, so raising B does not fix
it. `gbcMetricsFromSamples` takes an optional `tauSpan` argument that undoes
it exactly; `gbcMetrics` passes it automatically. Omit it to reproduce the
reference's numbers as published.

## Choices that are mine, not the paper's or the code's

- **Quantile rearrangement.** `gbcPredict` sorts each row's quantiles by
  default. Neither the paper nor the reference does this; both rely on the
  ordering surrogate, which discourages crossings without forbidding them.
  Sorting is the Chernozhukov–Fernández-Galichon rearrangement, which weakly
  reduces the aggregate check-loss risk (verified in the test suite) and
  leaves the CRPS estimator unchanged, since that estimator is permutation
  invariant. Pass `false` as the fourth argument to inspect raw output and
  measure the crossing rate.
- **Mini-batching** is available but off; the reference is full-batch only.

## What is not implemented

- **GBC-Aug**, the boundary-augmented variant. Per the reference README, it is
  a preprocessing step: EM-cluster y into two components, train an MLP
  classifier x → P(regime | x), append that probability as an extra input
  feature, then train a standard IQN. Buildable on top of this code without
  touching it — `gbcTrain([X, phat], Y)`.
- **GP baselines** (hetGP, MJGP, deepgp) and the active-learning loops. The
  paper's comparisons run those in R. Nothing here reimplements them, so the
  demos report GBC's own numbers only; `demo_friedman10.m` prints the oracle
  CRPS of the true N(f(x), σ²) law as a reference floor instead of a
  reimplemented competitor.
- **Ten of the fourteen benchmarks**, which need external data (the fab AMHV
  runs, LGBB aerodynamics, GRACE drag, and the Flowers et al. jumpgp CSVs).

## Verification

`test_gbc` checks the implementation against things outside it: the cosine
embedding against its definition computed by an explicit loop; the CRPS
pairwise identity against brute-force double summation; the CRPS estimator
against the **closed-form Gaussian CRPS**; permutation invariance; the
rearrangement inequality; the pinball minimiser against the empirical
quantile by grid search; the ordering surrogate's sign convention against
hand-computed values; **every gradient against central finite differences** in
double precision; the presence and removability of the bottleneck layer; the
permuted CRPS estimator against the exact one; the empirical-quantile rule
against numpy's linear interpolation; ensemble pooling shape and member
distinctness; column-order preservation; bit-reproducibility under a fixed
seed; and (slow test) end-to-end recovery of the **analytically known**
conditional quantiles of a heteroskedastic Gaussian problem.

The gradient check deserves a note, because it bit twice during development.
The loss is piecewise linear, so central differences are exact away from a
kink and meaningless on one. `gbcInit` sets biases to zero, and the
multiplicative merge means any example whose first-layer pre-activations are
all negative produces an exactly-zero hidden vector and so an exactly-zero
pre-activation downstream — sitting precisely on a ReLU kink. The check
therefore draws its probe point with nonzero biases and refuses to run until
every kink is at least 100× the step size away.


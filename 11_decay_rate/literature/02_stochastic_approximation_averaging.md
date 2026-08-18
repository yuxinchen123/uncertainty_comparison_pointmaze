# Stochastic approximation and least-squares theory for a $n^{-1/2}$ residual

Notes on learning-rate schedules, Polyak–Ruppert and tail averaging, and AdaGrad, read against the
concrete question: can the per-position residual of a random-network-distillation predictor be made
to decay as $n^{-1/2}$ at *every* position at once, and as $n_i^{-1/2}$ when position $i$ is visited
$n_i$ times.

## 0. Notation used throughout

Every symbol below is defined here and reused unchanged.

- $N$ is the number of maze positions, about 100 in our setting. Position $i$ has input
  $x_i \in \mathbb{R}^4$ (the pair $(x,y)$ padded with two zeros).
- $f$ is the frozen randomly initialised target network, $g_n$ the predictor after $n$ optimizer
  steps, $m = 128$ the output width.
- The bonus at position $i$ is $b_i(n) = \|g_n(x_i) - f(x_i)\|_2$. The stacked residual is
  $r(n) \in \mathbb{R}^{N \times m}$ with rows $g_n(x_i) - f(x_i)$.
- $w_n$ is the predictor's weight vector, $w_0$ its initialisation, $P$ the number of weights.
- $J(x) \in \mathbb{R}^{m \times P}$ is the Jacobian of the predictor's output at $x$ with respect to
  its weights. $J$ without an argument is the stacked Jacobian over all $N$ positions.
- $K = \tfrac{1}{N} J J^{\top}$ is the tangent-kernel Gram matrix on the $N$ positions, with
  eigenvalues $\lambda_1 \ge \dots \ge \lambda_N > 0$ and eigenvectors $v_1, \dots, v_N$. In the
  linearised (tangent-kernel) regime the residual obeys a linear recursion driven by $K$.
- $H$ is the second-moment matrix of the tangent features in weight space, $H = \sum_i p_i \phi_i
  \phi_i^{\top}$, where $\phi_i$ is position $i$'s tangent feature vector and $p_i$ is the
  probability that a sampling step picks position $i$. $K$ and $H$ have the same non-zero spectrum.
- $\eta$ is a constant step size, $\eta_n$ a step-size sequence, $\gamma$ the same thing in the
  papers' notation.
- $\sigma^2$ is the variance of the noise in the regression target (zero in the current
  deterministic setup; several methods below deliberately make it non-zero).
- $n_i$ is the number of times position $i$ has been visited. Under full-batch training
  $n_i = n$ for all $i$; under sampling with probability $p_i$, $\mathbb{E}[n_i] = p_i n$.
- $\bar w_n = \tfrac{1}{n}\sum_{k<n} w_k$ is the Polyak–Ruppert average; $w_{t:T} =
  \tfrac{1}{T-t}\sum_{\tau=t}^{T-1} w_\tau$ is the tail average that discards the first $t$ iterates.

**Which quantity decays at which rate.** Papers in this area almost always bound the *excess risk*
$L(w) - L(w^{*}) = \tfrac{1}{2}\|w - w^{*}\|_H^2 = \tfrac{1}{2}\,\mathbb{E}_x[(\langle w - w^{*},
\phi(x)\rangle)^2]$, i.e. half the mean *squared* prediction residual. A statement "the risk is
$O(1/n)$" therefore means the *root mean square residual* is $O(n^{-1/2})$, which is the quantity
our bonus is. Every rate below is labelled with the quantity it applies to. Confusing the two costs
a factor of two in the log-log slope, which is exactly the gap between the $-1$ and the $-0.5$ that
this project is trying to hit.

## 1. The obstruction that no step-size schedule can remove

This is a short derivation, not a citation, and it explains the heterogeneity already measured with
the $1/t$ schedule.

In the linearised regime, full-batch gradient descent with step sizes $\eta_1, \eta_2, \dots$ gives

$$ r(n) = \prod_{m \le n} (I - \eta_m K)\, r(0), \qquad\text{so mode } k \text{ carries the factor }
\prod_{m \le n}(1 - \eta_m \lambda_k). $$

For $\eta_m \lambda_k \ll 1$ the logarithm of that factor is $-\lambda_k \sum_{m\le n} \eta_m +
O(\sum_m \eta_m^2\lambda_k^2)$. Write $S_n = \sum_{m \le n}\eta_m$. Then

$$ \log(\text{mode } k \text{ factor}) \approx -\lambda_k S_n . $$

Two consequences:

1. **Every mode's log-decay is the same function of $n$, namely $S_n$, scaled by that mode's own
   eigenvalue.** The ratio of the log-decays of modes $k$ and $j$ is $\lambda_k / \lambda_j$ *for any
   schedule whatsoever*. No global learning-rate schedule can make two modes with different
   eigenvalues decay at the same relative rate, because the schedule enters every mode through the
   same scalar $S_n$.
2. A schedule $\eta_m = c/m$ gives $S_n \approx c\log n$, hence mode $k$ decays as $n^{-c\lambda_k}$
   — a power law whose *exponent* is proportional to the eigenvalue. This is precisely why the
   aggregate slope can be tuned to $-0.503$ while individual positions visibly disagree: only the
   modes with $c\lambda_k \approx 1/2$ sit at the target slope, faster modes overshoot and slower
   modes undershoot, and each position is a different mixture of modes.

So a uniform $n^{-1/2}$ requires leaving the "one global step size on the raw gradient" family. The
three exits found in the literature are:

1. **Change the eigenvalues** — precondition so that all modes have the same effective curvature,
   then a $c/n$ schedule with $c = 1/2$ hits $n^{-1/2}$ in every mode at once.
2. **Put a noise floor under the residual** — with gradient or target noise, the averaged iterate's
   error settles at a statistical floor whose size decays as $n^{-1/2}$ *in every direction* with a
   rate that does not depend on the eigenvalues at all. This is the Polyak–Ruppert route and it is
   the only route in this literature that produces $n^{-1/2}$ rather than $n^{-1}$.
3. **Give each position its own step** — then position $i$'s exponent can be tied to $n_i$ directly.
   Nothing in the classical stochastic-approximation literature does this; it needs a per-position
   update, which the least-squares structure makes easy to construct (Section 7).

**How ill-conditioned is our problem in practice.** For the actual architecture (4 → 256 → ReLU →
128) on a 10-by-10 grid of positions, the tangent-kernel Gram matrix has largest eigenvalue
$1.7\times10^{5}$, median $2.0$, smallest $0.31$ — a condition number of $5.6\times10^{5}$
(computed here, Section 8). Any statement of the form "all modes behave the same once
$n \gg 1/(\eta\lambda_{\min})$" therefore only starts to hold after a very large number of steps.

## 2. Question 1 — classical results giving $O(1/n)$ risk, hence $n^{-1/2}$ residual

### 2.1 Polyak and Juditsky (1992); Ruppert (1988)

- **What they show.** For a Robbins–Monro recursion with step sizes decaying slower than $1/n$
  (specifically $\eta_n = c n^{-\alpha}$ with $\alpha \in (1/2, 1)$), the *averaged* iterate
  $\bar w_n$ satisfies a central limit theorem
  $\sqrt{n}\,(\bar w_n - w^{*}) \Rightarrow \mathcal{N}(0,\; H^{-1}\Sigma H^{-1})$,
  where $H$ is the Hessian (here the feature second moment) and $\Sigma$ is the covariance of the
  gradient noise at the optimum. The asymptotic covariance is the smallest achievable by any
  estimator (it matches the Cramér–Rao bound), and it does not depend on the step-size constant $c$
  or the exponent $\alpha$.
- **What it means for a residual.** The squared parameter error decays as $1/n$, so the parameter
  error norm — and hence the prediction residual at any fixed input — decays as $n^{-1/2}$. Crucially
  the $1/n$ applies to the whole covariance matrix, so **every direction decays at the same rate**;
  only the constants differ, and they are given by the entries of $H^{-1}\Sigma H^{-1}$.
- **The catch for us.** $\Sigma = 0$ when there is no gradient noise, and then the theorem says
  nothing (the limit is degenerate and the iterate converges faster). The $n^{-1/2}$ is a
  *statistical* rate, not an optimization rate.
- **Where it applies.** This is the single most relevant classical result: it is the only mechanism
  in the whole literature that yields a rate uniform across directions and equal to $n^{-1/2}$ in the
  residual.

### 2.2 Bach and Moulines, "Non-asymptotic analysis of stochastic approximation algorithms for machine learning" (NeurIPS 2011)

- **Theorem 1 (strongly convex, step size $\gamma_n = C n^{-\alpha}$), quoted.** With
  $\delta_n = \mathbb{E}\|w_n - w^{*}\|^2$, strong convexity constant $\mu$, smoothness $L$, noise
  level $\sigma$:
  - for $0 \le \alpha < 1$: $\delta_n \le 2\exp(4L^2C^2\phi_{1-2\alpha}(n))\exp(-\tfrac{\mu C}{4}
    n^{1-\alpha})(\delta_0 + \tfrac{\sigma^2}{L^2}) + \tfrac{4C\sigma^2}{\mu n^{\alpha}}$;
  - for $\alpha = 1$: $\delta_n \le \dfrac{\exp(2L^2C^2)}{n^{\mu C}}\Big(\delta_0 +
    \dfrac{\sigma^2}{L^2}\Big) + 2\sigma^2C^2\dfrac{\phi_{\mu C/2 - 1}(n)}{n^{\mu C/2}}$.
- **The $1/n$ schedule is exactly our problem, stated by the authors.** Their discussion of the
  $\alpha = 1$ case: "the choice of $C$ is critical … too small $C$ leads to convergence at
  arbitrarily small rate of the form $n^{-\mu C/2}$, while too large $C$ leads to explosion due to
  the initial condition." Read per eigenmode ($\mu \to \lambda_k$), this *is* Section 1's
  observation, in print: with a $c/n$ schedule the initial-condition term of mode $k$ decays as
  $n^{-c\lambda_k}$ in squared error, i.e. $n^{-c\lambda_k/2}$ in the residual. They also note the
  recursion is an equality for quadratic objectives, so the bound is tight up to constants for
  least squares.
- **Theorem 3 (averaging).** For $\alpha \in (0,1)$,
  $(\mathbb{E}\|\bar w_n - w^{*}\|^2)^{1/2} \le
  \dfrac{[\operatorname{tr}(f''(w^{*})^{-1}\Sigma f''(w^{*})^{-1})]^{1/2}}{\sqrt{n}} + \text{lower-order
  terms}$, with the leading term independent of the step-size sequence. This is the non-asymptotic
  version of Polyak–Juditsky: **root mean square parameter error $= n^{-1/2}$, uniform in direction,
  with the constant $\operatorname{tr}(H^{-1}\Sigma H^{-1})$.** They also state that for $\alpha=1$
  averaging does not help and does not fix the sensitivity to $C$.
- **Initial conditions under averaging.** They flag that averaging forgets the initial condition only
  at rate $O(n^{-2})$ in squared error, not geometrically, and note the standard remedy of starting
  the average after a burn-in — which is the tail averaging of Section 2.5.

### 2.3 Bach and Moulines, "Non-strongly-convex smooth stochastic approximation with convergence rate $O(1/n)$" (NeurIPS 2013)

- **Setting.** Least squares (the recursion is called least-mean-squares), constant step size,
  Polyak–Ruppert averaging, no strong convexity assumed, dimension $d$, features bounded so that
  $\mathbb{E}[\|x\|^2 x x^{\top}] \preceq R^2 H$ and noise $\mathbb{E}[\xi\xi^{\top}] \preceq
  \sigma^2 H$.
- **Theorem 1, quoted.** For any constant step size $\gamma < 1/R^2$,
  $$ \mathbb{E}\big[f(\bar w_{n-1}) - f(w^{*})\big] \;\le\; \frac{1}{2n}\left(\frac{\sigma\sqrt{d}}
  {1 - \sqrt{\gamma R^2}} + \frac{R\|w_0 - w^{*}\|}{\sqrt{\gamma R^2}}\right)^{2}, $$
  and with $\gamma = 1/(4R^2)$ this is $\le \dfrac{2}{n}\big[\sigma\sqrt{d} + R\|w_0-w^{*}\|\big]^2$.
- **What this says about our quantity.** $f(\bar w) - f(w^{*})$ is half the mean squared prediction
  residual, so the bound gives root-mean-square residual $\le \sqrt{2}\,[\sigma\sqrt d + R\|w_0 -
  w^{*}\|]\, n^{-1/2}$. This is an $n^{-1/2}$ residual with a *constant step size* and averaging — no
  schedule tuning at all, and no dependence on the smallest eigenvalue of $H$. That last property is
  what makes it attractive for our badly conditioned tangent kernel.
- **Important caveat.** The bound's initial-condition term also carries $1/n$, but that part is
  **not tight**: it is an upper bound. In the noiseless case ($\sigma = 0$) the true rate of the
  initial-condition term is faster; see Section 2.4 and Section 3. So one cannot read this theorem as
  "even without noise, averaged constant-step gradient descent gives a residual $n^{-1/2}$".

### 2.4 Défossez and Bach, "Constant step size least-mean-square: bias-variance trade-offs and optimal sampling distributions" (AISTATS 2015)

- **What they show.** An asymptotic expansion of the excess risk of averaged constant-step-size
  least-mean-squares that separates the two terms:
  - a **variance term decaying as $O(\sigma^2 d/n)$, independent of the step size $\gamma$**;
  - a **bias (initial-condition) term decaying as $O(\|w_0-w^{*}\|^2/(\gamma^2 n^2))$.**
- **Why this matters for us more than anything else in Section 2.** In our current deterministic
  setup $\sigma = 0$, so the variance term vanishes and the bias term is the whole story: the excess
  risk decays as $n^{-2}$ and the **residual as $n^{-1}$**, not $n^{-1/2}$. The measured slope of
  $-1$ for averaged full-batch gradient descent (Section 3) is exactly this term. To get $n^{-1/2}$
  one must make the variance term non-zero, i.e. introduce noise.
- They also study non-uniform sampling of the data and show the optimal sampling density differs
  depending on which of the two terms dominates — relevant background for the non-uniform-visitation
  extension, though their objective is minimising the risk, not shaping per-position rates.

### 2.5 Jain, Kakade, Kidambi, Netrapalli, Pillutla, Sidford, "A Markov chain theory approach to characterizing the minimax optimality of stochastic gradient descent (for least squares)" (FSTTCS 2017)

- **Theorem 1, quoted.** For step size $\gamma < 1/R^2$ and the tail average $w_{t:T}$,
  $$ \mathbb{E}[L(w_{t:T})] - L(w^{*}) \;\le\; \left(\sqrt{\tfrac{1}{2}\exp(-\gamma\mu t)\,
  R^2\|w_0 - w^{*}\|^2} \;+\; \sqrt{\Big(1 + \tfrac{\gamma R^2}{1-\gamma R^2}\rho_{\text{misspec}}\Big)
  \tfrac{\sigma^2_{\text{MLE}}}{T-t}}\right)^{2}. $$
- **Reading it.** The bias term decays **geometrically in the burn-in length $t$**, and the variance
  term is $\sigma^2_{\text{MLE}}/(T-t)$. With $\gamma = 1/(2R^2)$ and a well-specified model the
  variance term in root form is $2\sigma_{\text{MLE}}/\sqrt{T-t}$.
- **Why tail averaging is the right averaging for us.** Full Polyak–Ruppert averaging drags the
  initial condition along at rate $1/n$ (Section 2.2), which pollutes the early part of the log-log
  plot and bends the measured slope. Tail averaging removes it geometrically, leaving a residual that
  is *purely* the $\sigma_{\text{MLE}}/\sqrt{T-t}$ term — a clean straight line of slope $-1/2$ over
  the whole measured range. Practically: keep several running tail averages (starting at $T/2$, $T/4$,
  …) so the burn-in does not have to be known in advance, as the authors suggest.
- $\sigma^2_{\text{MLE}} = \tfrac{1}{2}\mathbb{E}[(y - w^{*}\!\cdot x)^2\|x\|^2_{H^{-1}}]$ — this is
  the quantity that sets the constant in front of the $n^{-1/2}$, and it is a leverage-weighted noise
  level. It reappears position by position in Section 6.

### 2.6 Jain, Kakade, Kidambi, Netrapalli, Sidford, "Parallelizing stochastic gradient descent for least squares regression: mini-batching, averaging, and model misspecification" (JMLR 18(223), 2018)

- **What it adds.** Tight non-asymptotic excess-risk bounds for mini-batching and tail averaging
  together, and the precise extent to which mini-batching parallelises. For our purposes the relevant
  content is: mini-batching reduces the variance term by the batch size until a problem-dependent
  threshold, after which extra batch size buys nothing — i.e. a full-batch gradient is *not*
  statistically better than a well-chosen mini-batch, it just removes the noise that generates the
  $n^{-1/2}$ floor.
- **Misspecification.** The maximal step size that still achieves the minimax risk must depend on the
  noise properties when the model is misspecified. Worth remembering if we generate the noise floor
  by underfitting (a predictor too small to interpolate) rather than by injecting noise.

### 2.7 Dieuleveut, Flammarion, Bach, "Harder, better, faster, stronger convergence rates for least-squares regression" (JMLR 18(101), 2017)

- **What they show.** An averaged *accelerated* regularised gradient method for least squares that
  simultaneously forgets the initial condition at $O(1/n^2)$ (excess risk) and attains the optimal
  noise- and dimension-dependent term. Prior methods got one or the other.
- **Relevance.** It is the sharpest statement of the two-term structure: initial condition $1/n^2$,
  noise $\sigma^2 d/n$, in excess risk. In residual terms: $n^{-1}$ from the initial condition,
  $n^{-1/2}$ from the noise. Both slopes are visible in a single log-log plot of our bonus if noise
  is present — a knee where the curve bends from $-1$ to $-1/2$. Predicting and then observing that
  knee is a good sanity check that the mechanism is the one we think it is.

### 2.8 Zou, Wu, Braverman, Gu, Kakade, "Benign overfitting of constant-stepsize SGD for linear regression" (COLT 2021; JMLR 24, 2023)

- **What they show.** Sharp instance-dependent excess-risk bounds for constant-step-size SGD with
  iterate or tail averaging in the overparameterised regime, stated through the full eigenspectrum of
  the feature covariance: the variance term is governed by an effective dimension specific to SGD,
  and the bias term by how the initialisation aligns with the covariance.
- **Relevance.** Our problem is overparameterised (34,176 weights, 100 positions), so the classical
  $\sigma^2 d/n$ with $d$ = parameter count is meaningless; the correct constant is an effective
  dimension. This paper is the reference for what replaces $d$.

### 2.9 Ge, Kakade, Kidambi, Netrapalli, "The step decay schedule: a near optimal, geometrically decaying learning rate procedure for least squares" (NeurIPS 2019)

- **What they show.** The *final iterate* of SGD under any polynomially decaying learning rate
  (including $1/t$) is highly sub-optimal relative to the statistical minimax rate — by a condition
  number factor in the strongly convex case. A geometrically decaying ("step decay") schedule is
  near-optimal.
- **Relevance, and a warning.** This says the $1/t$ schedule is a bad way to reach the statistical
  floor, and the reason is exactly the condition-number sensitivity of Section 1. It does *not* say
  step decay would help us: a geometric schedule makes $S_n$ bounded, so each mode's residual settles
  at a constant rather than following any power law. Step decay is the right answer for "reach the
  optimum fast" and the wrong answer for "follow a prescribed power law".

### 2.10 Neu and Rosasco, "Iterate averaging as regularization for stochastic gradient descent" (COLT 2018)

- **What they show.** A geometrically weighted average of the iterates has the same regularising
  effect as, and is asymptotically equivalent to, ridge regression for linear least squares; they
  give finite-sample bounds matching the best known for regularised stochastic gradient methods.
- **Relevance.** It supplies the interpretation of the averaged iterate as applying a ridge-type
  spectral filter with regularisation strength about $1/(\eta n)$, which is exactly the shape of the
  averaged-gradient-descent filter derived in Section 3. It is the cleanest way to see *why* the
  averaged iterate's per-mode residual is $1/(n\eta\lambda_k)$ rather than geometric.

## 3. Question 2 — what survives when there is no gradient noise

This is our base setting: full-batch gradient descent, deterministic, every position visited every
step.

### 3.1 The per-eigenmode formula for the averaged iterate — confirmed

The formula proposed in the question is correct. Derivation: with a constant step $\eta$ on a
quadratic, mode $k$ of the residual is $r_k(n) = (1-\eta\lambda_k)^n r_k(0)$, so the average over the
first $n$ iterates is a geometric sum,

$$ \bar r_k(n) \;=\; \frac{1}{n}\sum_{m=0}^{n-1}(1-\eta\lambda_k)^m\, r_k(0)
\;=\; \frac{1 - (1-\eta\lambda_k)^n}{n\,\eta\,\lambda_k}\; r_k(0). $$

Three regimes per mode:

1. $n \ll 1/(\eta\lambda_k)$: the factor is $\approx 1$ — the mode has not started to move and the
   average tracks the initialisation. Log-log slope $0$.
2. $n \approx 1/(\eta\lambda_k)$: the transition.
3. $n \gg 1/(\eta\lambda_k)$: the factor is $\approx \dfrac{1}{n\eta\lambda_k}$ — **slope exactly
   $-1$, identical for every mode**, with a mode-dependent constant $\dfrac{r_k(0)}{\eta\lambda_k}$.

Numerical check (Section 8): with a 100-position linear model, tangent-kernel condition number 10,
horizon $10^7$, the per-position log-log slope of the averaged iterate's residual was
$-1.0000$ for all 100 positions, spread $0.0000$; the per-position *levels* differed by a factor of
about 119. With condition number 100 and a horizon of $10^5$ the slowest modes were still in regime 1
and the per-position slopes ranged from $+0.49$ to $-1.39$ — i.e. **the uniformity is real but it only
appears after $n \gg 1/(\eta\lambda_{\min})$**, and our measured tangent-kernel condition number is
$5.6\times10^{5}$.

So: averaged full-batch gradient descent solves the *uniformity* half of the problem exactly, and
gets the *exponent* wrong by a factor of two ($-1$ instead of $-1/2$).

### 3.2 The filter-function view, and the inverse-problems literature

Writing $r(n) = \psi_n(K)\,r(0)$ makes the schedule question a question about which spectral filter
$\psi_n$ one is applying:

- plain gradient descent: $\psi_n(\lambda) = (1-\eta\lambda)^n$ — the Landweber iteration of linear
  inverse problems;
- averaged gradient descent: $\psi_n(\lambda) = \dfrac{1-(1-\eta\lambda)^n}{n\eta\lambda}$, which for
  $\lambda \gg 1/(\eta n)$ behaves like $\dfrac{1}{n\eta\lambda}$ — the ridge filter
  $\dfrac{\alpha}{\lambda+\alpha}$ with $\alpha = 1/(\eta n)$, matching Neu and Rosasco;
- a $c/m$ schedule: $\psi_n(\lambda) \approx n^{-c\lambda}$, which is not a standard regularisation
  filter and has the pathological property that its *exponent* depends on $\lambda$.

**Yao, Rosasco, Caponnetto, "On early stopping in gradient descent learning" (Constructive
Approximation 26:289–315, 2007)** is the reference that sets this up: gradient descent with
polynomially decaying step sizes on the least-squares risk in a reproducing-kernel Hilbert space is a
finite-rank Monte-Carlo approximation of the Landweber iteration, and the number of iterations plays
the role of the regularisation parameter, with early stopping chosen by a bias-variance trade-off.
The value of this framing for us is that any filter $\psi$ we want can be *designed* — the question
"which optimizer gives $b_i(n) \propto n^{-1/2}$" becomes "which filter has
$\psi_n(\lambda) \approx n^{-1/2}$ for all $\lambda$ in the spectrum", and the answer is that no
$\lambda$-independent filter built from a single global step size does, which is Section 1 again.

### 3.3 Noiseless and interpolating regimes, explicitly

- **Berthier, Bach, Gaillard, "Tight nonparametric convergence rates for stochastic gradient descent
  under the noiseless linear model" (NeurIPS 2020, arXiv 2006.08212).** Single-pass constant-step
  SGD on the least-squares risk when the model is exactly realisable, so the noise at the optimum is
  zero. They give tight rates for the iterates and the generalisation error that depend on the decay
  of the spectrum of the covariance and on the regularity of the optimum. The message for us: in the
  noiseless case the rate is *entirely spectrum-determined* — there is no universal $n^{-1/2}$ floor,
  and the achievable rate is whatever the eigenvalue decay allows. This is the formal version of "a
  noiseless problem has no statistical rate to fall back on".
- **Varre, Pillaud-Vivien, Flammarion, "Last iterate convergence of SGD for least-squares in the
  interpolation regime" (NeurIPS 2021, arXiv 2102.03183).** The noiseless least-squares model with
  constant step size, analysing the *last* iterate rather than an average — showing an explicit
  convergence result for a non-strongly-convex problem where the usual results need averaging. Again
  the rates are spectrum-dependent and there is no $n^{-1/2}$ floor.
- **Flammarion and Bach, "From averaging to acceleration, there is only a step-size" (COLT 2015,
  arXiv 1504.01577).** Averaged gradient descent, accelerated gradient descent and heavy ball on
  quadratic non-strongly-convex problems are all the same two-term recursion with different
  constants, and stability of that recursion is equivalent to convergence of the function value at
  rate $O(1/n^2)$ — the residual at $O(1/n)$. This is the clean citation for "averaging a noiseless
  quadratic buys you $n^{-1}$ in the residual", i.e. the result of Section 3.1 in the literature.

### 3.4 Answer to question 2 in one line

Of the classical results, only the *variance* halves survive with a meaning for us, and they are
identically zero when there is no noise. What survives is the *bias* half: averaged full-batch
gradient descent on a quadratic gives residual $\dfrac{r_k(0)}{n\eta\lambda_k}$ per mode, i.e. slope
$-1$ uniformly across modes and positions once past the transient. To reach $-1/2$ one must either
halve the exponent by a readout change, or reintroduce noise so that the Polyak–Ruppert variance
term becomes the leading term.

## 4. Question 3 — AdaGrad

### 4.1 Duchi, Hazan, Singer (JMLR 12:2121–2159, 2011)

- **What they prove.** Regret bounds for online convex optimization with the accumulator
  $G_n = \sum_{m\le n} g_m g_m^{\top}$: the diagonal version's regret is controlled by
  $\sum_j \|g_{1:T,j}\|_2$ and the full-matrix version's by $\operatorname{tr}(G_T^{1/2})$, in both
  cases as good as the best fixed preconditioner chosen in hindsight. Converting regret to
  stochastic optimization gives excess risk $O(1/\sqrt{T})$ for the averaged iterate on general
  convex Lipschitz problems.
- **What that does and does not say about least squares.** $O(1/\sqrt{T})$ in *excess risk* means
  $T^{-1/4}$ in the residual — far slower than what least squares actually achieves, because the
  regret bound is worst-case over convex Lipschitz functions and ignores the quadratic structure. It
  is not a useful prediction for our setting.

### 4.2 Ward, Wu, Bottou, "AdaGrad stepsizes: sharp convergence over nonconvex landscapes" (ICML 2019; JMLR 21, 2020)

- **What they prove.** AdaGrad-Norm (a single scalar accumulator) converges to a stationary point at
  $O(\log N/\sqrt{N})$ in the squared gradient norm in the stochastic setting, and at the optimal
  $O(1/N)$ in the batch (deterministic) setting, without knowing the smoothness constant or the noise
  level.
- **Relevance.** The batch case is our case, and $O(1/N)$ in squared gradient norm is a statement
  about the *aggregate*, not per direction, and it is an upper bound that does not force a power law.

### 4.3 Xie, Wu, Ward, "Linear convergence of adaptive stochastic gradient descent" (AISTATS 2020)

- **What they prove.** AdaGrad-Norm converges *linearly* (geometrically) for strongly convex
  functions and for functions satisfying the Polyak–Łojasiewicz inequality, under a condition they
  call the restricted uniform inequality of gradients, in a two-stage argument.
- **Relevance.** This is the formal statement of the phenomenon that kills AdaGrad for our purpose:
  once the gradients get small, the accumulator stops growing and the effective step size stops
  shrinking, so the decay becomes geometric rather than a power law.

### 4.4 What AdaGrad does to a quadratic, worked out

This is a derivation, checked numerically in Section 8.

Take a quadratic with Hessian eigenvalues $\lambda_k$ and error components $e_k$. The gradient
component is $g_k = \lambda_k e_k$. Full-matrix AdaGrad (equivalently: diagonal AdaGrad written in the
Hessian's eigenbasis, since on a quadratic the accumulator's eigenbasis is the Hessian's) updates

$$ e_k(n+1) \;=\; e_k(n) - \eta\,\frac{\lambda_k e_k(n)}{\sqrt{\sum_{m \le n}\lambda_k^2 e_k(m)^2}}
\;=\; e_k(n)\left(1 - \frac{\eta}{\sqrt{\sum_{m\le n} e_k(m)^2}}\right). $$

**The curvature $\lambda_k$ cancels exactly.** Every direction follows the same recursion, driven only
by its own error history. This is precisely the "equalise the rate across directions" property we
want, and it is a full-matrix property: diagonal AdaGrad in the raw weight basis only has it if the
Hessian happens to be diagonal in that basis, which it is not.

But the shape is wrong. With no gradient noise the accumulator $\sum_m e_k(m)^2$ converges to a finite
limit, so the multiplier settles at a constant $1 - \eta/\sqrt{S_\infty} < 1$ and the decay is
**geometric, not a power law**. Numerically (30 directions spanning a 100-fold curvature range, all
starting at error 1): the trajectories of all 30 directions were bit-identical, and the error fell
below $10^{-300}$ by $n \approx 10^5$ with a log-log slope of $0.00$ over the last half of the run —
a straight line on a linear-log plot, not on a log-log plot. This matches the reported behaviour of
constant-rate Adam in the existing experiments (slope about $-0.8$ and then saturating): an adaptive
method with a saturating accumulator produces a curve that looks like a steep power law for a while
and then flattens.

Adding gradient noise keeps the accumulator growing, so the effective step decays like $n^{-1/2}$,
but the resulting error does *not* settle onto a clean $n^{-1/2}$ line either (measured: the three
sampled directions spread over a factor of five and did not share a slope).

**Conclusion for question 3.** Diagonal AdaGrad on least squares has no useful per-direction rate
guarantee; full-matrix AdaGrad has exactly the per-direction equalisation we need but delivers it as
a geometric rate. The practical lesson is to keep the curvature-cancelling preconditioner and replace
the accumulator by something that grows the way we want — a visit count — which is candidate method 3
in Section 7.

### 4.5 Practical full-matrix preconditioners

- **Agarwal, Bullins, Chen, Hazan, Singh, Zhang, Zhang, "Efficient full-matrix adaptive
  regularization" (ICML 2019, arXiv 1806.02958).** Makes full-matrix AdaGrad practical by computing
  the inverse square root of a low-rank matrix built from a window of recent gradients. At our scale
  (34,176 weights, 100 positions) full-matrix preconditioning is directly affordable and this
  machinery is not needed, but the paper is the reference for the claim that full-matrix
  preconditioning is what changes the per-direction behaviour.
- **Xie, Wang, Reddi, Kumar, Li, "Structured preconditioners in adaptive optimization: a unified
  analysis" (arXiv 2503.10537, 2025).** One analysis covering diagonal AdaGrad, full-matrix AdaGrad,
  AdaGrad-Norm and Shampoo, recovering the known rates and improving the rate for a one-sided
  Shampoo variant; notably they find that more structured (cheaper) preconditioners can beat
  full-matrix ones. Useful as the current map of which preconditioner gets which guarantee.

## 5. Question 4 — residual decay at individual data points

There is very little literature that controls the residual *at each data point separately*. Here is
everything I found that speaks to it, with what each actually delivers.

### 5.1 The classical exact answer: leverage / the hat matrix

For ordinary least squares with design matrix $A$ and independent noise of variance $\sigma^2$, the
fitted values are $\hat y = Hy$ with $H = A(A^{\top}A)^{-1}A^{\top}$, and

$$ \operatorname{Var}(\hat y_i) = \sigma^2 h_{ii}, \qquad
\operatorname{Var}(y_i - \hat y_i) = \sigma^2 (1 - h_{ii}), $$

where $h_{ii}$ is the leverage of observation $i$. **Hoaglin and Welsch, "The hat matrix in
regression and ANOVA" (The American Statistician 32(1):17–22, 1978)** is the standard reference.
Read as a statement about rates: the per-point prediction error is $\sigma\sqrt{h_{ii}}$ and, since
$h_{ii}$ scales like (degrees of freedom)/(sample size), the per-point error decays as $n^{-1/2}$
**at every point, with only the constant varying by point**. This is the exact classical statement of
what we want, and Section 6 shows it is what Polyak–Ruppert averaging reproduces iteratively.

### 5.2 Rates in stronger norms (the worst position, not each position)

**Fischer and Steinwart, "Sobolev norm learning rates for regularized least-squares algorithms"
(JMLR 21, 2020; arXiv 1702.07254)** extend least-squares learning rates from the $L^2$ norm to
stronger norms including, under conditions, the supremum norm — the first such $L^\infty$ rates for
this problem. The supremum norm bounds $\max_i b_i(n)$, so it constrains the *worst* position, and the
rate is strictly slower than the $L^2$ rate. This is the right tool for a guarantee of the form "no
position lags behind", but it says nothing about a position's own visit count.

### 5.3 Per-eigenmode, not per-point

**Bordelon, Canatar, Pehlevan, "Spectrum dependent learning curves in kernel regression and wide
neural networks" (ICML 2020, arXiv 2002.02561)** decompose the generalisation error of kernel
regression (and, through the tangent kernel, of wide networks) into contributions from each spectral
mode, and show that as the training set grows the learner fits successively higher modes. This is the
kernel-regression analogue of Section 1: the natural coordinates are eigenmodes, each with its own
rate, and a per-point statement requires recombining the modes — which is where the heterogeneity in
our measurements comes from.

### 5.4 Row-wise methods

- **Strohmer and Vershynin, "A randomized Kaczmarz algorithm with exponential convergence" (Journal
  of Fourier Analysis and Applications 15:262–278, 2009; arXiv math/0702226).** Each step projects
  onto the solution set of a single equation (a single position, in our language), zeroing that row's
  residual exactly; the expected error contracts geometrically at a rate set by the scaled condition
  number, independent of the number of equations. This is the natural starting point for a
  *per-position* update rule: the update at position $i$ can be made to change position $i$'s
  residual by a chosen factor.
- **Epperly, Goldshlager, Webber, "Randomized Kaczmarz with tail averaging" (arXiv 2411.19877,
  2024–2025).** Plain randomized Kaczmarz does not converge to the least-squares solution when the
  system is inconsistent (noisy); tail-averaging the final iterates restores convergence at a
  polynomial rate that is optimal for any method that accesses one row at a time. This is the exact
  combination our candidate method 1 uses: noisy targets plus tail averaging, with per-row updates.

### 5.5 What is missing

No paper I found states a per-point residual rate as a function of *that point's own* number of
visits under non-uniform sampling. The nearest thing is the leverage formula, and Section 6 derives
the per-position version of it for the averaged-iterate estimator, which turns out to give exactly
$n_i^{-1/2}$.

## 6. The per-position Polyak–Ruppert formula, derived and checked

This is the central technical result of these notes. It is a direct consequence of Polyak–Juditsky
plus a leverage computation, but I did not find it stated anywhere in this form.

**Setting.** Positions $i = 1,\dots,N$ with tangent features $\phi_i$. Each step samples position $i$
with probability $p_i$ and takes a stochastic gradient step on the squared error against a *noisy*
target $f(x_i) + \xi$, with $\xi$ zero-mean of variance $\sigma^2$ drawn afresh at each visit. The
bonus is measured against the *clean* target $f(x_i)$. Let
$H = \sum_i p_i \phi_i\phi_i^{\top}$ and $\Sigma = \sigma^2 H$ (this is the well-specified case; it
holds because the noise is independent of the input).

**Step 1 (Polyak–Juditsky).** The averaged iterate satisfies
$\operatorname{Cov}(\bar w_n) \approx \dfrac{1}{n} H^{-1}\Sigma H^{-1} = \dfrac{\sigma^2}{n}H^{-1}$,
so the residual at position $i$ has variance $\dfrac{\sigma^2}{n}\,\phi_i^{\top}H^{-1}\phi_i$.

**Step 2 (the leverage identity).** If the $N$ tangent features $\phi_1,\dots,\phi_N$ are linearly
independent — i.e. the predictor can fit the $N$ positions independently, which holds whenever the
model can interpolate them — then

$$ \phi_i^{\top} H^{-1}\phi_i \;=\; \frac{1}{p_i} \qquad \text{exactly, for every } i. $$

Proof: stack the features as $\Phi$ (rows $\phi_i^{\top}$) with thin singular value decomposition
$\Phi = USV^{\top}$, $U$ orthogonal $N\times N$. Then $H = \Phi^{\top}\!P\Phi$ with
$P = \operatorname{diag}(p)$, and
$\Phi H^{+}\Phi^{\top} = U(U^{\top}PU)^{-1}U^{\top} = P^{-1}$, whose $i$-th diagonal entry is
$1/p_i$. The feature geometry, the conditioning of $K$, and the eigenvalue spread all cancel.

**Step 3 (the conclusion).** Writing $n_i = p_i n$ for the expected number of visits to position $i$,

$$ \mathbb{E}\big[b_i(n)^2\big] \;\approx\; \frac{\sigma^2}{p_i\, n} \;=\; \frac{\sigma^2}{n_i},
\qquad\text{so}\qquad b_i(n) \;\approx\; \frac{\sigma}{\sqrt{n_i}} . $$

**Every position decays as the inverse square root of its own visit count, with the same constant
$\sigma$.** This is the count-based bonus, exactly, and it answers both the uniform-visitation
question and the non-uniform extension in one statement.

**If the model cannot interpolate.** With linearly dependent features (fewer effective degrees of
freedom than positions) the same argument gives $\phi_i^{\top}H^{+}\phi_i \le 1/p_i$, so
$b_i(n) \le \sigma/\sqrt{n_i}$: the bonus is a *conservative* count bonus, reduced exactly where the
model generalises between positions. For exploration that is arguably the behaviour one wants; it is
also the same conclusion Ciosek, Fortuin, Tomioka, Hofmann and Turner reach for fitted random priors
in "Conservative uncertainty estimation by fitting prior networks" (ICLR 2020), by a different route.

**Numerical check** (details in Section 8):

- Uniform sampling, $N = d = 40$: the predicted per-position variance times $n$ was $40.0000$ for
  every position (ratio between the largest and smallest: $1.00$), equal to $\sigma^2 N$.
- Non-uniform sampling, $p_i$ spread over a factor of 6, $N = d = 30$: the predicted per-position
  variance times $n$ matched $\sigma^2/p_i$ to four decimal places for every position; running
  averaged SGD for $10^6$ steps over 400 replicas gave a measured
  $b_i \big/ (\sigma/\sqrt{n_i})$ between $0.955$ and $1.153$ across all 30 positions, with
  per-position log-log slopes over the last decade between $-0.528$ and $-0.449$.

**Practical consequences.**

1. The noise must be injected. There is no noise in the current setup, so this mechanism is inactive
   and cannot be the explanation for any slope currently observed.
2. The bonus must be measured against the *clean* target while training against the *noisy* one.
3. The transient matters: the initial-condition term decays as $n^{-1}$ under full averaging and must
   fall below the noise floor before the slope reads $-1/2$. Tail averaging (Section 2.5) removes it
   geometrically and is the recommended form.
4. Set the noise level so the bonus starts at the desired value: with $b_i \approx \sigma/\sqrt{n_i}$
   and $n_i = 1$ at the first visit, $\sigma = 1$ gives $b_i \approx 1$ at the first visit.

## 7. Candidate methods, in the order I would try them

### 7.1 Noisy targets plus tail averaging

- **Rule.** Sample position $i$ (uniformly, or with the visitation distribution). Regress the
  predictor on $f(x_i) + \xi$ with $\xi \sim \mathcal{N}(0,\sigma^2 I_m)$ drawn fresh each visit.
  Keep a tail average of the weights, $w_{t:T}$, restarted at $t = T/2$ (or maintain several running
  tail averages). Report $b_i = \|g_{w_{t:T}}(x_i) - f(x_i)\|_2$, measured against the clean target.
- **Why it should give a uniform $n_i^{-1/2}$.** Section 6: the averaged iterate's per-position
  residual variance is exactly $\sigma^2/n_i$ once the initial-condition term is below the noise
  floor, for any feature geometry, any conditioning, and any visitation distribution.
- **Knobs.** $\sigma$ (sets the level, $b_i \approx \sigma/\sqrt{n_i}$); the constant step size
  $\gamma$ (must satisfy $\gamma < 1/R^2$ for stability but does not affect the leading term); the
  burn-in fraction.
- **Risks.** The predictor is now chasing a moving target, so with a large $\sigma$ the linearisation
  may break; the tangent features change as the weights move, which the theory assumes fixed. Check
  by measuring whether $b_i\sqrt{n_i}$ is constant across positions.

### 7.2 Shrink each position's residual by an explicit factor each visit

- **Rule.** Keep a visit count $n_i$. On a step that visits a set $S$ of positions, form targets
  $$ T_i \;=\; f(x_i) + \Big(1 - \frac{c}{n_i}\Big)\big(g(x_i) - f(x_i)\big) \quad (i \in S), \qquad
  T_i = g(x_i) \quad (i \notin S), $$
  and take an inner optimization step (or a few) that fits $g$ to $T$. With $c = 1/2$ the residual at
  position $i$ is multiplied by $1 - \tfrac{1}{2n_i}$ on each of its visits, so after $n_i$ visits
  $$ b_i \;=\; b_i(0)\prod_{m=1}^{n_i}\Big(1 - \frac{1}{2m}\Big) \;\approx\; b_i(0)\, n_i^{-1/2}. $$
- **Why it works.** The rate is imposed directly in function space and never passes through the
  tangent-kernel spectrum, so Section 1's obstruction does not apply. It handles non-uniform
  visitation for free, because the schedule is indexed by $n_i$, not by $n$.
- **The exact version.** In the linearised regime the required weight step is
  $\Delta w = -J^{+} D r$ with $D = \operatorname{diag}(c/n_i)$ — a Gauss–Newton step through the
  tangent-kernel Gram matrix. At our scale ($N = 100$) the linear solve is trivial.
- **Measured behaviour on the real network** (Section 8): with a plain inner Adam solve (150 steps
  per outer step), 400 outer steps, per-position slopes were $-0.41$ to $-0.33$ under uniform
  visitation (spread $0.08$) and $-0.58$ to $-0.37$ under 6-fold non-uniform visitation, with
  $b_i\sqrt{n_i}$ constant to within a factor $1.5$ to $1.9$. Compare the same network under constant
  Adam: slopes $-0.89$ to $-0.44$, spread $0.45$. The residual gap from $-0.5$ is inner-solve
  undershoot (the inner fit realises slightly less shrinkage than requested every step, which acts as
  a smaller effective $c$); fixes are more inner steps, an exact Gauss–Newton inner solve, or
  calibrating $c$ upward by the measured realised shrinkage.
- **Risks.** The predictor is trained on its own outputs at the positions it is not visiting, which
  is a fixed point that plain training would also reach but which can drift. Watch for the inner
  solve degrading over time.

### 7.3 Cancel the curvature, then use a $c/n$ step size with $c = 1/2$

- **Rule.** Apply a preconditioner that makes the effective curvature the same in every direction —
  full-matrix AdaGrad (Section 4.4 shows the $\lambda_k$ cancels), or an explicit Gauss–Newton
  preconditioner $K^{-1}$ on the tangent-kernel Gram matrix — and drive it with a step size $c/n$,
  $c = 1/2$. Then every mode's factor is $\prod_{m\le n}(1 - c/m) \approx n^{-1/2}$, so the residual
  vector as a whole is multiplied by a scalar and **every position decays as $n^{-1/2}$ with its own
  initial value as the constant**.
- **Why it should work.** With equal effective curvature, Section 1's ratio $\lambda_k/\lambda_j$
  becomes 1 and the schedule exponent applies to every mode identically.
- **Limitation.** This is uniform-visitation only. A single global step size cannot give position $i$
  an exponent tied to $n_i$; the non-uniform case needs the per-position diagonal of method 7.2.
- **Relation to method 7.2.** Method 7.2 with $D = (c/n)I$ *is* this method. Method 7.2 is the
  generalisation that replaces the scalar by a per-position diagonal.

### 7.4 Square-root readout on top of averaged full-batch gradient descent

- **Rule.** Run plain full-batch gradient descent with a constant step, keep the Polyak (or tail)
  average of the weights, and report $b_i = \sqrt{\|g_{\bar w}(x_i) - f(x_i)\|_2}$, normalised so
  that $b_i = 1$ at $n = 1$.
- **Why it works, partially.** Section 3.1: the averaged residual has log-log slope exactly $-1$ at
  every position once past the transient, so its square root has slope exactly $-1/2$ at every
  position. This is the cheapest change on this list — one line — and it borrows the *uniformity*
  that averaging already provides.
- **Limitations.** The per-position *constants* remain heterogeneous (a factor of about 119 in the
  linear test), so positions start at 1 by normalisation but separate later; and the uniformity only
  holds for $n \gg 1/(\eta\lambda_{\min})$, which with a measured condition number of
  $5.6\times10^{5}$ is a long wait. It also does not extend to non-uniform visitation. Worth running
  as a diagnostic because it isolates whether the uniformity mechanism is present.

### 7.5 Make the tangent features of the positions independent

- **Rule.** Change the architecture so each position's output depends on a private set of parameters
  — for example a wide last layer with a localised readout, a per-position output head, or random
  features narrow enough in input space that the 100 positions barely overlap.
- **Why it helps.** It makes the leverage identity of Section 6 hold with equality (so method 7.1
  gives exactly $\sigma/\sqrt{n_i}$ rather than something smaller), and it makes method 7.2's inner
  solve exact rather than approximate, because moving one position's output no longer disturbs the
  others.
- **Cost.** It removes generalisation between nearby positions, which for an exploration bonus is a
  real loss — a never-visited position adjacent to a much-visited one would keep a full bonus. Treat
  this as a diagnostic configuration that isolates the mechanism, not as the final method.

### 7.6 Things that will not work, and why

1. **Any pure global learning-rate schedule** (cosine, step decay, $1/t$, $1/\sqrt t$, warmup) —
   Section 1: the ratio of per-mode log-decays is fixed at $\lambda_k/\lambda_j$ whatever the
   schedule.
2. **Step decay / geometric schedules** — they make $\sum_m\eta_m$ bounded, so each mode settles at a
   constant instead of following any power law (Ge et al. 2019 recommend them for reaching the
   optimum fast, which is the opposite of what we want).
3. **AdaGrad or Adam with their own accumulators, in the deterministic setting** — the accumulator
   saturates as gradients vanish, so the decay becomes geometric (Xie, Wu, Ward 2020; measured in
   Section 8). This is the mechanism behind the reported "slope about $-0.8$, then saturating".
4. **Averaging alone, hoping for $-1/2$** — averaging a noiseless quadratic gives $-1$, not $-1/2$
   (Défossez and Bach 2015; Flammarion and Bach 2015; measured in Section 8).

## 8. Numerical checks run for these notes

All computed with the project's canonical environment
(`/p/rlprojects/RND/.venvs/exploration/bin/python`); the scripts are in
`/p/rlprojects/RND/11_decay_rate/literature/02_averaging_checks/`, numbered in the order of the list
below. These are my own computations, not results taken from the papers.

1. **Per-mode behaviour of the averaged iterate.** 100-position linear model, tangent-kernel
   condition number 10, constant step $\eta = 1/\lambda_1$, horizon $10^7$. Per-position log-log
   slope of the averaged residual: $-1.0000$ for all 100 positions, spread $0.0000$; per-position
   levels differ by a factor of 119. With condition number 100 and horizon $10^5$ (so the slowest
   modes are still in transient) the per-position slopes ranged $+0.49$ to $-1.39$. Confirms the
   formula $\bar r_k(n) = \dfrac{1-(1-\eta\lambda_k)^n}{n\eta\lambda_k}r_k(0)$ and its transient.
2. **Per-position Polyak–Ruppert variance.** Uniform sampling, $N = d = 40$: predicted per-position
   variance times $n$ equal to $\sigma^2 N = 40$ at every position (ratio 1.00). Non-uniform
   sampling, 6-fold spread in $p_i$, $N = d = 30$: predicted values matched $\sigma^2/p_i$ exactly;
   measured over 400 replicas at $n = 10^6$, $b_i\big/(\sigma/\sqrt{n_i}) \in [0.955, 1.153]$ and
   per-position slopes over the last decade $\in [-0.528, -0.449]$.
3. **AdaGrad on a noiseless quadratic.** 30 directions, curvature spread 100-fold, equal initial
   errors. All 30 trajectories identical (the curvature cancels), decay geometric to below $10^{-300}$
   by $n\approx10^5$, log-log slope $0.00$ over the last half. With gradient noise added the
   directions separated by a factor of five and shared no common slope.
4. **Tangent-kernel conditioning of the actual architecture.** 4 → 256 → ReLU → 128 on a 10-by-10
   grid of positions: Gram eigenvalues $\lambda_{\max} = 1.7\times10^{5}$, median $2.0$,
   $\lambda_{\min} = 0.31$; condition number $5.6\times10^{5}$.
5. **The real network under three optimizers** (2,000 to 20,000 full-batch steps, per-position
   log-log slopes of $b_i(n)/b_i(0)$ over the second half of the run):
   - constant-step Adam: mean $-0.58$, range $-0.89$ to $-0.44$, spread $0.45$;
   - constant-step gradient descent with Polyak weight averaging: the residual was still falling as
     about $n^{-1/3}$ at $n = 5000$, i.e. deep in the transient predicted by the condition number —
     the uniform $-1$ regime was not reached;
   - method 7.2 (per-position residual shrinkage, $c = 1/2$, 400 outer steps, 150 inner Adam steps):
     uniform visitation mean $-0.37$, range $-0.41$ to $-0.33$, spread $0.08$; 6-fold non-uniform
     visitation mean slope against each position's own visit count $-0.45$, range $-0.54$ to $-0.35$,
     and $b_i\sqrt{n_i}$ constant to within a factor 1.9.

The last line is the practically important one: **the per-position spread under the residual-shrinkage
method was 0.08, against 0.45 for constant Adam**, and the method extended to non-uniform visitation
without any change beyond indexing the schedule by $n_i$.

## 9. Papers, with identifiers

| paper | where | identifier |
|---|---|---|
| Ruppert, *Efficient estimations from a slowly convergent Robbins–Monro process* | Cornell School of Operations Research and Industrial Engineering, technical report 781, 1988 | Cornell eCommons |
| Polyak and Juditsky, *Acceleration of stochastic approximation by averaging* | SIAM Journal on Control and Optimization 30(4):838–855, 1992 | — |
| Bach and Moulines, *Non-asymptotic analysis of stochastic approximation algorithms for machine learning* | NeurIPS 2011, pages 451–459 | — |
| Bach and Moulines, *Non-strongly-convex smooth stochastic approximation with convergence rate $O(1/n)$* | NeurIPS 2013 | arXiv:1306.2119 |
| Défossez and Bach, *Constant step size least-mean-square: bias-variance trade-offs and optimal sampling distributions* | AISTATS 2015 | arXiv:1412.0156 |
| Flammarion and Bach, *From averaging to acceleration, there is only a step-size* | COLT 2015 | arXiv:1504.01577 |
| Dieuleveut, Flammarion, Bach, *Harder, better, faster, stronger convergence rates for least-squares regression* | JMLR 18(101):1–51, 2017 | arXiv:1602.05419 |
| Jain, Kakade, Kidambi, Netrapalli, Pillutla, Sidford, *A Markov chain theory approach to characterizing the minimax optimality of stochastic gradient descent (for least squares)* | FSTTCS 2017 | arXiv:1710.09430 |
| Jain, Kakade, Kidambi, Netrapalli, Sidford, *Parallelizing stochastic gradient descent for least squares regression: mini-batching, averaging, and model misspecification* | JMLR 18(223):1–42, 2018 | arXiv:1610.03774 |
| Neu and Rosasco, *Iterate averaging as regularization for stochastic gradient descent* | COLT 2018 | arXiv:1802.08009 |
| Ge, Kakade, Kidambi, Netrapalli, *The step decay schedule: a near optimal, geometrically decaying learning rate procedure for least squares* | NeurIPS 2019 | arXiv:1904.12838 |
| Berthier, Bach, Gaillard, *Tight nonparametric convergence rates for stochastic gradient descent under the noiseless linear model* | NeurIPS 2020 | arXiv:2006.08212 |
| Varre, Pillaud-Vivien, Flammarion, *Last iterate convergence of SGD for least-squares in the interpolation regime* | NeurIPS 2021 | arXiv:2102.03183 |
| Zou, Wu, Braverman, Gu, Kakade, *Benign overfitting of constant-stepsize SGD for linear regression* | COLT 2021; JMLR 24, 2023 | arXiv:2103.12692 |
| Duchi, Hazan, Singer, *Adaptive subgradient methods for online learning and stochastic optimization* | JMLR 12:2121–2159, 2011 | — |
| Ward, Wu, Bottou, *AdaGrad stepsizes: sharp convergence over nonconvex landscapes* | ICML 2019; JMLR 21, 2020 | arXiv:1806.01811 |
| Xie, Wu, Ward, *Linear convergence of adaptive stochastic gradient descent* | AISTATS 2020 | arXiv:1908.10525 |
| Agarwal, Bullins, Chen, Hazan, Singh, Zhang (Cyril), Zhang (Yi), *Efficient full-matrix adaptive regularization* | ICML 2019 | arXiv:1806.02958 |
| Xie, Wang, Reddi, Kumar, Li, *Structured preconditioners in adaptive optimization: a unified analysis* | preprint, 2025 | arXiv:2503.10537 |
| Yao, Rosasco, Caponnetto, *On early stopping in gradient descent learning* | Constructive Approximation 26:289–315, 2007 | — |
| Fischer and Steinwart, *Sobolev norm learning rates for regularized least-squares algorithms* | JMLR 21, 2020 | arXiv:1702.07254 |
| Bordelon, Canatar, Pehlevan, *Spectrum dependent learning curves in kernel regression and wide neural networks* | ICML 2020 | arXiv:2002.02561 |
| Hoaglin and Welsch, *The hat matrix in regression and ANOVA* | The American Statistician 32(1):17–22, 1978 | — |
| Strohmer and Vershynin, *A randomized Kaczmarz algorithm with exponential convergence* | Journal of Fourier Analysis and Applications 15:262–278, 2009 | arXiv:math/0702226 |
| Epperly, Goldshlager, Webber, *Randomized Kaczmarz with tail averaging* | preprint, 2024–2025 | arXiv:2411.19877 |
| Ciosek, Fortuin, Tomioka, Hofmann, Turner, *Conservative uncertainty estimation by fitting prior networks* | ICLR 2020 | OpenReview BJlahxHYDS |

## 10. Open questions I would settle next

1. Does injecting target noise break the linearised regime at the noise level needed to put the
   variance term above the bias term? Measure $b_i\sqrt{n_i}$ across positions for
   $\sigma \in \{0.03, 0.1, 0.3, 1\}$ and look for the value at which it stops being flat.
2. Does the knee predicted by the two-term structure (slope $-1$ from the initial condition, then
   slope $-1/2$ from the noise floor) actually appear, and at the predicted step count? If it does,
   the mechanism is confirmed; if the curve goes straight to $-1/2$, something else is producing it.
3. How much of the residual-shrinkage method's gap from $-0.5$ is inner-solve undershoot? Compare the
   requested shrinkage $1 - c/n_i$ against the realised one at each outer step, and check whether
   correcting $c$ by the measured ratio closes the gap.
4. Does an exact Gauss–Newton inner step (solve the $100\times100$ tangent-kernel system) give
   per-position slopes at $-0.500$ with zero spread, as the algebra says it should?
5. Under non-uniform visitation, does method 7.1 reproduce $\sigma/\sqrt{n_i}$ on the real network,
   or does the tangent-feature overlap between neighbouring maze positions pull the bonus below the
   count bonus by a measurable and position-dependent amount?

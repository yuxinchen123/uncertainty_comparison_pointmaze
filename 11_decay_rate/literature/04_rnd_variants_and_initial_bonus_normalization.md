# RND variants and control of the initial bonus scale across inputs

Notes for the study in `11_decay_rate/`: a predictor $g$ (4 -> 256 -> ReLU -> 128) is fit by
full-batch mean-squared-error distillation to a frozen random target $f$ on a fixed set of about
100 maze positions, input $[x, y, 0, 0]$. The per-position bonus is the residual norm
$b_i(n) = \|g_n(x_i) - f(x_i)\|_2$ after $n$ full-batch steps. The goal is that at every position
at once, (1) $b_i(0) = 1$, and (2) $b_i(n) \propto n^{-1/2}$.

Throughout, "squared error" means $\|g_n(x_i) - f(x_i)\|_2^2$ and "residual norm" means
$\|g_n(x_i) - f(x_i)\|_2$. A rate on one is half the rate on the other: squared error
$\propto n^{-1}$ is the same statement as residual norm $\propto n^{-1/2}$. Every rate below says
which quantity it applies to and what has to be true for it to hold.

---

## 1. What the original RND paper normalizes, and what it does not

**Paper.** Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov, *Exploration by Random Network
Distillation*, ICLR 2019, arXiv:1810.12894.

### 1.1 The two normalizations, quoted

Both live in Section 2.4, "Reward and Observation Normalization".

- **Observation normalization.** "Observation normalization is often important in deep learning but
  it is crucial when using a random neural network as a target, since the parameters are frozen and
  hence cannot adjust to the scale of different datasets." The procedure: "we whiten each dimension
  by subtracting the running mean and then dividing by the running standard deviation. We then clip
  the normalized observations to be between -5 and 5."
- **Intrinsic reward normalization.** "In order to keep the rewards on a consistent scale we
  normalized the intrinsic reward by dividing it by a running estimate of the standard deviations of
  the intrinsic returns."

### 1.2 What axis each of these normalizes

- Observation normalization is **per input coordinate, pooled over the whole data stream**. It makes
  each of the 4 input dimensions have mean 0 and standard deviation 1 across the states seen so far.
  It does nothing to equalize the bonus **between two different positions**; it only sets the overall
  scale and location of the input cloud that the frozen random target is asked to read.
- Intrinsic reward normalization is **one scalar per update, pooled over all states**. The divisor is
  a running estimate of the standard deviation of the intrinsic *returns* (discounted sums of the
  bonus), estimated over the batch and over time. Dividing the whole bonus field by one number
  rescales every position by the same factor. By construction it can never change the *ratio*
  $b_i / b_j$ between two positions.

So: RND's normalizations are **across time and across input coordinates, never across positions**.
Nothing in the original paper makes $b_i(0)$ equal across $i$, and nothing in it makes the decay
exponent equal across $i$. This is not an oversight in the paper — for its purpose (keep the reward
scale usable by PPO's value head across games and across training) time normalization is the right
axis. It is simply not the axis this project cares about.

### 1.3 The paper's own list of why the prediction error is large

Section 2.2.1, "Sources of prediction errors", lists four factors, quoted:

1. "Amount of training data. Prediction error is high where few similar examples were seen by the
   predictor (epistemic uncertainty)."
2. "Stochasticity. Prediction error is high because the target function is stochastic (aleatoric
   uncertainty)."
3. "Model misspecification. Prediction error is high because necessary information is missing, or the
   model class is too limited."
4. "Learning dynamics. Prediction error is high because the optimization process fails to find a
   predictor in the model class that best approximates the target function."

Factor 1 is the one that should carry the count-like $n^{-1/2}$ signal. Factor 4 is what the
project's current runs are actually measuring: with a deterministic target and a fixed 100-point
training set, there is no factor-1 uncertainty at all once every point is in the training set —
every full-batch step trains on *all* of them. The residual is entirely a property of the
optimization path. That is worth stating explicitly, because it explains why the decay looks like
whatever the learning-rate schedule makes it look like, and why constant-rate Adam saturates
(see Section 5.1).

### 1.4 Why the untrained bonus differs across positions at all

The mechanism is that a random network's output magnitude is not constant over the input space.

- In the wide-layer limit a randomly initialized network is a Gaussian process with some kernel $k$
  (Neal 1996; Jaehoon Lee et al., *Deep Neural Networks as Gaussian Processes*, ICLR 2018,
  arXiv:1711.00165). For a $d$-dimensional output with independent coordinates,
  $\mathbb{E}\|f(x)\|_2^2 = d\,k(x,x)$.
- For a bias-free ReLU layer with He-scaled weights, the wide-limit covariance is the arc-cosine
  kernel of order 1 (Youngmin Cho, Lawrence Saul, *Kernel Methods for Deep Learning*, NIPS 2009),
  whose diagonal is $k(x,x) \propto \|x\|_2^2$. With bias terms included the diagonal becomes
  $k(x,x) = a + c\|x\|_2^2$ for constants $a, c > 0$ set by the initialization scales.
- Predictor and target are independent draws, so
  $\mathbb{E}\|g_0(x) - f(x)\|_2^2 = d\,(k_g(x,x) + k_f(x,x))$, and the expected initial residual
  norm is proportional to $\sqrt{a + c\|x\|_2^2}$.

Consequences for this project's inputs $[x, y, 0, 0]$:

- The systematic part of the initial bonus is a **monotone function of the distance of the maze
  position from the coordinate origin**. Positions near the origin get a small initial bonus,
  positions in the far corners get a large one, and the ratio between them is
  $\sqrt{(a + c R_{\max}^2)/(a + c R_{\min}^2)}$. Without biases and with a coordinate origin inside
  the maze, that ratio is unbounded.
- The fluctuation part is small at $d = 128$. In the wide limit
  $\|g_0(x) - f(x)\|_2^2 / (k_g + k_f)(x,x)$ is $\chi^2_d$, so the residual norm is
  $\chi_d$-distributed with relative standard deviation about $1/\sqrt{2d}$ — about 6% at $d = 128$.
  So most of the spread in the measured initial bonus field is the systematic $\|x\|$ trend, not
  draw-to-draw randomness. This is directly checkable: average $\|g_0(x) - f(x)\|_2^2$ over many
  seeds and plot it against $\|x\|_2^2$; the prediction is a straight line.
- The fluctuation part is **spatially smooth, not white**. The correlation between the residuals at
  $x$ and $x'$ is governed by the normalized kernel
  $k(x,x')/\sqrt{k(x,x)\,k(x',x')}$, which for the arc-cosine kernel is a smooth decreasing function
  of the angle between the two inputs. Nearby maze cells therefore have nearly identical initial
  residuals, and the initial bonus field looks like a smooth random surface laid on top of the radial
  trend.

DRND (Section 3.1 below) reports exactly this dependence: its Lemma 4.2 gives the difference in
expected initial bonus between two states as proportional to $\|x_2\|_2^2 - \|x_1\|_2^2$.

---

## 2. Randomized prior functions, and the prior view of the initial residual field

**Paper.** Ian Osband, John Aslanides, Albin Cassirer, *Randomized Prior Functions for Deep
Reinforcement Learning*, NeurIPS 2018, arXiv:1806.03335.

### 2.1 What it does

Each ensemble member is the sum of a trainable network and a fixed random "prior" network. The
training objective (their Equation 5, as rendered on the arXiv HTML) is

- $\tilde\theta + \arg\min_\theta \sum_i \|\tilde y_i - (f_{\tilde\theta} + f_\theta)(x_i)\|^2 + (\sigma^2/\lambda)\|\theta\|^2$,

with the fixed prior $f_{\tilde\theta}$ added to the trainable part, a ridge penalty on the trainable
weights, and — this is the part that matters here — **noisy regression targets**
$\tilde y_i \sim \mathcal{N}(y_i, \sigma^2)$ and a random prior draw
$\tilde\theta \sim \mathcal{N}(\bar\theta, \lambda I)$.

Their Lemma 3 states that in the linear-Gaussian case this procedure generates an exact sample from
the Bayesian posterior over $\theta$. **The noise on the targets is not optional** — the exactness
result requires it. Ciosek et al. (Section 3.6 below) restate this in their related-work section:
the existing justification for randomized prior functions "only holds for Bayesian linear regression
with non-zero noise added to the priors".

That requirement is the single most useful fact in this note. Injected target noise is what turns a
deterministic interpolation problem, whose residual goes to zero, into a posterior-variance problem,
whose residual decays like the inverse square root of the number of observations at each point.

### 2.2 The prior-network view of the initial residual

Reading RND through this lens: before any training, the residual $g_0(x) - f(x)$ is a draw from the
difference of two independent prior processes. Its distribution is entirely determined by the two
initialization kernels, and its magnitude is $\sqrt{d\,(k_g + k_f)(x,x)}$ in expectation. That is the
formal content of Section 1.4 above.

On spatial structure specifically: the randomized-prior papers do not analyze how the prior's
magnitude varies across the input space — the prior is described as fixed and random, with no claim
of spatial uniformity, and the only knob discussed is a single global scale factor $\beta$
multiplying the prior network. A global $\beta$ multiplies every position's initial residual by the
same number and therefore cannot equalize them. The spatial structure has to be read off the kernel,
as in Section 1.4.

---

## 3. Published RND modifications relevant to bonus scale and bonus decay

### 3.1 Distributional RND (the paper that names the exact problem)

**Paper.** Kai Yang, Jian Tao, Jiafei Lyu, Xiu Li, *Exploration and Anti-Exploration with
Distributional Random Network Distillation*, ICML 2024, arXiv:2401.09750.

This paper is the closest published statement of this project's problem. It names a "bonus
inconsistency" problem with two halves:

- **Initial bonus inconsistency** — "uneven distribution of bonuses among states at the beginning of
  training". This is exactly requirement (1) of this project.
- **Final bonus inconsistency** — "the final bonuses do not align with the dataset distribution,
  making it hard for the agent to effectively distinguish between frequently visited states". This is
  the weaker cousin of requirement (2).

Their demonstration: a dataset of 100 one-hot categories where category $i$ occurs $i$ times.
Figure 1 shows RND's bonus heat map before training is far from uniform, and after training does not
track visit frequency.

**Method.** Use $N$ fixed random target networks $\bar f_1,\dots,\bar f_N$ instead of one. On each
visit to $x$, sample an index uniformly and regress on that network's output:

- loss $L(\theta) = \|f_\theta(x) - c(x)\|^2$ where $c(x)$ is drawn uniformly from
  $\{\bar f_1(x),\dots,\bar f_N(x)\}$.

Define the first two moments across the $N$ target networks,
$\mu(x) = \frac{1}{N}\sum_i \bar f_i(x)$ and $B_2(x) = \frac{1}{N}\sum_i \bar f_i(x)^2$. The bonus is

- $b(x) = \alpha\|f_\theta(x) - \mu(x)\|^2 + (1-\alpha)\sqrt{\dfrac{[f_\theta(x)]^2 - [\mu(x)]^2}{B_2(x) - [\mu(x)]^2}}$.

**What each half does for us.**

- The first term reduces the *initial* spread. Their Lemma 4.1 gives expected squared error
  $(1 + 1/N)\,x^\top\Sigma x$ against the averaged target, and Lemma 4.2 gives the initial bonus
  difference between two states as proportional to $(1+N)\sigma^2/N \cdot (\|x_2\|^2 - \|x_1\|^2)$.
  Averaging $N$ targets shrinks the variance of the *random* part of the initial bonus by roughly
  $1/N$, and their Table 1 reports the Kullback-Leibler divergence of the pre-training bonus
  distribution from uniform as $0.0070 \pm 0.0063$ for DRND against $0.0377 \pm 0.0248$ for RND. Note
  what this does **not** fix: the $\|x\|^2$-dependent systematic part survives averaging, because
  every target network has the same kernel diagonal. Averaging kills the draw-to-draw noise, not the
  trend.
- The second term is a **per-position normalized statistic**. Their Lemma 4.3, verbatim: "Let $f^*(x)$
  be the optimal function which satisfy Equation 4, the statistic
  $y(x)=\frac{[f^*(x)]^2-[\mu(x)]^2}{B_2(x)-[\mu(x)]^2}$ is an unbiased estimator of $1/n$ with
  consistency." Here $n$ is the number of times $x$ has been visited. The denominator
  $B_2(x) - [\mu(x)]^2$ is the variance of the target distribution **at that specific $x$** — that is,
  the bonus is divided pointwise by a quantity computed at $x$ itself. The paper describes the second
  bonus term, $\sqrt{y(x)}$, as "the estimation of $1/\sqrt{n}$".

So DRND already contains a published construction whose second bonus term is, in expectation, a
per-position $n^{-1/2}$ signal that equals 1 at $n = 1$ regardless of the position. The mechanism is
sampling noise from the discrete target distribution, and the pointwise normalizer is that
distribution's own variance at $x$.

Caveats to carry: the $1/n$ statement is for the **converged** predictor $f^*$, i.e. the predictor
that has reached the least-squares solution given the sampled targets; and the estimator is unbiased
for $1/n$ (a squared-error-like quantity), so its square root is only approximately $n^{-1/2}$ and is
biased low by Jensen. Their algorithm also normalizes intrinsic rewards over the batch, which is the
time-axis normalization inherited from RND and is irrelevant to cross-position uniformity.

### 3.2 Random Distribution Distillation (the cleanest published version of the mechanism)

**Paper.** Zhirui Fang, Kai Yang, Jian Tao, Jiafei Lyu, Lusong Li, Li Shen, Xiu Li, *Exploration by
Random Distribution Distillation*, arXiv:2505.11044 (preprint; no venue confirmed as of this note).

**Method.** Replace the fixed target value at $s$ by a fresh draw from a Gaussian whose mean is a
fixed random network's output:

- "Each time a target output $f_{tar}(s)$ is sampled from this distribution
  $f_{tar}(s)\sim\mathcal{N}(\mu_{\bar\theta}(s),\sigma^2_{\bar\phi}(s))$. We use the MSE loss to
  distill the target output by predictor network $f_\theta(s)$, i.e., minimize the loss below:
  $L(\theta)=\|f_\theta(s)-f_{tar}(s)\|^2$."

A **fresh** sample is drawn on **every visit**. In their implementation the variance is not
state-dependent: "in the actual implementation of the algorithm, we did not assign a unique variance
to each distribution; instead, we set them all to a fixed value of $\sigma^2$."

**Why it gives $n^{-1/2}$ per position.** The converged predictor is the empirical mean of the draws
at that state: $f^{n}_{\theta^*}(s) = \frac{1}{n}\sum_{i=1}^{n} f_{tar}^{i}(s)$. Hence the deviation
from the distribution mean is the standard error of a mean of $n$ independent draws. Their Lemma 4.1
states that $z_n(s)=\|f^{*}_{n\theta}(s)-\mu_{\bar\theta}(s)\|^2/\sigma^2_{\bar\phi}(s)$ is an
unbiased and consistent estimator of $1/n$; Corollary 4.2 shows $\mathrm{Var}(z_n) \le \mathrm{Var}(y_n)$,
i.e. strictly less variance than DRND's statistic at equal mean; Theorem 4.3 gives "a convergence
rate ranging from $O(n^{-1})$ to $O(n^{-1.5})$" for the concentration of $z_n$ around its mean.
Section 4.4 relates the bonus to a pseudo-count through a $\sigma^2(s)/n$ term.

Restated in this note's units: **the squared error at a state decays as $\sigma^2 d / n$ and the
residual norm as $\sigma\sqrt{d}\,n^{-1/2}$, with $n$ the count at that state**, provided the
predictor tracks the running mean of the draws at that state and provided the fit at one state does
not borrow from the others.

**Initial value.** From their ablation (Section 5.4): "When the mean was 1, for an unseen state,
$z_n(s)$ would initially give a reward close to 1 (achieved through $\|f_\theta(s)-\mu\|^2$)." So the
normalized statistic starts at 1 by construction — at $n = 1$, and at every state, because it is
normalized by that state's own target variance.

**What their practical bonus actually is.** Equation 11 is $b(s) = \frac{1}{d}\|f_\theta(s) -
\mu_{\bar\theta}(s)\|^2$, i.e. the pointwise $\sigma^2(s)$ division appears in the analysis statistic
$z_n$, not in the deployed bonus — which is harmless for them because they use a single scalar
$\sigma$, so the two differ by a constant. For this project the pointwise division is the whole
point, so use $z_n$, not their Equation 11.

### 3.3 Anti-exploration by RND (conditioning, not normalization)

**Paper.** Alexander Nikulin, Vladislav Kurenkov, Denis Tarasov, Sergey Kolesnikov,
*Anti-Exploration by Random Network Distillation*, ICML 2023, arXiv:2301.13616.

RND used as an out-of-distribution penalty in offline reinforcement learning. Their finding is about
**how the action is fed into the networks**: a naive conditioning makes the RND bonus impossible for
the actor to minimize, and Feature-wise Linear Modulation conditioning fixes it. No pointwise bonus
normalization; relevant here only as evidence that the *architecture of the target*, not just its
randomness, decides whether the residual field is well behaved.

### 3.4 Self-supervised network distillation (a trained target)

**Paper.** Matej Pecháč, Michal Chovanec, Igor Farkaš, *Self-supervised network distillation: an
effective approach to exploration in sparse reward environments*, arXiv:2302.11563 (also published in
a journal version).

Trains the target network too, with a self-supervised objective, instead of freezing a random one.
Motivated by the observation that a purely random target has an arbitrary and uninformative
geometry. Relevant as a reminder that the initial bonus field's shape is a free design choice, but it
offers nothing that makes $b_i(0)$ equal across $i$.

### 3.5 Never Give Up (a time-axis normalizer that starts at 1)

**Paper.** Adrià Puigdomènech Badia et al., *Never Give Up: Learning Directed Exploration
Strategies*, ICLR 2020, arXiv:2002.06038.

Uses the RND error as a lifelong novelty **multiplier**:
$\alpha_t = 1 + \dfrac{e_{RND}(s_t) - \mu_e}{\sigma_e}$, with $\mu_e, \sigma_e$ running mean and
standard deviation of the RND error, clipped from below at 1. Worth noting because it is the closest
published thing to "make the bonus equal 1 by construction" — but the normalizer is a **running
scalar over the state stream**, so it centres the *population* of errors at 1, not each position's
error at 1. Same axis mistake as Section 1.2 for our purposes.

### 3.6 Fitting random priors as uncertainty estimation (the theory that ties it together)

**Paper.** Kamil Ciosek, Vincent Fortuin, Ryota Tomioka, Katja Hofmann, Richard E. Turner,
*Conservative Uncertainty Estimation By Fitting Prior Networks*, ICLR 2020 (OpenReview `BJlahxHYDS`;
no arXiv version found).

Setup: draw $B$ random prior functions $f^i$, fit a predictor $h_{Xf^i}$ to each by squared error on
the training inputs $X$, and use the mean over bootstraps of $\frac{1}{M}\|f(x^\star) -
h_{Xf^i}(x^\star)\|^2$ as the uncertainty estimate ($M$ output units). They report $B = 1$ is usually
enough in practice. There is no per-input normalization in their estimator.

**Proposition 1 (strict conservatism in expectation).** If the prior process $f$ is a Gaussian
process, then for any predictor $h$,

- $\tilde\sigma^2_\mu(x^\star) = \sigma^2_X(x^\star) + \mathbb{E}_{f(X)}\left[\frac{1}{M}\|\mu_{Xf}(x^\star) - h_{Xf}(x^\star)\|^2\right] \ \ge\ \sigma^2_X(x^\star)$,

with equality if and only if $h_{Xf}(x^\star) = \mu_{Xf}(x^\star)$. In words: **the expected squared
RND residual equals the Gaussian-process posterior variance at that point plus the squared gap
between the trained predictor and the posterior mean.** The first term is factor 1 of Burda et al.'s
list (data), the second is factor 4 (learning dynamics).

**Proposition 2 (concentration).** Under Lipschitz continuity and boundedness of the predictor class,
and assuming training converges to a fixed training loss, the estimate tends to 0 as the training set
grows. No rate in the number of visits to a single point is given.

**The design lever they hand us**, from their Section 5 ("Modeling Epistemic and Aleatoric
Uncertainty"): choose the prior process as a sum $\{f(x)\} = \{n(x) + \epsilon(x)\sigma_A\}$, with
$\epsilon(x) \sim \mathcal{N}(0,1)$ drawn independently at each $x$. This is the same noise injection
that Osband et al.'s Lemma 3 requires, stated as a modelling choice. With such a term the posterior
variance at a point observed $n$ times does not go to zero as an interpolation residual does; it goes
to roughly $\sigma_A^2/n$ — squared error $\propto n^{-1}$, residual norm $\propto n^{-1/2}$.

### 3.7 The infinite-width identification of the RND signal

**Paper.** Moritz A. Zanger, Yijun Wu, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T. J.
Spaan, *On the Equivalence of Random Network Distillation, Deep Ensembles, and Bayesian Inference*,
arXiv:2602.19964 (v1 23 Feb 2026, v2 26 Feb 2026; no venue listed).

Setup analyzed: predictor $u(x;\vartheta_t)$ with $\vartheta_0 \sim \mathcal{N}(0,I)$ in neural
tangent kernel parameterization; fixed random target $g(x;\psi_0)$; loss
$\mathcal{L}(\vartheta_t) = \frac12\|u(\mathcal{X};\vartheta_t) - g(\mathcal{X};\psi_0)\|_2^2$;
$K$ parallel heads with signal $\epsilon^2(x) = \frac1K\sum_i (u_i(x) - g_i(x))^2$. Trained by
gradient flow, no ridge, no injected noise.

Results relevant here:

- **Theorem 3.1** gives the post-convergence covariance of the RND error at test points as a kernel
  expression built from the neural network Gaussian process kernels
  $\kappa_\epsilon = \kappa_u + \kappa_g$ and the neural tangent kernel $\Theta$:
  $\Sigma_\epsilon(\mathcal{X}_T,\mathcal{X}_T) = \kappa_\epsilon(\mathcal{X}_T,\mathcal{X}_T) +
  \Theta_{(\mathcal{X}_T,\mathcal{X})}\Theta_{(\mathcal{X},\mathcal{X})}^{-1}\kappa_\epsilon(\mathcal{X},\mathcal{X})\Theta_{(\mathcal{X},\mathcal{X})}^{-1}\Theta_{(\mathcal{X},\mathcal{X}_T)} - (\text{cross terms})$.
  It has the shape of a posterior covariance but with two different kernels, one carrying the prior
  scale and one carrying the training dynamics.
- **Theorem 3.4** states the multi-head RND squared error and the deep-ensemble predictive variance
  are equal in distribution (both a scaled chi-squared with $K$ degrees of freedom), and Corollary
  3.2 gives the scalar expectation version $\mathbb{E}[\epsilon^2] = \mathbb{V}[f(x;\theta_\infty)]$.
- **Proposition 4.1 ("Bayesian RND")** modifies the target to
  $\tilde g(x;\vartheta_0,\psi_0) = \nabla_{\vartheta_0}u(x;\vartheta_0)^\top\psi_0^*$ with
  $\psi_0^* = \{\psi_0^{(\le L-1)}, 0\}$ — that is, **the target copies the random parameters of all
  but the last layer and sets the last layer to zero**. Theorem 4.2 then gives the error covariance
  as the exact posterior predictive covariance
  $\Theta_{(\mathcal{X}_T,\mathcal{X}_T)} - \Theta_{(\mathcal{X}_T,\mathcal{X})}\Theta_{(\mathcal{X},\mathcal{X})}^{-1}\Theta_{(\mathcal{X},\mathcal{X}_T)}$.
  Corollary 4.3 turns this into a posterior sampling procedure.
- **No ridge, no injected noise, no rate.** A footnote says extending to a noisy observation model by
  adding $\sigma_n^2 I$ inside the kernel inversions "is straightforward", but they do not do it. And
  at training inputs, the residual covariance of the noiseless posterior vanishes.

That last point is the one to take away: **in the noiseless, converged, interpolating regime the
residual at a training point goes to zero, not to $n^{-1/2}$.** Our setting trains on all 100 points
at every step, so we are always in that regime unless we deliberately break it.

### 3.8 Other members of the same family, for completeness

- Moritz A. Zanger, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T. J. Spaan, *Contextual
  Similarity Distillation: Ensemble Uncertainties with a Single Model*, arXiv:2503.11339. Recasts
  ensemble-variance estimation as supervised regression onto kernel similarity targets, using neural
  tangent kernel theory. No normalization by a prior quantity in the abstract.
- Moritz A. Zanger, Max Weltevrede, Yaniv Oren, Pascal R. Van der Vaart, Caroline Horsch, Wendelin
  Böhmer, Matthijs T. J. Spaan, *Universal Value-Function Uncertainties*, arXiv:2505.21119. Uses
  temporal-difference learning against a synthetic reward drawn from a fixed random network; squared
  prediction error as uncertainty; equivalence to ensemble variance at infinite width.
- Abdul Wahab, Raksha Kumaraswamy, Martha White, *Value Bonuses using Ensemble Errors for Exploration
  in Reinforcement Learning*, Reinforcement Learning Journal / RLC 2025 (arXiv:2602.12375). Keeps an
  ensemble of random action-value functions and uses their estimation errors as value bonuses, aiming
  at first-visit optimism. The design goal — the bonus should be at full value on the first visit and
  fall thereafter — is the same as this project's requirement (1), reached by a different route.
- Mikael Henaff, Roberta Raileanu, Minqi Jiang, Tim Rocktäschel, *Exploration via Elliptical Episodic
  Bonuses*, NeurIPS 2022. The elliptical bonus
  $\sqrt{\phi(x)^\top(\lambda I + \sum_j \phi_j\phi_j^\top)^{-1}\phi(x)}$ is a count generalization
  whose value for a repeatedly visited, feature-orthogonal point is exactly
  $\|\phi\|/\sqrt{\lambda + n\|\phi\|^2}$. This is the algebraically cleanest object with both
  properties we want, and it is already implemented in this project as `EllipticalBonus`.
- Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Rémi Munos,
  *Unifying Count-Based Exploration and Intrinsic Motivation*, NeurIPS 2016, arXiv:1606.01868. The
  reference point for why $n^{-1/2}$ is the rate to aim for: pseudo-counts converted into a bonus of
  the form $\beta/\sqrt{\hat N}$.

### 3.9 Direct answer to question 3

Nothing in the published RND line normalizes the bonus by an **initial-predictor-copy residual**, and
nothing unit-normalizes the target output per input. The two published pointwise normalizations are:

- DRND's division by $B_2(x) - [\mu(x)]^2$, the variance of its $N$-target distribution at $x$; and
- RDD's division by $\sigma^2_{\bar\phi}(s)$, the variance of its Gaussian target distribution at $s$
  (used in the analysis statistic; their deployed bonus uses a constant $\sigma$).

Both are divisions by **the variance of the target distribution at that input**, which is the same
role the quantity $k(x,x)$ plays in Section 1.4 — a per-input prior scale. That is precisely the
family that constructions (a) and (b) of question 4 belong to.

---

## 4. Constructions that make $b(x) = 1$ at every $x$ at initialization

### 4.1 Construction (a): divide by a frozen copy of the initial predictor

Keep $g_0$ frozen and read out

- $\tilde b_i(n) = \dfrac{\|g_n(x_i) - f(x_i)\|_2}{\|g_0(x_i) - f(x_i)\|_2}$.

**Status in the literature.** Not found. The closest published relatives are the two pointwise target
variance divisions in Section 3.9, which divide by the *expected* magnitude rather than by a
realized sample of it.

**Properties.**

- $\tilde b_i(0) = 1$ exactly at every $i$, by construction, with no assumptions at all. This is its
  whole strength.
- It is valid online: the denominator needs no knowledge of the visit distribution, only a stored
  copy of the network at initialization (one extra forward pass, or a cached 100 x 128 table since
  the point set is fixed).
- The denominator is a single $\chi_d$ draw around $\sqrt{d\,(k_g+k_f)(x_i,x_i)}$. Relative standard
  deviation of the norm is about $1/\sqrt{2d}$: about 6% at $d = 128$, about 12% at $d = 32$, about
  35% at $d = 4$. So the "division amplifies noise where the denominator is small" hazard is real but
  weak at $d = 128$: the probability that $\|g_0(x) - f(x)\|$ falls to a fraction $\epsilon$ of its
  typical value scales like $\epsilon^{d}$, which is astronomically small at $d = 128$ and a genuine
  problem only for small output width. Guard anyway with a floor
  $\max(\|g_0(x_i) - f(x_i)\|, \varepsilon)$, and report the realized denominator spread.
- Cheaper variance reduction if wanted: average the denominator over $B$ independent predictor
  initializations, which cuts its relative standard deviation by $\sqrt{B}$; or use the analytic
  kernel diagonal instead of a sample.
- **The limitation that matters.** The denominator is a constant in $n$. Dividing by it shifts each
  per-position log-log curve down by a constant; it cannot change any curve's slope. If per-position
  slopes differ, the normalized curves still fan out, just from a common starting point. Construction
  (a) solves requirement (1) and does nothing at all for requirement (2).

### 4.2 Construction (b): zero-initialize the predictor head and unit-normalize the target

Set the predictor's last linear layer to zero so $g_0 \equiv 0$, and use the unit-normalized target
$\hat f(x) = f(x)/\|f(x)\|_2$. Then $b_i(0) = \|\hat f(x_i)\|_2 = 1$ exactly at every $i$, with no
estimator noise whatsoever.

**Status in the literature.**

- Zeroing the last layer of a network to control which kernel a target carries appears in Zanger et
  al. 2026, Proposition 4.1: their "Bayesian RND" target copies the random parameters of all layers
  but the last and sets the last layer to zero. Their purpose is different (making the target's
  kernel be the neural tangent kernel of the earlier layers rather than the neural network Gaussian
  process kernel), but the construction is the same one.
- Zero-initialized output heads are standard practice in other parts of deep learning (residual
  branch and adapter initialization), so the trick is not exotic.
- **Unit-normalizing the target output per input: not found in the RND literature.** Its meaning is
  clean though. In the wide-limit view, replacing $f$ by $f/\|f\|$ replaces the covariance kernel $k$
  by the correlation kernel $k(x,x')/\sqrt{k(x,x)k(x',x')}$, whose diagonal is exactly 1 at every
  $x$. Cho and Saul 2009 already discuss the normalized arc-cosine kernel; this construction is the
  network-level realization of that normalization. So construction (b) is exactly "make the prior
  variance uniform over the input space", which is precisely the defect diagnosed in Section 1.4.

**Pitfalls.**

- With the head at exactly zero, the gradient with respect to every parameter below the head is zero
  at step 0, so the first step moves only the head. In practice the model then behaves like linear
  regression on the frozen 256-dimensional random features for a while. For this study that is
  arguably a feature, not a bug — it makes the dynamics analyzable — but it should be stated, and it
  means the effective model class is a linear head on random features unless the head is initialized
  small-but-nonzero.
- Unit normalization does not change the *conditioning* of the fitting problem. Nearby maze positions
  still have nearly identical normalized targets and nearly identical features, so the neural tangent
  kernel spectrum, which is what drives the per-position slope heterogeneity (Section 5.1), is
  essentially untouched. Construction (b), like (a), fixes the start and not the slope.
- $\|f(x)\|_2$ can in principle be near zero. At $d = 128$ this is negligible, and it is zero
  probability for a bias-free network only at $x = 0$; still, use $f(x)/\max(\|f(x)\|_2,\varepsilon)$.
- A subtlety: $b_i(0) = 1$ becomes exact only if the readout is $\|g_n(x) - \hat f(x)\|_2$ with
  $g_0 \equiv 0$. If the head is instead initialized small but nonzero, $b_i(0) = 1 + O(\text{head
  scale})$, still uniform to first order but no longer exact.

### 4.3 A third construction: normalize by the prior kernel diagonal

Since the source of the nonuniformity is $k(x,x)$, one can divide by it directly:

- $\tilde b_i(n) = \dfrac{\|g_n(x_i) - f(x_i)\|_2}{\sqrt{d\,(k_g+k_f)(x_i,x_i)}}$,

with the denominator either computed analytically for the given architecture and initialization
(one hidden ReLU layer: an arc-cosine expression in $\|x_i\|$ and the bias variance) or estimated by
Monte Carlo over many independent $(g_0, f)$ draws. This is the noise-free version of (a) and the
model-based version of (b), and it makes the *expected* initial bonus exactly 1 at every position
while leaving the residual $\chi_d$ fluctuation visible — which is the honest thing to do if the
fluctuation is what one wants to study.

---

## 5. Synthesis: the start and the slope are two separate problems

### 5.1 Why a global learning-rate schedule cannot give a uniform slope

Work in the linearized (neural tangent kernel) regime. Let $\Theta \in \mathbb{R}^{m \times m}$ be the
empirical neural tangent kernel Gram matrix of the predictor over the $m \approx 100$ fixed points,
and let $R_n \in \mathbb{R}^{m \times d}$ hold the residuals $g_n(x_i) - f(x_i)$ as rows. Full-batch
gradient descent with step size $\eta_n$ on the squared error gives

- $R_{n+1} = (I - \eta_n\Theta)R_n$.

Diagonalizing $\Theta = \sum_j \lambda_j v_j v_j^\top$, the component of the residual along $v_j$ is
multiplied by $\prod_{k \le n}(1 - \eta_k\lambda_j)$.

1. **Constant step size.** Each eigendirection decays geometrically as $(1-\eta\lambda_j)^n$. That is
   exponential, not a power law. On a log-log plot an exponential looks steeper than any power law
   and then bends sharply down — which is exactly the reported behaviour of constant-rate Adam
   (apparent slope about $-0.8$, then saturating). Adam adds its own per-coordinate preconditioning
   on top, but the qualitative point stands: with a constant rate and a deterministic target, the
   residual at a training point tends to zero geometrically, and there is no power law to measure.
2. **Step size $\eta_n = c/n$.** Then
   $\prod_{k\le n}(1 - c\lambda_j/k) \approx \exp(-c\lambda_j \ln n) = n^{-c\lambda_j}$. Each
   eigendirection now decays as a power law **with its own exponent $c\lambda_j$**. The residual at a
   single position is a mixture over eigendirections,
   $b_i(n)^2 = \sum_j (v_j)_i^2\,\|R_0^\top v_j\|_2^2\, n^{-2c\lambda_j}$, so its local log-log slope
   is a weight-dependent average of the $-c\lambda_j$ that drifts with $n$ toward the smallest
   exponent. **The heterogeneity across positions observed in this project is therefore not noise: it
   is the neural tangent kernel spectrum showing through.** An aggregate slope near $-1/2$ with
   visibly different per-position slopes is exactly what this model predicts.

The fix follows from the same algebra: **precondition the update by $\Theta^{-1}$** so that every
eigendirection contracts by the same factor. With
$R_{n+1} = (I - \eta_n\Theta\Theta^{-1})R_n = (1-\eta_n)R_n$ and $\eta_n = \dfrac{1}{2n}$,

- $\displaystyle \prod_{k=1}^{n}\Bigl(1-\frac{1}{2k}\Bigr) = \frac{\Gamma(n+\tfrac12)}{\Gamma(\tfrac12)\,\Gamma(n+1)} \sim \frac{1}{\sqrt{\pi n}}$,

so **every** position's residual norm follows the same curve, exactly $\propto n^{-1/2}$, and the
ratio $b_i(n)/b_i(0)$ is identical across $i$ at every $n$ — not just asymptotically. Combined with
construction (a) or (b) for the denominator, this satisfies both requirements simultaneously and
exactly, in the linear regime.

### 5.2 The three mechanisms that can produce $n^{-1/2}$, and what each assumes

| mechanism | what decays as $n^{-1/2}$ | what it needs | handles non-uniform visitation? |
|---|---|---|---|
| Learning-rate schedule $\eta_n = c/n$ on a deterministic target | residual norm, per eigendirection with exponent $c\lambda_j$ | linearized dynamics; a preconditioner if the exponent must be uniform | No — one global schedule, one global clock |
| Fresh target noise on every visit, predictor tracking the running mean | residual norm at $x_i$, with $n_i$ the count at $x_i$ | predictor must reach (or track) the least-squares solution; features at different points must not leak into each other | Yes — the count is per position by construction |
| Explicit ridge and a posterior-standard-deviation readout | residual norm at $x_i$, as $\sqrt{\lambda/(\lambda+n_i\|\phi_i\|^2)}$ | frozen features; maintained inverse covariance; approximately orthogonal features across points | Yes — the covariance accumulates per position |

Mechanisms 2 and 3 are the same object seen twice: mechanism 2's converged residual **is** the
posterior standard deviation of mechanism 3 with $\lambda = \sigma^2$. Ciosek et al.'s Proposition 1
says this explicitly: expected squared RND residual = posterior variance + optimization gap.
Mechanism 1 lives entirely inside the "optimization gap" term and therefore cannot know about
per-position counts.

**The single most important consequence for the harder extension:** a global learning-rate schedule
is provably the wrong tool, because it is a function of the global step index alone. Under
non-uniform visitation the bonus at $i$ must depend on $n_i$, and only mechanisms 2 and 3 make that
dependence automatic.

### 5.3 The initial scale and the decay rate are controlled by different quantities

- Initial scale is controlled by the **prior variance at that input**, $k(x,x)$: fix it by dividing by
  a per-input reference (a frozen initial residual, an analytic kernel diagonal, a target
  distribution's own variance, or the norm of the target) or by constructing the target so its norm
  is 1 everywhere.
- Decay rate is controlled by the **conditioning of the fit** — the neural tangent kernel or feature
  Gram spectrum over the point set — and by whether a noise floor or ridge exists at all.

Fixing one does not fix the other. Any full solution combines one normalization from Section 4 with
one mechanism from Section 5.2.

---

## 6. Concrete methods to implement in this setting

Each is stated as: what to change, what it predicts, what to check, what can go wrong.

### 6.1 Noisy-target distillation with a per-position normalizer (the RDD/DRND mechanism)

- **Change.** At each full-batch step $n$, regress on $f(x_i) + \sigma\varepsilon_{i,n}$ with fresh
  $\varepsilon_{i,n}\sim\mathcal{N}(0,I_d)$ per point per step. Read out
  $\tilde b_i(n) = \dfrac{\|g_n(x_i) - f(x_i)\|_2}{\sigma\sqrt{d}}$.
- **Predicts.** If $g_n$ tracks the running mean of the $n$ draws at $x_i$, then
  $\mathbb{E}\|g_n(x_i)-f(x_i)\|_2^2 = \sigma^2 d / n_i$, so the **squared error decays as $n_i^{-1}$
  and the residual norm as $n_i^{-1/2}$**, with $\tilde b_i(1) = 1$ in expectation at every position.
  Under non-uniform sampling the count is that position's own count with no further change.
- **How to make the predictor actually track the running mean.** Either (i) freeze the hidden layer
  and update the linear head by recursive least squares (exact running mean, no tuning), or (ii) run
  gradient descent with a $1/n$ step size, which is the Robbins-Monro schedule whose fixed point is
  the running mean, or (iii) use iterate averaging (Boris Polyak, Anatoli Juditsky, *Acceleration of
  Stochastic Approximation by Averaging*, SIAM Journal on Control and Optimization 30(4), 1992),
  which gives the $n^{-1/2}$ parameter-error rate without needing the step size tuned to the spectrum.
- **Check.** Plot $\tilde b_i(n)$ per position on log-log; expect slope $-1/2$ at every $i$ and
  intercept 1 at $n=1$. Also plot the realized $\tilde b_i(n)\sqrt{n_i}$, which should be flat and
  concentrated around 1 with relative spread about $1/\sqrt{2d}$.
- **Can go wrong.** (i) Leakage between positions: if the model generalizes strongly between two
  nearby maze cells, the fit at one borrows the other's samples and the effective count is a
  kernel-weighted count larger than $n_i$; measure by comparing the fitted values at held-out
  positions. (ii) Under-trained predictor: if the fit lags the running mean, the residual carries an
  extra optimization term and the slope is too shallow early on. (iii) The residual is itself random
  with about 6% relative spread at $d=128$; average over seeds before reading a slope.

### 6.2 Ridge readout on frozen random features, normalized by its own initial value

- **Change.** Freeze the 256-dimensional hidden layer, write $\phi_i = \phi(x_i)$, maintain
  $\Lambda_n = \lambda I + \sum_i n_i \phi_i\phi_i^\top$, and read out
  $\tilde b_i(n) = \sqrt{\dfrac{\phi_i^\top\Lambda_n^{-1}\phi_i}{\phi_i^\top\Lambda_0^{-1}\phi_i}} =
  \dfrac{\sqrt{\lambda\,\phi_i^\top\Lambda_n^{-1}\phi_i}}{\|\phi_i\|_2}$.
- **Predicts.** $\tilde b_i(0) = 1$ exactly at every position, with no randomness. If the $\phi_i$
  across the 100 positions are mutually orthogonal, then
  $\tilde b_i(n)^2 = \dfrac{\lambda}{\lambda + n_i\|\phi_i\|_2^2}$, so with unit-normalized features
  and $\lambda = 1$, $\tilde b_i(n) = (1+n_i)^{-1/2}$ **exactly**, per position, under arbitrary
  visitation.
- **Why it is the same thing as 6.1.** This is the Gaussian-process posterior standard deviation at
  $x_i$ with observation noise $\lambda$; by Ciosek et al.'s Proposition 1 it is the floor that a
  perfectly optimized noisy-target RND residual reaches. It is also the E3B elliptical bonus with the
  initialization normalization added.
- **Can go wrong.** The 100 maze positions have highly correlated random features (nearby positions
  have kernel correlation near 1), so orthogonality fails badly and visiting one position shrinks its
  neighbours' bonuses. Two responses: (i) accept it and report the effective count
  $n^{\text{eff}}_i = (\tilde b_i^{-2} - 1)\lambda/\|\phi_i\|^2$ as the thing that actually decays;
  (ii) whiten the features over the fixed 100-point set, $\tilde\Phi = \Phi K^{-1/2}$ with
  $K = \Phi^\top\Phi$, which restores exact orthogonality and hence exact $(1+n_i)^{-1/2}$ — at the
  price of using knowledge of the point set, which is available in this controlled study but would be
  oracle knowledge in a real environment.

### 6.3 Gram-preconditioned full-batch gradient descent with a $1/(2n)$ step

- **Change.** Compute the Gram matrix over the 100 points once — either the last-layer feature Gram
  $K = \Phi^\top\Phi$ (if only the head trains) or the empirical neural tangent kernel Gram from
  per-sample Jacobians (100 x 100, trivial at this size) — and apply the update
  $R_{n+1} = (I - \eta_n\Theta(\Theta + \mu I)^{-1})R_n$ with a small stabilizing $\mu$ and
  $\eta_n = \dfrac{1}{2n}$. In parameter terms this is a Gauss-Newton / natural-gradient step
  restricted to the fixed point set.
- **Predicts.** Every position's residual norm follows
  $\prod_{k\le n}(1-\tfrac{1}{2k}) \sim (\pi n)^{-1/2}$ — the same curve at every position, not just
  the same asymptotic slope. Combined with the readout of 6.4 or 6.5, both requirements hold exactly.
- **Can go wrong.** (i) It does not generalize to non-uniform visitation: the preconditioner ties all
  positions to one global step counter. A per-position variant — shrink position $i$ by
  $(1 - \tfrac{1}{2n_i})$ using its own count — is straightforward here but is closer to a
  count-based bonus than to a distillation method, and should be labelled as such. (ii) The
  linearization is only approximate for a real ReLU network; recompute $\Theta$ periodically and
  report how much it moves. (iii) $\Theta$ over 100 nearby maze positions is badly conditioned; the
  ridge $\mu$ then reintroduces heterogeneity in the smallest directions, so report the condition
  number and the sensitivity of the per-position slopes to $\mu$.

### 6.4 Ratio to a frozen initial predictor copy (readout only)

- **Change.** Store $g_0$ (or, since the point set is fixed, just the 100 x 128 table of
  $g_0(x_i) - f(x_i)$) and report $\tilde b_i(n) = \|g_n(x_i)-f(x_i)\|_2 / \|g_0(x_i)-f(x_i)\|_2$.
- **Predicts.** $\tilde b_i(0) = 1$ exactly, at every position, with no assumptions. Slope unchanged.
- **Use it as.** The default readout for every experiment in this study, not as a method on its own.
  It costs nothing and removes the initial-scale confound from every slope measurement.
- **Can go wrong.** The denominator carries about 6% relative sampling spread at $d = 128$ (about 35%
  at $d = 4$), which appears as a fixed per-position offset in the log-log intercept. Reduce it by
  averaging the denominator over several predictor initializations, or replace it with the analytic
  kernel diagonal (6.6).

### 6.5 Zero-initialized predictor head with a unit-normalized target

- **Change.** Set the predictor's final linear layer weights and bias to zero; replace the target by
  $\hat f(x) = f(x)/\max(\|f(x)\|_2,\varepsilon)$.
- **Predicts.** $b_i(0) = 1$ exactly at every position, with **zero** estimator noise — strictly
  better than 6.4 on requirement (1). Slope unchanged.
- **Extra property worth measuring.** The unit-normalized target realizes the correlation kernel
  $k(x,x')/\sqrt{k(x,x)k(x',x')}$, whose diagonal is 1 everywhere. So this construction also
  equalizes the *prior variance* over the input space, not just the realized initial residual — which
  means it also equalizes the noiseless posterior variance at $n=0$, and is therefore the right
  starting point for methods 6.1 and 6.2.
- **Can go wrong.** With the head exactly at zero the layers below receive zero gradient at the first
  step, so the model is effectively a linear head on frozen random features until the head becomes
  nonzero. Decide deliberately whether that is wanted; if not, initialize the head at a small scale
  $\epsilon_0$ and accept $b_i(0) = 1 + O(\epsilon_0)$.

### 6.6 Analytic prior-kernel normalization

- **Change.** Divide the raw residual norm by $\sqrt{d\,(k_g+k_f)(x_i,x_i)}$, computed in closed form
  for the one-hidden-layer ReLU architecture and its initialization (an arc-cosine expression in
  $\|x_i\|_2$ plus the bias variance), or estimated by Monte Carlo over many independent draws.
- **Predicts.** The *expected* initial bonus is exactly 1 at every position; the realized value keeps
  its $\chi_d$ fluctuation, so the plot shows honestly how much of the observed initial spread is
  systematic and how much is sampling.
- **Use it as.** The diagnostic that separates the two contributions in Section 1.4, before choosing
  between 6.4 and 6.5.

### 6.7 DRND's construction, run directly in this setting

- **Change.** Draw $N$ target networks. On each visit to $x_i$, regress on a uniformly chosen one.
  Report the normalized statistic
  $\sqrt{\dfrac{[g_n(x_i)]^2-[\mu(x_i)]^2}{B_2(x_i)-[\mu(x_i)]^2}}$ with $\mu, B_2$ the first and
  second moments over the $N$ targets at $x_i$.
- **Predicts.** By their Lemma 4.3 the quantity under the square root is an unbiased estimator of
  $1/n_i$, so the statistic itself is approximately $n_i^{-1/2}$ per position and equals 1 at
  $n_i = 1$. This is 6.1 with a discrete target distribution instead of a Gaussian one.
- **Why 6.1 is preferable here.** RDD's Corollary 4.2 shows the Gaussian version has strictly smaller
  variance at the same mean, and the Gaussian version has no $N$ to tune. Run DRND's version as the
  published-baseline arm.

---

## 7. What to measure first

1. **Confirm the diagnosis of the initial field.** Average $\|g_0(x_i)-f(x_i)\|_2^2$ over many seeds
   and regress it on $\|x_i\|_2^2$. The prediction from Section 1.4 is a straight line with a
   positive intercept set by the bias variance. Report the ratio of the largest to the smallest fitted
   value — that is the size of the problem construction (a) or (b) removes.
2. **Confirm the diagnosis of the slope heterogeneity.** Compute the 100 x 100 Gram matrix (last-layer
   features, or empirical neural tangent kernel) and its eigenvalues $\lambda_j$. Under the $c/n$
   schedule, the model in Section 5.1 predicts per-position slopes lying inside
   $[-c\lambda_{\max}, -c\lambda_{\min}]$ and drifting toward $-c\lambda_{\min}$ as $n$ grows. Check
   the observed spread against the spectrum.
3. **Then run the two mechanisms that actually carry a count** — 6.1 and 6.2 — with the readout of
   6.5, and compare per-position slopes on the same plot as the current $1/t$ SGD baseline.
4. **Only after uniform visitation works**, move to non-uniform $p_i$ and check that the bonus at
   position $i$ tracks $n_i^{-1/2}$ rather than $n^{-1/2}$.

---

## 8. Papers referenced, with links

Every entry below was retrieved and checked during this note; the identifier given is the one found
on the paper's own page.

- Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. *Exploration by Random Network
  Distillation*. ICLR 2019. arXiv:1810.12894. <https://arxiv.org/abs/1810.12894>
- Ian Osband, John Aslanides, Albin Cassirer. *Randomized Prior Functions for Deep Reinforcement
  Learning*. NeurIPS 2018. arXiv:1806.03335. <https://arxiv.org/abs/1806.03335>
- Kai Yang, Jian Tao, Jiafei Lyu, Xiu Li. *Exploration and Anti-Exploration with Distributional
  Random Network Distillation*. ICML 2024. arXiv:2401.09750. <https://arxiv.org/abs/2401.09750>
- Zhirui Fang, Kai Yang, Jian Tao, Jiafei Lyu, Lusong Li, Li Shen, Xiu Li. *Exploration by Random
  Distribution Distillation*. arXiv:2505.11044. <https://arxiv.org/abs/2505.11044>
- Alexander Nikulin, Vladislav Kurenkov, Denis Tarasov, Sergey Kolesnikov. *Anti-Exploration by
  Random Network Distillation*. ICML 2023. arXiv:2301.13616. <https://arxiv.org/abs/2301.13616>
- Matej Pecháč, Michal Chovanec, Igor Farkaš. *Self-supervised network distillation: an effective
  approach to exploration in sparse reward environments*. arXiv:2302.11563.
  <https://arxiv.org/abs/2302.11563>
- Adrià Puigdomènech Badia et al. *Never Give Up: Learning Directed Exploration Strategies*.
  ICLR 2020. arXiv:2002.06038. <https://arxiv.org/abs/2002.06038>
- Kamil Ciosek, Vincent Fortuin, Ryota Tomioka, Katja Hofmann, Richard E. Turner. *Conservative
  Uncertainty Estimation By Fitting Prior Networks*. ICLR 2020.
  <https://openreview.net/forum?id=BJlahxHYDS>
- Moritz A. Zanger, Yijun Wu, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T. J. Spaan. *On the
  Equivalence of Random Network Distillation, Deep Ensembles, and Bayesian Inference*.
  arXiv:2602.19964. <https://arxiv.org/abs/2602.19964>
- Moritz A. Zanger, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T. J. Spaan. *Contextual
  Similarity Distillation: Ensemble Uncertainties with a Single Model*. arXiv:2503.11339.
  <https://arxiv.org/abs/2503.11339>
- Moritz A. Zanger, Max Weltevrede, Yaniv Oren, Pascal R. Van der Vaart, Caroline Horsch, Wendelin
  Böhmer, Matthijs T. J. Spaan. *Universal Value-Function Uncertainties*. arXiv:2505.21119.
  <https://arxiv.org/abs/2505.21119>
- Abdul Wahab, Raksha Kumaraswamy, Martha White. *Value Bonuses using Ensemble Errors for Exploration
  in Reinforcement Learning*. Reinforcement Learning Journal / RLC 2025. arXiv:2602.12375.
  <https://rlj.cs.umass.edu/2025/papers/Paper201.html>
- Mikael Henaff, Roberta Raileanu, Minqi Jiang, Tim Rocktäschel. *Exploration via Elliptical Episodic
  Bonuses*. NeurIPS 2022.
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/f4f79698d48bdc1a6dec20583724182b-Abstract-Conference.html>
- Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Rémi Munos.
  *Unifying Count-Based Exploration and Intrinsic Motivation*. NeurIPS 2016. arXiv:1606.01868.
  <https://arxiv.org/abs/1606.01868>
- Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, Jascha
  Sohl-Dickstein. *Deep Neural Networks as Gaussian Processes*. ICLR 2018. arXiv:1711.00165.
  <https://arxiv.org/abs/1711.00165>
- Youngmin Cho, Lawrence K. Saul. *Kernel Methods for Deep Learning*. NIPS 2009.
  <http://papers.neurips.cc/paper/3628-kernel-methods-for-deep-learning.pdf>
- Bobby He, Balaji Lakshminarayanan, Yee Whye Teh. *Bayesian Deep Ensembles via the Neural Tangent
  Kernel*. NeurIPS 2020. arXiv:2007.05864. <https://arxiv.org/abs/2007.05864>
- Boris T. Polyak, Anatoli B. Juditsky. *Acceleration of Stochastic Approximation by Averaging*.
  SIAM Journal on Control and Optimization 30(4):838-855, 1992.
  <https://epubs.siam.org/doi/10.1137/0330046>

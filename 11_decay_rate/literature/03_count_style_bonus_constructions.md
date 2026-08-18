# Exploration bonuses that decay like one over the square root of the count by construction

Notes for the `11_decay_rate` campaign. Topic: constructions in which the bonus decays as
$n^{-1/2}$ because of how the learning problem is set up, not because an optimizer happened to
converge at that speed. The coin-flip / random-target-averaging family is the centre of these
notes.

Everything below states, for each decay rate, **which quantity** decays (a squared loss, a
squared residual, or a residual norm) and **under what assumptions**.

Conventions used throughout:

- $x_i$ is one of the roughly one hundred fixed maze positions, presented to the network as
  $[x, y, 0, 0]$.
- $n_i$ is the number of times position $i$ has been trained on. In the uniform full-batch
  regime $n_i = n$ for every $i$, where $n$ is the number of optimizer steps.
- $g_n$ is the trained predictor after $n$ steps, $f$ the frozen random target.
- $b_i(n)$ is the per-position bonus. In plain Random Network Distillation,
  $b_i(n) = \lVert g_n(x_i) - f(x_i) \rVert_2$.
- $d$ is the output dimension of the predictor (128 in the current setup).

---

## 1. The one distinction that organises this whole literature

Two different mechanisms can make a residual shrink, and only one of them can produce a rate
that is the same at every position.

**Mechanism one — optimization convergence (this is what plain Random Network Distillation
uses).** The target $f(x_i)$ is a fixed number. The residual shrinks only because gradient
descent is converging. In the linearised (neural tangent kernel) regime, with $K$ the tangent
kernel Gram matrix over the fixed point set and $\eta_n$ the step size at step $n$, the
residual vector $r_n$ obeys $r_n = (I - \eta_n K)\, r_{n-1}$. Writing $r_n$ in the eigenbasis
of $K$ with eigenvalues $\lambda_j$:

- constant step size $\eta$: the component along eigenvalue $\lambda_j$ decays like
  $(1 - \eta \lambda_j)^n$, that is, geometrically, with a **different geometric rate per
  eigen-direction**. Fast directions vanish quickly, the residual then sits on the slowest
  direction and looks saturated. This is exactly the reported "constant-rate Adam, slope about
  $-0.8$, saturating".
- step size $\eta_n = a/n$: $\log r_n \approx -a\lambda_j \sum_{k \le n} 1/k \approx
  -a \lambda_j \log n$, so the component decays like $n^{-a\lambda_j}$. **The exponent is
  proportional to the eigenvalue.** Tuning $a$ so that the typical $a\lambda_j$ is $1/2$ gives
  an aggregate slope near $-0.5$ (the reported $-0.503$) while individual positions, which are
  different mixtures of eigen-directions, get different slopes and slopes that drift with $n$
  as the mixture shifts towards the smallest eigenvalue.

So the observed per-position heterogeneity is not an accident of tuning. With a deterministic
target the exponent is set by the local curvature of the fit problem, and the curvature is not
the same at every position.

**Mechanism two — statistical averaging of fresh random targets (the coin-flip family).** The
target presented at position $i$ on visit $k$ is a **fresh random draw** $c_i^{(k)}$ with mean
$\mu$ and per-coordinate variance $\sigma^2$. The least-squares optimum for position $i$ after
$n_i$ visits is the running mean of that position's own draws,
$m_i^{(n_i)} = \frac{1}{n_i}\sum_{k \le n_i} c_i^{(k)}$, and

$$\mathbb{E}\big[\lVert m_i^{(n_i)} - \mu \rVert_2^2\big] = \dfrac{d\,\sigma^2}{n_i}
\qquad\Longrightarrow\qquad
\lVert m_i^{(n_i)} - \mu \rVert_2 \;\approx\; \sigma\sqrt{d}\; n_i^{-1/2}.$$

Here the decaying quantity is the **residual norm**, the exponent is $-1/2$ for **every**
position, and the constant $\sigma\sqrt{d}$ is also the same for every position. The exponent
does not depend on curvature, on the learning rate constant, or on how many other positions
there are. It depends only on that position's own visit count, which is exactly what the
non-uniform-visitation extension needs.

Everything in section 2 is a variation on mechanism two, or a way of computing the same
quantity in closed form (the elliptical family, section 4).

---

## 2. The coin-flip / random-target-averaging family

### 2.1 Lobel, Bagaria, Konidaris — "Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning" (ICML 2023, arXiv:2306.03186)

This is the direct answer to question 1 and the closest published method to what the campaign
needs.

**What it does.** Every time a state $s$ is observed, draw a fresh vector
$c \in \{-1, +1\}^d$ with independent uniform coordinates (each coordinate a Rademacher draw,
a coin flip) and store the pair $(s, c)$ in a buffer. Train a network $f_\phi$ by ordinary
mean-squared regression from states to their stored coin vectors.

**Why the magnitude of the learned output behaves like one over the square root of the count.**
The paper's chain is three short steps.

1. For one coordinate: let $z_n$ be the average of $n$ independent coin flips. Then
   $\mathbb{E}[z_n] = 0$ and, because the variance of a sample mean is the variance divided by
   the sample size, $\mathbb{E}[z_n^2] = 1/n$. The paper states this as
   $\mathcal{M}_2(z_n) = \mathbb{E}[z_n^2] = 1/n$.
2. The regression objective is
   $f_\phi^\ast = \arg\min_\phi \sum_{i=1}^{|\mathcal{D}|} \sum_{j=1}^{d}
   \big(c_{ij} - f_\phi(s_i)_j\big)^2$. Because squared error is minimised by the conditional
   mean, and the coin draws attached to repeated occurrences of the same state are independent,
   the minimiser at a state seen $n$ times is that state's own running mean of coin vectors:
   $f_\phi^\ast(s) = \frac{1}{n}\sum_{k=1}^{n} c_k$.
3. Combining, $\mathbb{E}\big[\frac{1}{d}\lVert f_\phi^\ast(s)\rVert_2^2\big] = 1/n$, so the
   readout used as the bonus is
   $$B(s) := \sqrt{\dfrac{1}{d}\lVert f_\phi(s)\rVert_2^2} \;\approx\; \dfrac{1}{\sqrt{N(s)}}.$$

**Exactly which quantity decays, and under which assumptions.** The *expected squared* readout
equals $1/n$ exactly, given: (a) the regression reaches its least-squares optimum, (b) the
coin draws are independent across visits, (c) the function class can represent a distinct value
per state (no forced sharing across states). The readout $B(s)$ itself decays like $n^{-1/2}$
up to the fluctuation of a chi-type variable; see the variance below. Nothing here requires a
learning-rate schedule, a particular optimizer, or a particular architecture — that is the
paper's stated selling point over pseudo-count methods built on density models.

**Variance, and why coin flips rather than Gaussians.** The paper's appendix gives, for a
zero-mean unit-variance target distribution $X$ and one output coordinate,

$$\mathrm{Var}[z_n^2] = \dfrac{1}{n^3}\mathbb{E}[X^4] + \dfrac{2}{n^2} - \dfrac{3}{n^3}.$$

The fourth moment is minimised over zero-mean unit-variance distributions by the two-point
distribution on $\{-1,+1\}$, where $\mathbb{E}[X^4] = 1$, giving
$\mathrm{Var}[z_n^2] = \frac{2}{n^2} - \frac{2}{n^3}$. Averaging $d$ independent output
coordinates divides the variance by $d$:
$\mathrm{Var}\big[\frac{1}{d}\sum_{j=1}^d z_{n,j}^2\big] = \frac{1}{d}\mathrm{Var}[z_n^2]$.

Consequences that matter directly for the campaign's `dev_worst` metric:

- The relative standard deviation of the squared readout is about $\sqrt{2/d}$ for large $n$.
  With $d = 128$ that is about $12.5$ percent on $b^2$, so about $6.3$ percent on $b$, that is
  about $0.061$ in natural log. That is a **floor on the per-position deviation** from a single
  seed's curve that no method in this family can beat without variance reduction. Raising $d$
  to 512 halves it to about $0.031$.
- Rademacher targets have another property the paper does not stress but the campaign needs:
  at $n = 1$ the average is the single draw itself, whose norm is $\sqrt{d}$ **exactly**, with
  zero variance, because every coordinate is $\pm 1$. So $B(s) = 1$ exactly on the first visit,
  at every position. Gaussian targets give a chi-distributed norm at $n=1$ and would not.

**Implementation details from the paper, for reproduction in a small network.**

- Architecture: "the same neural network architecture as RND's prediction network". No target
  network exists in this method; the coin vector replaces it.
- Output dimension: $d = 20$ in all their experiments. No ablation over $d$ is reported.
- Loss: mean squared error, $L(s_i, c_i) = \lVert c_i - f_\phi(s_i)\rVert_2^2$.
- Optimizer: Adam. Learning rate $10^{-4}$ for gridworld, Fetch, Adroit, Ant; $10^{-4}$ or
  $10^{-5}$ for Montezuma's Revenge (grid searched). Adam epsilon $1.5\times 10^{-4}$ on Atari,
  $10^{-8}$ on continuous control.
- Separate replay buffer for the counting network, distinct from the agent's buffer: $10^6$
  entries on gridworld/Fetch/Adroit, $2\times 10^7$ on Montezuma's Revenge. Batch size 512 to
  1024. One counting-network update per environment step in most tasks.
- **Optimistic prior** (their section 3.5.2): the network is written as
  $f_\phi(s) = \hat f_\phi(s) + f_{\text{prior}}(s)$, where $f_{\text{prior}}$ is frozen and
  randomly initialised and is normalised so that
  $\mathbb{E}_{s \sim \mathcal{D}}\big[f_{\text{prior}}^{(j)}(s)^2\big] = 1$ per output
  coordinate. A state the network has never trained on then gets a readout of about 1, that is,
  a pseudocount of about 1, instead of whatever the untrained network happens to output. This
  is precisely the mechanism the campaign needs for requirement (1), "start at one everywhere".
- **Prioritised sampling** (their section 3.5): the counting network's buffer is sampled with
  priority
  $\mathrm{priority}(s) = \alpha \big(1/n_{\text{updates}}(s)\big)
  + (1-\alpha)\big(\frac{1}{d}\lVert f_\phi(s)\rVert_2^2\big)$ with $\alpha = 0.5$, so recently
  added and apparently novel states are trained on more. (Formula transcribed from a machine
  reading of the paper; check against the PDF before relying on the exact form.)
- Reward normalisation by a running mean and variance is used on the visual gridworld only.
- Bonus in the reinforcement-learning loop:
  $y_t = R_t + \lambda B(s_t) + \gamma \max_{a'} Q(s_{t+1}, a')$ with $\lambda$ between
  $10^{-3}$ and $0.03$ depending on the task.

**Guarantees.** There are no regret bounds and no formal propositions about the neural case.
What is proved: unbiasedness of the squared readout as an estimator of $1/n$ for states in the
training data; minimum variance of the Rademacher choice among zero-mean unit-variance target
distributions; and, for a linear predictor, an appendix computing the readout at a new state as
a combination of the inverse counts of the training states along the singular directions of the
design.

**Failure modes the paper reports or implies.**

- For a state the network has not trained on, the readout is whatever the network generalises
  to; the paper notes it "tends to be close to 0" when the state is unlike the training data,
  which reads as "already visited many times". The optimistic prior is the patch, and it is not
  theoretically grounded.
- The count is a count over the buffer, not over all history. When high-count states are evicted
  from a finite buffer they reappear as novel. Their patch is a very large buffer.
- The regression must actually reach its optimum. Every statement above is about
  $f_\phi^\ast$, not about the iterate an optimizer produces after a finite number of steps.
  In a non-stationary buffer with a fixed learning rate this is an approximation of unknown
  quality — the paper does not quantify it.
- Generalisation between nearby states mixes their coin means. Two nearby positions with
  independent coin sequences have means that partially average; the averaged mean has a smaller
  norm. Under uniform visitation this changes the constant, not the exponent (see section 6).
  Under non-uniform visitation it mixes different counts, which is a genuine bias.

**Comparison against Random Network Distillation reported in the paper.** Their figures 2 and 7
state that the Random Network Distillation bonus "falls off more sharply than $1/\sqrt{N(s)}$"
at high counts and that its decay rate is governed by neural-network learning dynamics rather
than by any count. This is the same observation the campaign started from.

**Downstream use, confirming the construction transfers.** Bai, Zhang, Qiu, Zhang, Xu, Li,
"Online Preference Alignment for Language Models via Count-based Exploration" (ICLR 2025,
arXiv:2501.12735) uses "a simple coin-flip counting module to estimate the pseudo-count of a
prompt-response pair in previously collected data" inside an upper-confidence-bound term for
online preference optimisation.

### 2.2 Yang, Tao, Lyu, Li — "Exploration and Anti-Exploration with Distributional Random Network Distillation" (ICML 2024, arXiv:2401.09750)

The same averaging mechanism, but the random target is drawn from a **finite** set of frozen
target networks instead of from a coin distribution — which makes this the smallest possible
edit to an existing Random Network Distillation implementation.

- Setup: $N$ frozen randomly initialised target networks $\bar f_1, \dots, \bar f_N$
  ($N = 10$). At each occurrence of an input $x$, one index is drawn uniformly and the target
  for that occurrence is $c(x) = \bar f_{\text{index}}(x)$. Loss
  $L(\theta) = \lVert f_\theta(x) - c(x) \rVert_2^2$.
- Define $\mu(x) = \frac1N \sum_i \bar f_i(x)$ and $B_2(x) = \frac1N \sum_i \bar f_i(x)^2$
  (coordinatewise).
- First bonus term, addressing what they call the initial inconsistency:
  $b_1(x) = \lVert f_\theta(x) - \mu(x)\rVert_2^2$.
- Second bonus term, the count-like one:
  $$b_2(x) = \sqrt{\dfrac{\,[f_\theta(x)]^2 - [\mu(x)]^2\,}{\,B_2(x) - [\mu(x)]^2\,}}.$$
  (Transcribed from a machine reading; verify the bracketing against the PDF.)
- **Lemma 4.3**: the quantity inside the square root, evaluated at the optimal predictor
  $f^\ast$, is an **unbiased and consistent estimator of $1/n$**, where $n$ is the number of
  times $x$ has been visited. So $b_2$ estimates $n^{-1/2}$ — the *squared* readout is the
  unbiased $1/n$ object, the readout itself is the $n^{-1/2}$ object.
- Assumptions of that lemma: the predictor reaches the optimum
  $f^\ast(x) = \frac{1}{n}\sum_{k=1}^n c_k(x)$; the target index is drawn uniformly and
  independently at each visit; the target networks are independent draws from a common
  distribution. The variance of the estimator goes to zero as $n$ grows.
- Combination used: $b(x) = 0.9\, b_1(x) + 0.1\, b_2(x)$.
- Hyperparameters: $N = 10$; hidden width 64 (Gym), 256 (offline), 512 (Atari); ReLU; Adam;
  learning rate $3\times 10^{-4}$ online Gym, $10^{-4}$ Atari, $10^{-6}$ for the distillation
  network in the offline setting.
- Their stated diagnosis of plain Random Network Distillation is "bonus inconsistency": at the
  start, bonuses deviate considerably from any sensible distribution across states; at the end,
  frequently and rarely visited states receive indistinguishable bonuses.

The important structural difference from coin flips: with $N = 10$ discrete targets, the target
distribution's variance $B_2 - \mu^2$ is **input-dependent**, so the readout has to be divided
by it to recover a count. Coin flips have variance exactly 1 everywhere and need no such
normalisation. For this campaign, coin flips are the cleaner choice; the finite-target version
is worth knowing because it keeps the frozen-random-network flavour of Random Network
Distillation intact.

### 2.3 Fang, Yang, Tao, Lyu, Li, Shen, Li — "Exploration by Random Distribution Distillation" (arXiv:2505.11044, May 2025)

The Gaussian version of the same idea, and the paper that states the decomposition most
usefully for the campaign.

- The target network outputs a **distribution** rather than a value:
  $f_{\text{tar}}(s) \sim \mathcal{N}\big(\mu_{\bar\theta}(s), \sigma^2_{\bar\varphi}(s)\big)$
  with $\bar\theta, \bar\varphi$ frozen random parameters. The predictor is trained by mean
  squared error against a sample from that distribution.
- Readout: $b(s) = \frac{1}{d}\lVert f_\theta(s) - \mu_{\bar\theta}(s)\rVert_2^2$ — that is,
  the residual is measured against the target distribution's **mean**, not against the drawn
  sample.
- Their bound, by a triangle inequality:
  $$\lVert f_\theta(s) - \mu_{\bar\theta}(s)\rVert_2^2 \;\le\;
  \dfrac{\sigma^2_{\bar\varphi}(s)}{n} \;+\;
  \mathbb{E}\big[\lVert f^\ast_{\theta_n}(s) - f_\theta(s)\rVert_2^2\big].$$
  The first term is the count term, decaying as $1/n$ in the **squared** residual (so
  $n^{-1/2}$ in the residual norm). The second term is the gap between the current iterate and
  the least-squares optimum — the optimization error.
- This decomposition is the right way to instrument the campaign's experiments: measure the
  statistical part and the optimization part separately, and require the second to be small
  relative to the first at every checkpoint (see section 7).
- Implementation: hidden width 64 (512 on Atari), output dimension 64 (512 on Atari), ReLU,
  Adam, learning rate $3\times 10^{-4}$ ($10^{-4}$ on Atari), target mean 1 and target standard
  deviation 1, bonus coefficient 1.
- They state plainly that in plain Random Network Distillation "the rate at which the reward
  diminishes with repeated visits is not well defined, as it depends on the learning dynamics
  of the neural network and stochastic gradient descent", and that different states with the
  same number of visits receive noticeably different Random Network Distillation rewards. That
  is the campaign's problem statement, taken from an independent source.

### 2.4 Where the three sit relative to each other

| method | target at each visit | readout | quantity that is exactly $1/n$ | extra normalisation needed |
|---|---|---|---|---|
| Coin flip network (Lobel 2023) | fresh $\pm 1$ vector | $\sqrt{\frac1d \lVert f_\phi\rVert_2^2}$ | expected squared readout | none — target variance is 1 by construction |
| Distributional Random Network Distillation (Yang 2024) | one of $N$ frozen nets, uniform | $\sqrt{\frac{f_\theta^2 - \mu^2}{B_2 - \mu^2}}$ | the ratio inside the root | divide by input-dependent target variance |
| Random distribution distillation (Fang 2025) | Gaussian sample around a frozen mean | $\frac1d\lVert f_\theta - \mu\rVert_2^2$ | bounded above by $\sigma^2/n$ plus optimization error | divide by $\sigma^2$ if it varies |

---

## 3. Pseudo-count methods built on density models (question 2)

### 3.1 Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos — "Unifying Count-Based Exploration and Intrinsic Motivation" (NeurIPS 2016)

- A sequential density model $\rho_n(x)$ assigns a probability to $x$ after seeing
  $x_{1:n}$. The **recoding probability** $\rho'_n(x) := \rho(x; x_{1:n}x)$ is the probability
  the model would assign to $x$ after being trained once more on $x$ itself.
- The **pseudo-count** is defined so that the model behaves like an empirical frequency count
  under one more observation:
  $$\hat N_n(x) = \dfrac{\rho_n(x)\,\big(1 - \rho'_n(x)\big)}{\rho'_n(x) - \rho_n(x)}.$$
- **Prediction gain** is $PG_n(x) = \log \rho'_n(x) - \log \rho_n(x)$. The approximate relation
  is $\hat N_n(x) \approx \big(e^{PG_n(x)} - 1\big)^{-1}$, with equality in the limit
  $\rho'_n(x)\to 0$.
- Their Theorem 1 (for a learning-positive mixture model, meaning
  $\rho'_n(x) \ge \rho_n(x)$ always) chains information gain, prediction gain and the
  pseudo-count: $IG_n(x) \le PG_n(x) \le \hat N_n(x)^{-1}$ and
  $PG_n(x) \le \hat N_n(x)^{-1/2}$.
- Their Theorem 2 gives the asymptotic statement: the ratio $\hat N_n(x)/N_n(x)$ of pseudo-count
  to real count converges to a constant, that is, **pseudo-counts grow linearly on average with
  real counts**. This is an asymptotic, in-expectation statement about the count, not a
  per-state finite-$n$ statement about the bonus.
- Exploration bonus used in the Atari experiments:
  $R^+_n(x, a) := \beta\,\big(\hat N_n(x) + 0.01\big)^{-1/2}$ with $\beta = 0.05$. So the
  $n^{-1/2}$ shape is **imposed by the formula**; what the density model supplies is the count,
  not the rate.

**Precision actually demonstrated.** Linear growth of pseudo-count with real count, on average,
plus qualitative agreement on Atari frames. There is no per-state demonstration that the bonus
follows $n^{-1/2}$ with a controlled error, which is exactly the standard the campaign is
holding itself to.

### 3.2 Ostrovski, Bellemare, van den Oord, Munos — "Count-Based Exploration with Neural Density Models" (ICML 2017, arXiv:1703.01310)

The most directly instructive paper of the two for the campaign, because it is about the
mismatch between what the count theory needs and what a neural optimizer does.

- They replace the CTS density model with PixelCNN and keep the pseudo-count-from-prediction-gain
  machinery.
- The pseudo-count they actually use is
  $$\hat N_n(x) = \Big(\exp\big(c\, n^{-1/2}\, (PG_n(x))_+\big) - 1\Big)^{-1},$$
  with $c = 0.1$, $n$ the number of model updates, and $(\cdot)_+$ clipping negative prediction
  gain to zero.
- The reason for the extra factor $c\,n^{-1/2}$, in their words: "the theory of pseudo-counts
  requires the density model's rate of learning to decay over time. Optimization of a neural
  model, however, imposes constraints on the step-size regime which cannot be violated." They
  found empirically that a **constant** learning rate of $10^{-3}$ gave the lowest loss for
  PixelCNN, so the model's own learning rate could not be made to decay as the theory wants;
  they therefore injected a separate decay schedule into the pseudo-count formula.
- Exploration bonus: $r^+(x) := (\hat N_n(x))^{-1/2}$.

**Why this matters here.** This is a published precedent for exactly the temptation the
campaign's rules forbid: when the network would not produce the required decay, the authors
multiplied the readout by an explicit function of the training step. The campaign's rule "the
optimizer may know the time, the readout may not" is the line this paper crosses. It is a good
argument for the coin-flip family, where the rate comes from the statistics of the targets and
no such factor is needed anywhere.

---

## 4. Elliptical bonuses (question 3): the same quantity, computed in closed form

### 4.1 Abbasi-Yadkori, Pál, Szepesvári — "Improved Algorithms for Linear Stochastic Bandits" (NeurIPS 2011)

The self-normalised confidence set for linear regression. With features $\phi_\tau$ observed up
to time $t$ and $\Lambda_t = \lambda I + \sum_{\tau \le t} \phi_\tau \phi_\tau^{\!\top}$, the
width of the confidence interval at a new feature $\phi$ is proportional to
$\sqrt{\phi^{\!\top} \Lambda_t^{-1} \phi}$.

For the campaign's setting the special case is the one that matters. If the features of the
$\sim 100$ positions are **orthonormal** (one-hot, or any whitened basis), then
$\Lambda_t = \lambda I + \sum_i n_i\, \phi_i \phi_i^{\!\top}$ is diagonal in that basis, and

$$\sqrt{\phi_i^{\!\top}\Lambda_t^{-1}\phi_i} \;=\; \dfrac{1}{\sqrt{\lambda + n_i}}.$$

With $\lambda = 1$ this is exactly 1 at $n_i = 0$ and decays exactly as $n_i^{-1/2}$, at every
position simultaneously, and it uses each position's **own** count, so non-uniform visitation is
handled with no modification at all. This is the exact object the campaign is trying to make a
network produce, and it is worth running as a reference curve even if it is not the final method.

### 4.2 Henaff, Raileanu, Jiang, Rocktäschel — "Exploration via Elliptical Episodic Bonuses" (NeurIPS 2022, arXiv:2210.05805)

- Bonus: $b(s_t) = \phi(s_t)^{\!\top} C_{t-1}^{-1}\phi(s_t)$ with
  $C_{t-1} = \sum_{i<t} \phi(s_i)\phi(s_i)^{\!\top} + \lambda I$, the sum taken **within the
  current episode**.
- Note this is the **squared** elliptical norm, not the square root of it. In the one-hot case
  the paper states the bonus becomes $1/N_e(s_t)$ — a $1/n$ bonus, not $n^{-1/2}$. If the
  campaign borrows this, take the square root.
- The embedding $\phi$ is learned by an inverse dynamics model: minimise
  $-\log p\big(a_t \mid g(\phi(s_t), \phi(s_{t+1}))\big)$, so that features keep only what
  predicts the agent's own actions.
- **What is known about learned-feature versions**: very little that is quantitative. The paper
  makes the tabular connection and then relies on empirical results; it does not analyse what
  happens when $\phi$ drifts while $C$ is being accumulated from older features. That is the
  open problem in this family. For the campaign, the safe version is **frozen** features (random
  or pretrained), where the count semantics is exact.

### 4.3 Ash, Zhang, Goel, Krishnamurthy, Kakade — "Anti-Concentrated Confidence Bonuses for Scalable Exploration" (ICLR 2022, arXiv:2110.11202)

This is the bridge between the coin-flip family and the elliptical family, and it deserves more
attention than it usually gets in this context.

- Construction: maintain $M$ regressors, each trained by least squares to predict **independent
  standard Gaussian noise** from the observed features. For regressor $j$, minimise
  $\sum_\tau \big(\langle x_{\tau,a_\tau}, w\rangle - y^{(j)}_\tau\big)^2$ with
  $y^{(j)}_\tau \sim \mathcal{N}(0,1)$ drawn fresh per example per regressor.
- Bonus: $\beta \max_{j \in [M]} \big|\langle x_{t,a}, w^{(j)}_t\rangle\big|$.
- The key fact (their Lemma 5 and surrounding argument): the fitted prediction on random noise,
  $\langle w^{(j)}_t, x\rangle$, is distributed as
  $\mathcal{N}\big(0, \lVert x\rVert^2_{\tilde\Sigma_t^{-1}}\big)$. **The standard deviation of
  a least-squares fit to fresh noise is exactly the elliptical bonus.** Taking a maximum over
  $M = \Theta(\log(T/\delta))$ regressors turns that into a valid optimistic confidence width,
  bounded above and below by constants times $\lVert x \rVert_{\tilde\Sigma^{-1}}$. This yields
  $\tilde O(d\sqrt{T})$ regret for stochastic linear bandits with $\mathrm{poly}(d)$ fixed
  actions.
- Deep variant: features are gradient features
  $g(x;f,\theta) = \partial_\theta \ell_{ce}(f(x,\theta), \hat y)$ rather than penultimate-layer
  activations; $M = 128$ regressors; ridge $\lambda = 10^3$; RMSProp for the auxiliary weights;
  batch normalisation on the gradient features; the bonus normalised by a running standard
  deviation; the auxiliary policy network tail-averaged with $\alpha = 10^{-6}$; policy updated
  every $128^2$ steps.

**Why this matters for the campaign.** It says that "regress a network onto fresh random targets
and read out the magnitude of the fit" and "compute
$\sqrt{\phi^{\!\top}\Lambda^{-1}\phi}$ in the network's own feature space" are the *same
quantity*, one estimated by sampling and one in closed form. So the coin-flip method applied to
the campaign's 4-256-128 network is, in the linearised regime, computing the elliptical bonus in
the tangent-kernel feature space of that network. That gives a clean prediction to test: the
coin-flip readout should track $\sqrt{\phi_i^{\!\top}\Lambda_n^{-1}\phi_i}$ computed from the
network's own tangent features, and where the two disagree the cause is optimization error.

---

## 5. Analyses of Random Network Distillation itself (question 4)

### 5.1 Burda, Edwards, Storkey, Klimov — "Exploration by Random Network Distillation" (ICLR 2019, arXiv:1810.12894)

The original method. The bonus is the prediction error of a network trained to match a fixed
randomly initialised network on the observations seen so far. The paper offers an intuition
(prediction error is large off the training distribution) but **no claim at all about a decay
rate**, and no count interpretation. Everything about how the bonus decays with revisits is left
to the optimizer.

### 5.2 Zanger, Wu, Van der Vaart, Böhmer, Spaan — "On the Equivalence of Random Network Distillation, Deep Ensembles, and Bayesian Inference" (arXiv:2602.19964, February 2026)

The most complete theoretical account of what the Random Network Distillation residual is.

- Regime: infinite-width limit under neural-tangent-kernel parametrisation, gradient flow.
- Theorem 3.1: the converged Random Network Distillation error is Gaussian with zero mean and a
  covariance built from the network's neural-network-Gaussian-process kernel and its tangent
  kernel. The training dynamics obey
  $\mathrm{d}\epsilon/\mathrm{d}t = -\Theta_t(x, \mathcal{X})\,\epsilon$, with $\Theta$ the
  tangent kernel.
- Corollary 3.2: the expected squared Random Network Distillation error **equals the predictive
  variance of an ensemble** of independently initialised networks trained on the same data.
- Theorem 3.4: multi-headed Random Network Distillation errors match a finite ensemble's variance
  in distribution, both scaling as chi-squared variables.
- Theorem 4.2: with a specifically engineered target function the error distribution matches the
  exact Bayesian posterior predictive with the tangent kernel as prior.

**What it does and does not say about our question.** It does not give a decay rate in the number
of repeated occurrences of a training point. What it does say is decisive anyway: the converged
error at a point is a fixed function of the kernel and the training set — it depends on **which**
points are in the training set, not on **how many times** each appears. Fitting the same fixed
target on the same fixed 100 points forever converges to a fixed residual pattern; repetitions
add no information. The only reason the residual keeps shrinking in practice is that the
optimizer has not converged. That is a structural argument that plain Random Network Distillation
cannot produce a count-like decay by construction, no matter which optimizer is used, and it is
the strongest single reason to change the target rather than the optimizer.

### 5.3 Related uncertainty results worth knowing

- **Osband, Aslanides, Cassirer — "Randomized Prior Functions for Deep Reinforcement Learning"
  (NeurIPS 2018).** Each ensemble member is a trainable network plus an added untrainable random
  prior function. Proved efficient with linear representations; demonstrated with nonlinear ones.
  This is where the "add a frozen random function so that uncertainty exists before any data
  arrives" idea comes from, and it is the ancestor of the coin flip network's optimistic prior.
- **Ciosek, Fortuin, Tomioka, Hofmann, Turner — "Conservative Uncertainty Estimation By Fitting
  Prior Networks" (ICLR 2020).** Fitting random prior functions gives uncertainty estimates that
  are **conservative**, in the sense that they never underestimate the posterior uncertainty of a
  hypothetical Bayesian method, and that converge to zero as more data arrives. I could not
  retrieve the theorem statements themselves (OpenReview blocked automated access), so the
  **rate** at which they converge to zero is, as far as these notes go, unverified. Do not cite a
  rate from this paper without reading the theorems.
- **He, Lakshminarayanan, Teh — "Bayesian Deep Ensembles via the Neural Tangent Kernel"
  (NeurIPS 2020).** Adding a tractable random untrainable function to each ensemble member gives
  a posterior interpretation in the infinite-width limit; without it, a deep ensemble trained
  with squared error has no Gaussian-process posterior interpretation even at infinite width.
  Relevant as the reason the frozen prior term is not cosmetic.

---

## 6. Successor-feature and hash-based counts (question 5), briefly

- **Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, De Turck, Abbeel — "#Exploration: A
  Study of Count-Based Exploration for Deep Reinforcement Learning" (NeurIPS 2017,
  arXiv:1611.04717).** Hash the state to a discrete code (SimHash on raw or autoencoded
  features), keep an exact table of counts of codes, use $\beta/\sqrt{N(\text{hash}(s))}$. The
  decay is exactly $n^{-1/2}$ **by definition**, because the count is a real integer count; all
  the difficulty moves into whether the hash puts the right states in the same bucket. Not
  usable as a campaign method (it is a lookup table keyed by the input, which the campaign's
  rules exclude), but it is the cleanest possible reference for what "correct" looks like.
- **Machado, Bellemare, Bowling — "Count-Based Exploration with the Successor Representation"
  (AAAI 2020).** The norm of the successor representation, while it is still being learned, works
  as an exploration bonus. They introduce the substochastic successor representation and show it
  implicitly counts how many times each state (or feature) has been observed. Theoretically
  grounded in the tabular case, extended heuristically with function approximation. The
  interesting part for us is that the counting is a side effect of a partially-converged
  quantity, which is the same fragile mechanism as plain Random Network Distillation, not a
  by-construction one.
- **Badia, Sprechmann, Vitvitskyi, Guo, Piot, Kapturowski, Tieleman, Arjovsky, Pritzel, Bolt,
  Blundell — "Never Give Up: Learning Directed Exploration Strategies" (ICLR 2020,
  arXiv:2002.06038).** The episodic term is a kernel pseudo-count read directly:
  $$r^{\text{episodic}}_t = \dfrac{1}{\sqrt{n(f(x_t))}} \approx
  \dfrac{1}{\sqrt{\sum_{f_i \in N_k} K\big(f(x_t), f_i\big)} + c},$$
  with $K(x,y) = \dfrac{\epsilon}{d^2(x,y)/d_m^2 + \epsilon}$, $\epsilon = 10^{-3}$, $d_m^2$ a
  running average of squared distances to the $k$ nearest neighbours, and $c = 10^{-3}$ a
  pseudo-count constant. The lifelong term is a Random Network Distillation error normalised by
  its running mean and standard deviation,
  $\alpha_t = 1 + \big(\mathrm{err}(x_t) - \mu_e\big)/\sigma_e$, combined multiplicatively as
  $r^i_t = r^{\text{episodic}}_t \cdot \min\{\max\{\alpha_t, 1\}, L\}$ with $L = 5$. The point
  for us: the $n^{-1/2}$ shape is again written into the formula and computed from a stored
  memory, and Random Network Distillation is relegated to a bounded multiplier precisely because
  its own decay is not trustworthy.

---

## 7. Synthesis: what to take into the campaign

1. **The heterogeneity is a property of the deterministic target, not of the optimizer.** With a
   fixed target, a $1/n$ learning rate gives a per-eigen-direction exponent $a\lambda_j$ that is
   proportional to the tangent-kernel eigenvalue. Positions are different mixtures of
   eigen-directions, so they get different exponents, and each position's apparent slope drifts
   with $n$ as the mixture concentrates on the smallest eigenvalue. No amount of tuning $a$ fixes
   this: $a$ shifts all exponents together.
2. **With fresh random targets the exponent stops depending on curvature.** Take the
   linearised recursion for one eigen-direction of the tangent kernel, with step size
   $\eta_n = a/n$ and fresh zero-mean targets of variance $\sigma^2$:
   $u_n = (1 - \gamma_n) u_{n-1} + \gamma_n \xi_n$, $\gamma_n = a\lambda_j / n$. The standard
   stochastic-approximation result gives, for $a\lambda_j > 1/2$,
   $$\mathrm{Var}(u_n) \;\longrightarrow\; \dfrac{(a\lambda_j)^2\,\sigma^2}{(2a\lambda_j - 1)\,n},$$
   so the residual norm decays as $n^{-1/2}$ in **every** direction with $a\lambda_j > 1/2$, and
   only the constant depends on $\lambda_j$. Constants can be normalised away; exponents cannot.
   This is the whole argument for switching families.
3. **There is a condition to check, and it is the main risk.** Directions with
   $a\lambda_j < 1/2$ are still dominated by the initial condition and decay as $n^{-a\lambda_j}$,
   which is *slower* than $n^{-1/2}$ — a new failure mode, at the flat end of the spectrum rather
   than the steep end. The requirement is $a\lambda_{\min} > 1/2$ over the tangent-kernel spectrum
   restricted to the fixed point set, while stability at the start needs
   $a\lambda_{\max}/(1+n_0) < 2$. Both are satisfiable together by offsetting the schedule,
   $\eta_n = a/(n + n_0)$, at the cost of a burn-in of length about
   $n_0 \approx \tfrac{a\lambda_{\max}}{2}$, which scales with the condition number of the tangent
   kernel on the point set. With 4096 steps in the protocol, a condition number above a few
   hundred eats the budget. **Measure the 108-by-108 tangent-kernel Gram matrix at
   initialisation and look at its spectrum before tuning anything** — it predicts both the burn-in
   and which positions will lag.
4. **A constant learning rate with fresh targets does not decay at all — it floors.** The
   recursion becomes an exponential moving average with stationary standard deviation of order
   $\sqrt{\eta\lambda}$. So the schedule still matters; what changes is that the schedule sets the
   constant and the *statistics* set the exponent, instead of the schedule setting the exponent.
5. **Rademacher targets give requirement (1) for free.** The norm of a single $\pm 1$ vector in
   $\mathbb{R}^d$ is exactly $\sqrt{d}$, with no variance, so the first-visit readout is exactly 1
   at every position. Gaussian targets do not have this property.
6. **There is a variance floor.** With $d$ output coordinates the readout's per-position relative
   standard deviation is about $\frac12\sqrt{2/d}$, so about $0.061$ in natural log at $d = 128$
   and $0.031$ at $d = 512$. Since `dev_worst` is a maximum over about 100 positions, the
   expected maximum of roughly 100 such fluctuations is around 2.5 standard deviations, so a bare
   coin-flip method has an intrinsic `dev_worst` floor of roughly $0.15$ at $d = 128$ and
   $0.08$ at $d = 512$ from sampling alone, before any optimization error. **Raise $d$**, and
   consider the balanced-coin construction in section 8 to remove this floor entirely.
7. **Generalisation between nearby positions changes the constant, not the exponent, under
   uniform visitation.** If the network's smoothing effectively averages $k$ neighbouring
   positions' independent coin means, the averaged mean has norm about $\sqrt{d/(kn)}$ — still
   $n^{-1/2}$, with a smaller constant. Under non-uniform visitation it mixes different counts and
   *does* bias the exponent, which is the honest limitation of every pseudo-count method with
   function approximation.
8. **A compliance caution.** The campaign's rules forbid a per-position lookup table keyed by the
   training-set index. Storing the running mean of each position's coin draws in a table would be
   exactly that, and it would also not generalise off the training set. The compliant version is
   the one the published methods use: draw a fresh target each visit, feed it as the regression
   target, and let the network's own optimizer perform the averaging. The table version is still
   worth computing as a **reference curve** — it is the statistical ideal, and the gap between it
   and the network's curve is exactly the optimization-error term in the random-distribution-
   distillation bound of section 2.3.

---

## 8. Concrete method ideas, ready to implement in the campaign's `method.py`

Each entry states the change, the reason to expect the target behaviour, and the specific way it
can fail. All of them keep the readout free of any explicit function of the step counter or of
visit counts.

### A. Coin-flip targets with an inverse-step-count learning rate (the primary candidate)

- Frozen prior $p$: a randomly initialised network with output in $\mathbb{R}^d$, rescaled per
  input to $\tilde p(x) = \sqrt{d}\; p(x)/\lVert p(x)\rVert_2$, so that
  $\lVert \tilde p(x)\rVert_2 = \sqrt{d}$ at every input exactly.
- Predictor: $g(x) = \hat g(x) + \tilde p(x)$, with the final layer of $\hat g$ initialised to
  zero, so $g_0 = \tilde p$.
- At step $n$, draw $c_i^{(n)}$ uniformly from $\{-1, +1\}^d$, independently for each position
  $i$ and each step. Loss: mean over positions of
  $\lVert g(x_i) - c_i^{(n)}\rVert_2^2$.
- Optimizer: plain stochastic gradient descent with $\eta_n = a/(n + n_0)$.
- Readout: $b_i(n) = \lVert g_n(x_i)\rVert_2 / \sqrt{d}$.
- Requirement (1) holds exactly: $b_i(0) = \lVert \tilde p(x_i)\rVert_2/\sqrt{d} = 1$ at every
  position, by construction, with no dependence on the initialisation draw.
- Requirement (2) holds because the least-squares optimum at position $i$ is that position's own
  running coin mean and the schedule makes the iterate track it — see section 7 items 2 and 3.
- Failure modes: (i) the tangent-kernel condition number forces a long burn-in; (ii) directions
  with $a\lambda_j < 1/2$ decay too slowly; (iii) $\hat g$ has to learn the whole of $-\tilde p$
  plus a shrinking mean, which is an order-one fitting burden that may itself be heterogeneous
  across positions — if that shows up, replace the additive prior with the normalisation in
  variant A2.
- Variant A2 (no additive prior): drop $\tilde p$, zero-initialise the head, and keep a frozen
  copy $g_0$ only for the purpose of the readout — but with a zero-initialised head $g_0 = 0$ and
  the ratio is undefined. So either keep the additive prior, or accept $b_i(0) = 0$ and report
  `start_dev` as a known failure of that variant. The additive prior is the better route.
- Variant A3: several coin draws per position per step, averaged into the step's target. This
  reduces the gradient noise without changing the fixed point, and lets a larger $a$ be used.

### B. Coin-flip targets with an exact recursive least-squares head (removes optimization error)

- Freeze the network body; let $\phi(x) \in \mathbb{R}^m$ be the 256-dimensional hidden layer
  (or a random Fourier feature map of $[x,y]$).
- Keep a linear head $W_n \in \mathbb{R}^{m \times d}$ that is the **exact** ridge solution to
  all coin draws so far, updated recursively (Sherman-Morrison, $O(m^2)$ per step). Readout
  $b_i(n) = \lVert \phi(x_i)^{\!\top} W_n \rVert_2 / \sqrt{d}$.
- Why it should work: the least-squares optimum is reached exactly at every step, so the
  optimization-error term of section 2.3 is identically zero and only the statistical term
  remains. The expected squared readout is
  $\phi_i^{\!\top}\Lambda_n^{-1}\big(\sum_k \phi_k \phi_k^{\!\top}\big)\Lambda_n^{-1}\phi_i$,
  which for small ridge is $\phi_i^{\!\top}\Lambda_n^{-1}\phi_i \approx \kappa_i/n$ with
  $\kappa_i = \phi_i^{\!\top}\big(\sum_j \phi_j\phi_j^{\!\top}\big)^{-1}\phi_i$ the leverage of
  position $i$. So the slope is exactly $-1/2$ at every position and only the intercept
  $\sqrt{\kappa_i}$ varies.
- Making the intercepts equal too: whiten the features over the fixed point set so that
  $\sum_j \phi_j \phi_j^{\!\top} = I$ and rescale each $\phi_i$ to unit norm; then
  $\kappa_i = 1$ for every $i$ when $m$ equals the number of points. Choose the ridge
  $\lambda = 1$ so that $b_i(0) = \lVert\phi_i\rVert_2/\sqrt{\lambda} = 1$ exactly.
- Failure mode: this is close to a linear method. It answers the campaign's question but says
  less about deep networks; treat it as the reference that establishes the ceiling.

### C. Elliptical bonus on frozen features, with no random targets at all

- $b_i(n) = \sqrt{\phi(x_i)^{\!\top}\Lambda_n^{-1}\phi(x_i)}$ with
  $\Lambda_n = \lambda I + \sum_{\text{visits}} \phi\phi^{\!\top}$, unit-norm whitened frozen
  features, $\lambda = 1$.
- With orthonormal features this is exactly $(1 + n_i)^{-1/2}$: start at 1 exactly, slope exactly
  $-1/2$ at every position, and per-position counts used automatically, so the non-uniform
  extension is solved with no change.
- This is E3B / LinUCB with frozen features, and it is the deterministic closed form of what
  method B estimates by sampling. Run it first: it is cheap, it validates the harness and the
  metric, and any method that does worse than it is losing something to optimization or to
  sampling noise, which then tells you which of the two to attack.
- Failure mode: no learning happens, so it does not answer "can a trained network do this". Also
  the features must be fixed; the campaign's rules allow it, but it sidesteps the interesting
  part.

### D. Minimal edit to the existing Random Network Distillation code (Gaussian noise on the frozen target)

- Keep the frozen target $f$ and the existing predictor $g$. Change only the training target from
  $f(x_i)$ to $f(x_i) + s\, c_i^{(n)}$ with $c_i^{(n)}$ a fresh $\pm 1$ vector (or a fresh
  Gaussian) and $s$ a fixed scale. Keep the readout $b_i(n) = \lVert g_n(x_i) - f(x_i)\rVert_2$
  divided by its value at $n = 0$.
- The least-squares optimum becomes $f(x_i) + s\, m_i^{(n)}$, so the residual is
  $s\lVert m_i^{(n)}\rVert_2 \approx s\sqrt{d}\, n^{-1/2}$.
- This is the random-distribution-distillation construction of section 2.3, and it is the single
  smallest diff from the current code.
- Tuning note: the residual is the sum of a transient (the old, heterogeneous
  optimization-driven decay from $g_0 \ne f$) and the noise floor $s\sqrt{d}\,n^{-1/2}$. Pick $s$
  large enough that the noise floor dominates from the first checkpoint, otherwise the early part
  of every curve reproduces the old heterogeneity. A concrete rule: choose $s$ so that
  $s\sqrt{d} \approx \lVert g_0(x_i) - f(x_i)\rVert_2$ at the median position.

### E. Distributional Random Network Distillation readout

- $N$ frozen target networks, one drawn uniformly per position per step as the target; readout
  $b_2$ from section 2.2, which divides by the empirical target variance $B_2 - \mu^2$ at that
  input.
- Advantage: keeps the whole method inside the "frozen random networks" idiom, which may matter
  for the writeup's continuity with the earlier chapters.
- Disadvantage relative to coin flips: the target variance is input-dependent, so the readout
  needs the extra division, and the estimator's small-$N$ bias enters. With $N = 10$ the target
  distribution has only ten atoms; the running mean over $n \gg N$ visits concentrates on a
  discrete lattice and the $1/n$ law degrades. Prefer coin flips unless there is a reason not to.

### F. Balanced (Hadamard) coin sequences — variance reduction that removes the floor of item 6

- Problem being solved: with independent coins, $\lVert m_i^{(n)}\rVert_2$ fluctuates around
  $\sqrt{d/n}$ with relative standard deviation about $\frac12\sqrt{2/d}$, and `dev_worst` takes
  the maximum over positions, so the fluctuation directly sets a floor.
- Construction: for position $i$, draw the coin at step $n$ **subject to being orthogonal to the
  running sum**. Writing $S_i^{(n)} = \sum_{k \le n} c_i^{(k)}$, one has
  $\lVert S^{(n)}\rVert_2^2 = \lVert S^{(n-1)}\rVert_2^2 + d + 2\langle S^{(n-1)}, c^{(n)}\rangle$,
  so choosing $c^{(n)} \in \{-1,+1\}^d$ with $\langle S^{(n-1)}, c^{(n)}\rangle = 0$ makes
  $\lVert S^{(n)}\rVert_2^2 = n\,d$ **exactly**, hence
  $\lVert m^{(n)}\rVert_2/\sqrt{d} = n^{-1/2}$ exactly, with zero variance, at every position.
- How to choose it: for $n \le d$ use successive rows of a $d \times d$ Hadamard matrix in an
  independently drawn random order per position — Hadamard rows are $\pm 1$ vectors and mutually
  orthogonal, so the condition holds automatically. Beyond $n = d$, pick $c^{(n)}$ by greedy
  number partitioning on the coordinates of $S^{(n-1)}$ (sort by absolute value, assign signs to
  keep the running inner product near zero), which makes $|\langle S, c\rangle|$ small rather
  than exactly zero.
- Positions stay distinguishable because each uses an independently permuted design, so this is
  not the degenerate "same coin sequence at every position" trick. **That degenerate variant
  should be avoided**: if every position receives the identical coin sequence, the network only
  has to learn a constant function and would score perfectly while demonstrating nothing.
- Caveat: the increments are no longer independent, so the unbiasedness argument of the coin flip
  network no longer applies under a randomly ordered replay buffer. For the campaign's fixed
  visitation protocol this does not matter; for a real reinforcement-learning deployment it
  would.

### G. Flattening the tangent-kernel spectrum so the schedule condition holds everywhere

- The condition in section 7 item 3 is about the ratio $\lambda_{\max}/\lambda_{\min}$ of the
  tangent kernel on the 108 points. Anything that compresses that ratio shortens the burn-in and
  removes the slow-direction failure mode.
- Options, in increasing order of intrusiveness: (i) random Fourier features of $[x,y]$ as the
  input embedding, with the bandwidth chosen so that the feature Gram matrix over the point set
  is close to a multiple of the identity; (ii) layer normalisation; (iii) explicit whitening of
  the input embedding using the fixed point set's own statistics; (iv) a Gauss-Newton or
  full-matrix preconditioner on the 108-point residual, which makes every mode decay at the same
  rate by construction.
- This combines with method A rather than replacing it: A supplies the correct exponent, G makes
  the exponent reachable at every position within the 4096-step budget.

### H. Ensemble readout in the style of anti-concentrated confidence bonuses

- Train $M$ independent heads on independent noise targets from the same shared body; read out
  either the maximum absolute prediction (as in the paper, which gives an optimistic bound) or
  the empirical standard deviation across heads (which is the unbiased estimate of the elliptical
  bonus).
- The standard-deviation readout across $M$ heads divides the sampling variance by roughly $M$,
  which is an alternative to raising $d$ in method A and costs the same compute for the same
  variance. Use whichever is cheaper to implement; they are the same statistical device.

### I. Non-uniform visitation

Under the campaign's `nonuniform` regime, methods A, B, C, D, E and F need **no modification at
all** in principle, because each position's readout is a function of that position's own draws or
that position's own accumulated features. What does change:

- The learning-rate condition becomes per-position: position $i$ receives about
  $p_i \cdot(\text{number of steps})$ gradient contributions, so the effective schedule seen by
  position $i$ under a global $\eta_n = a/(n+n_0)$ is **not** $a/n_i$. Rarely visited positions
  get too small a step relative to their own count and lag. Two fixes worth testing:
  (i) weight the per-position squared error by $1/\hat p_i$ where $\hat p_i$ is a running estimate
  of the sampling frequency computed from the data stream (a statistic of the inputs, not a count
  of the position, so it stays within the rules if implemented as a density estimate rather than
  a per-index table); (ii) use a per-parameter adaptive optimizer, which partially performs the
  same rescaling.
- Method C is exempt from this entirely: $\Lambda_n$ accumulates exactly the visits that
  happened, so $b_i = (\lambda + n_i)^{-1/2}$ regardless of the sampling distribution. This makes
  C the natural control for the non-uniform regime.

---

## 9. What to measure, so a result is interpretable

- **Report the statistical ideal alongside the network's curve.** For methods A, D, E, F compute
  $\lVert m_i^{(n)}\rVert_2/\sqrt{d}$ from the coin draws directly (outside the method, purely
  as an instrument) and plot it against the network's readout. The first is the statistical term
  and the second is the statistical term plus optimization error; the gap is the thing to
  attack. This is exactly the decomposition of section 2.3.
- **Report the tangent-kernel spectrum on the point set at initialisation**, and the implied
  burn-in $n_0$ and slow-direction threshold $a\lambda_{\min}$. If a position lags, check whether
  it loads on a small-eigenvalue direction before blaming anything else.
- **Report the expected sampling floor for `dev_worst`** given $d$ and the number of positions,
  so that a measured `dev_worst` can be compared against what sampling alone would produce. A
  method that reaches its own sampling floor is finished; the next gain has to come from $d$,
  from an ensemble, or from the balanced-coin construction.
- **Separate slope from intercept in the reporting.** Several of these constructions give the
  exact exponent $-1/2$ at every position with per-position constants (leverage $\kappa_i$,
  smoothing across neighbours). The campaign's requirement (1) is about the constant and
  requirement (2) is about the exponent; a method can solve the exponent completely and still
  score badly on `dev_worst` purely through the constant, which is fixable by normalisation and
  is a different problem.

---

## 10. Sources

Every item below was retrieved and confirmed to exist during this literature pass; the
identifier given is the one that was checked.

- Lobel, Bagaria, Konidaris. Flipping Coins to Estimate Pseudocounts for Exploration in
  Reinforcement Learning. ICML 2023. arXiv:2306.03186. Proceedings of Machine Learning Research
  volume 202, pages 22594-22613. Code: `github.com/samlobel/CFN`.
- Yang, Tao, Lyu, Li. Exploration and Anti-Exploration with Distributional Random Network
  Distillation. ICML 2024. arXiv:2401.09750. Code: `github.com/yk7333/DRND`.
- Fang, Yang, Tao, Lyu, Li, Shen, Li. Exploration by Random Distribution Distillation.
  arXiv:2505.11044 (May 2025). Publication venue not confirmed.
- Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos. Unifying Count-Based Exploration and
  Intrinsic Motivation. NeurIPS 2016. arXiv:1606.01868.
- Ostrovski, Bellemare, van den Oord, Munos. Count-Based Exploration with Neural Density Models.
  ICML 2017. arXiv:1703.01310.
- Burda, Edwards, Storkey, Klimov. Exploration by Random Network Distillation. ICLR 2019.
  arXiv:1810.12894.
- Zanger, Wu, Van der Vaart, Böhmer, Spaan. On the Equivalence of Random Network Distillation,
  Deep Ensembles, and Bayesian Inference. arXiv:2602.19964 (February 2026).
- Ash, Zhang, Goel, Krishnamurthy, Kakade. Anti-Concentrated Confidence Bonuses for Scalable
  Exploration. ICLR 2022. arXiv:2110.11202. Code: `github.com/JordanAsh/acb`.
- Henaff, Raileanu, Jiang, Rocktäschel. Exploration via Elliptical Episodic Bonuses. NeurIPS
  2022. arXiv:2210.05805. Code: `github.com/facebookresearch/e3b`.
- Abbasi-Yadkori, Pál, Szepesvári. Improved Algorithms for Linear Stochastic Bandits. NeurIPS
  2011, pages 2312-2320.
- Osband, Aslanides, Cassirer. Randomized Prior Functions for Deep Reinforcement Learning.
  NeurIPS 2018, pages 8626-8638.
- Ciosek, Fortuin, Tomioka, Hofmann, Turner. Conservative Uncertainty Estimation By Fitting Prior
  Networks. ICLR 2020. OpenReview `BJlahxHYDS`. Theorem statements not retrieved — automated
  access to OpenReview was blocked.
- He, Lakshminarayanan, Teh. Bayesian Deep Ensembles via the Neural Tangent Kernel. NeurIPS 2020.
- Badia, Sprechmann, Vitvitskyi, Guo, Piot, Kapturowski, Tieleman, Arjovsky, Pritzel, Bolt,
  Blundell. Never Give Up: Learning Directed Exploration Strategies. ICLR 2020. arXiv:2002.06038.
- Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, De Turck, Abbeel. #Exploration: A Study
  of Count-Based Exploration for Deep Reinforcement Learning. NeurIPS 2017. arXiv:1611.04717.
- Machado, Bellemare, Bowling. Count-Based Exploration with the Successor Representation. AAAI
  2020, pages 5125-5133.
- Strehl, Littman. An analysis of model-based Interval Estimation for Markov Decision Processes.
  Journal of Computer and System Sciences, volume 74, issue 8, pages 1309-1331, 2008.
  DOI 10.1016/j.jcss.2007.08.009. The source of the $1/\sqrt{n}$ bonus shape the whole
  pseudo-count literature is trying to reproduce.
- Bai, Zhang, Qiu, Zhang, Xu, Li. Online Preference Alignment for Language Models via Count-based
  Exploration. ICLR 2025. arXiv:2501.12735. Uses a coin-flip counting module.

### Items to verify against the papers themselves before they are quoted in the writeup

The following were read through an automated summariser rather than from the typeset paper, and
the exact form should be confirmed before being reproduced in
`11_decay_rate_development_document.tex`:

- the coin flip network's prioritised-sampling formula and the exact normalisation of its
  optimistic prior;
- the bracketing of the distributional Random Network Distillation second bonus $b_2$ and the
  precise statement of its Lemma 4.3;
- the exact constants in the random-distribution-distillation bound;
- the theorem statements of the conservative-uncertainty paper, which were not retrieved at all.

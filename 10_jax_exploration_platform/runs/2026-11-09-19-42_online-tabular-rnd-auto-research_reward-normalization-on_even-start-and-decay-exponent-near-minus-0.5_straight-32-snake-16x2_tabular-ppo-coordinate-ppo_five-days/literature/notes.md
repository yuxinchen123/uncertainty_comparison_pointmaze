# Paper notes for the even-start, square-root-decay online RND study

Define $s$ as a grid state, $n(s)$ as the number of predictor updates made on $s$ so far (its visit count),
$f(s) \in \mathbb{R}^{d}$ as the frozen random target's output, $g(s)$ as the predictor's output, the residual
$r(s) = g(s) - f(s)$, and the bonus $b(s) = \lVert r(s) \rVert$. The study wants $b(s) = 1$ at every state before
training ("even start") and $b(s) \propto n(s)^{-1/2}$ at every state, with no stored counts in the bonus.

Our building blocks, named the same way in every section:

1. **Zero head with unit target**: predictor output layer initialised to zero, target output divided by its own length
   at each state, so $r(s) = -f(s)/\lVert f(s) \rVert$ and $b(s) = 1$ before training.
2. **Initial-ratio readout**: bonus divided by the untrained predictor's error at the same state.
3. **Power-law loss**: loss $\lVert r \rVert^{q}$ trained with plain SGD; $q = 4$ gives $b \propto n^{-1/2}$ per state
   when states do not interfere.
4. **Shrink-target loss**: regress $g(s)$ toward $f(s) + r(s)/\sqrt{1 + \lVert r(s) \rVert^{2}}$, so each update maps the
   bonus $b$ to $b/\sqrt{1+b^{2}}$.
5. **Input-layer-only training**: only the one-hot input layer (one column per state) is trained.
6. **AdaGrad or 1/t predictor**: AdaGrad or a step size falling like one over the update number.

"August note NN" means `/p/rlprojects/RND/11_decay_rate/literature/NN_*.md` (the August 2026 sweep). Those notes treat
coordinate inputs $[x, y, 0, 0]$ on PointMaze positions; this study uses one-hot inputs, which changes some
conclusions (stated per paper). Each section reports (a) the initial bonus level and how uneven it is, (b) the decay,
(c) what the paper controls, (d) the difference from our blocks.

## Burda, Edwards, Storkey, Klimov 2019 — Random Network Distillation (ICLR 2019)

- (a) The initial bonus is the error between two randomly initialised networks; nothing makes it equal across states.
  Observation whitening acts per input coordinate and intrinsic-reward division by a running standard deviation of
  intrinsic returns acts on all states at once, so neither changes the ratio between two states (August note 04,
  Section 1.2).
- (b) The decay with visits is left to the optimizer (Adam, squared error); the paper makes no claim about its rate.
  Section 2.2.1 lists four sources of prediction error, of which only "amount of training data" is count-like.
- (c) Neither.
- (d) Every block changes a part RND leaves open: initialisation and target (zero head with unit target), readout
  (initial-ratio readout), loss (power-law, shrink-target), trained parameters and optimizer (input-layer-only,
  AdaGrad or 1/t). The run keeps RND's reward normalization on: the reward PPO sees is the raw bonus times one
  time-varying number shared by all states, so it cannot make states uneven at a given time and does not change the
  raw bonus whose decay against $n(s)$ is measured.

## Lobel, Bagaria, Konidaris 2023 — coin-flip network (ICML 2023)

- (a) Without help, an untrained state's readout "tends to be close to 0". They add a frozen random prior network,
  scaled by a running mean and variance so that the average over the data of each squared output coordinate is 1
  (Section 3.5.2). The start is therefore about 1 on average over states, not exactly 1 at each state.
- (b) The network regresses onto a fresh $\pm 1$ vector per visit; at the least-squares optimum the expected squared
  readout is exactly $1/n$, so the readout falls like $n^{-1/2}$ (details: August note 03, Section 2.1). Their
  Figure 2 shows RND's bonus "falls off more sharply than" $1/\sqrt{N(s)}$ at high counts.
- (c) Both, by random targets plus the prior, assuming the regression reaches its optimum (Adam, learning rate
  $10^{-4}$, prioritised replay).
- (d) Appendix E.2 is the key contrast: with the prior but targets fixed at 0 (deterministic, like RND), states visited
  more than once get "almost no exploration bonus". The power-law and shrink-target losses are exactly attempts to
  make a fixed-target regression decay like $n^{-1/2}$ instead of collapsing; the coin-flip network gets the rate from
  averaging random targets instead. Appendix B states that with one-hot inputs and a linear model the solution
  recovers tabular counts, so the one-hot case of input-layer-only training is published in this linear form. Its
  prior gives an even start only on average; the zero head with unit target makes it exact per state.

## Yang, Tao, Lyu, Li 2024 — Distributional RND (ICML 2024)

- (a) Names "initial bonus inconsistency": "uneven distribution of bonuses among states at the beginning of
  training". Lemma 4.2 (linear model) gives the expected initial difference between two inputs as proportional to the
  difference of their squared lengths; with one-hot inputs all lengths are 1, so only the random fluctuation remains.
  Averaging $N$ target networks narrows it (Figure 4, a toy of 100 one-hot categories) but does not remove it.
- (b) Names "final bonus inconsistency" (bonuses not tracking visit frequency). A second bonus term divides by the
  per-state variance of the $N$ targets; Lemma 4.3: at the regression optimum the ratio inside its square root is an
  unbiased estimator of $1/n$.
- (c) Both partly: start statistically narrowed, decay from target sampling.
- (d) Its per-state division by target variance is the nearest published per-state normalizer to the initial-ratio
  readout; the difference is what is divided by: DRND uses a property of its target distribution, ours uses the
  realised error of the untrained predictor at that state. DRND keeps squared error and Adam, and gets its count from
  sampling, not from the loss shape. The paper's toy is a one-hot linear model, so one-hot analysis of RND is
  published (August note 04, Section 3.1; August note 03, Section 2.2).

## Fang, Yang, Tao, Lyu, Li, Shen, Li 2025 — Random Distribution Distillation (arXiv 2505.11044)

- (a) The target is a Gaussian draw around a frozen network's mean; with the mean set to 1 an unseen state "would
  initially give a reward close to 1". The predictor is randomly initialised, so the start is only approximately
  even.
- (b) The bonus is bounded by a count term $\sigma^{2}/n$ plus the gap to the optimal predictor (their Equation 13).
  The introduction states: "the rate at which the RND reward diminishes with repeated visits is not well defined, as
  it depends on the learning dynamics of the neural network and stochastic gradient descent" (the same sentence as
  the coin-flip paper's Section 2.2).
- (c) Decay by sampling; the start approximately.
- (d) Same family as DRND and coin flips (statistical averaging). Its analysis divides by $\sigma^{2}(s)$, but the
  deployed bonus uses one constant $\sigma$. Nothing about loss shape, zeroed heads, or which layers train (August
  note 04, Section 3.2).

## Pecháč, Chovanec, Farkaš — Self-supervised network distillation (arXiv 2302.11563, v4 2024)

- (a) Not addressed.
- (b) The stated problem is that RND "loses the motivational signal, which is caused by the adaptation of the predictor
  model". Training the target with a self-supervised loss (contrastive, augmentation-based, or variance-invariance-
  covariance) raises feature variance and gives "a slowdown in the adaptation of the predictor model".
- (c) Slows the decay qualitatively; no target rate.
- (d) Different lever: a harder, moving target. Our blocks keep a frozen random target and set the rate through the
  loss or the trained parameters (August note 04, Section 3.4).

## Badia et al. 2020 — Never Give Up (ICLR 2020)

- (a), (b) The episodic bonus is $1/\sqrt{\text{kernel pseudo-count}}$ from a nearest-neighbour memory reset each
  episode, so the $n^{-1/2}$ shape is written into the formula. The lifelong RND error becomes a multiplier
  $1 + (e - \mu_e)/\sigma_e$ with running mean $\mu_e$ and standard deviation $\sigma_e$ of the RND error $e$, clipped
  between 1 and 5.
- (c) Decay by an explicit episodic memory; RND only modulates.
- (d) Its RND normalization uses one mean and one standard deviation for all states, not a per-state initial value,
  so it differs from the initial-ratio readout (August note 04, Section 3.5; August note 03, Section 6).

## Zhang et al. 2021 — NovelD (NeurIPS 2021)

- (a), (b) Reward is the clipped difference of RND novelty between the next and the current state, given only on the
  first visit within an episode (explicit episodic count). The motivation is that tabular optimistic initialisation
  with a $1/\sqrt{n}$ bonus explores novel regions breadth-first; the tabular version uses $1/\sqrt{N}$.
- (c) Changes how RND novelty is used, not its start or decay.
- (d) Orthogonal: our blocks change the novelty itself and could be stacked under NovelD. Useful as evidence that the
  even-start, square-root-decay target is what tabular exploration theory asks for.

## Zanger, Wu, Van der Vaart, Böhmer, Spaan 2026 — RND, deep ensembles and Bayesian inference (arXiv 2602.19964)

- (a) In the infinite-width limit the initial residual is a draw from the difference of two prior processes. Their
  "Bayesian RND" target copies the predictor's random hidden layers and sets the final layer to zero (Proposition 4.1).
- (b) The converged error at a training input is set by which inputs are in the training set, not how often each
  appears; at training inputs the noiseless posterior variance is zero. No per-visit rate.
- (c) Neither.
- (d) A zeroed final layer appears here, but in the target and for a different purpose (making the error an exact
  posterior). Our zero head is in the predictor so that the initial residual equals the unit target. Their result
  also says squared-error RND on fixed targets has no defined count dependence at convergence, which is the gap the
  power-law and shrink-target losses address (August note 04, Section 3.7; August note 03, Section 5.2).

## Osband, Aslanides, Cassirer 2018 — Randomized prior functions (NeurIPS 2018)

- (a) Each ensemble member is a trainable network plus a frozen random prior network times one global scale $\beta$;
  the prior gives uncertainty before any data. No per-input evenness claim.
- (b) Posterior sampling is exact only for linear-Gaussian models with noisy targets.
- (c) Only the global scale.
- (d) The coin-flip network's optimistic prior comes from here. A global $\beta$ cannot equalise states; the zero head
  with unit target does so per state (August note 04, Section 2).

## Ciosek, Fortuin, Tomioka, Hofmann, Turner 2020 — fitting prior networks (ICLR 2020; not downloaded)

Not downloaded: OpenReview refused automated access. From August note 04, Section 3.6: the expected squared residual
of a predictor fit to a random prior equals the Gaussian-process posterior variance plus the squared gap to the
posterior mean (conservative estimate); no rate in the visits to one point; no per-input normalization. Relevance:
same reading of RND error as posterior variance plus optimization gap; no overlap with any block.

## Ash, Zhang, Goel, Krishnamurthy, Kakade 2022 — anti-concentrated confidence bonuses (ICLR 2022)

- (a), (b) Least-squares regressors fit to fresh Gaussian noise, with warm-start rows $\sqrt{\lambda} e_i$ that also
  get random targets (their Algorithm 1); the fitted value at $x$ is then Gaussian with standard deviation equal to
  the elliptical width $\sqrt{x^{\top}\Sigma^{-1}x}$, $\Sigma = \lambda I + \sum x x^{\top}$, which with one-hot features
  is $1/\sqrt{\lambda + n}$. The ridge $\lambda$ sets the start; the maximum over many regressors gives an optimistic
  bonus.
- (c) Both, through noise targets and ridge.
- (d) The shrink-target loss reaches the same elliptical quantity by a deterministic per-visit map, without noise
  targets or an ensemble (see `novelty.md`, shrink-target row). Their start depends on the feature length; the zero
  head with unit target fixes it at 1 (August note 03, Section 4.3).

## Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos 2016 — pseudo-counts (NeurIPS 2016)

- (a), (b) A density model's pseudo-count $\hat N$ goes into the bonus $\beta(\hat N + 0.01)^{-1/2}$ with
  $\beta = 0.05$, so the start and the $n^{-1/2}$ shape are set by the formula.
- (c) Decay by formula. Theorem 2 (pseudo-counts grow in proportion to real counts) needs an assumption on the density
  model's rate of learning; their example is a step size $\alpha_n$, where $\alpha_n = 1/n$ gives the empirical
  distribution.
- (d) This is the published reason a $1/n$ step matters for counting, the role the AdaGrad or 1/t predictor plays for
  us; but they compute a count and insert it into a formula, while our bonus is the residual itself (August note 03,
  Section 3.1).

## Ostrovski, Bellemare, van den Oord, Munos 2017 — counts from neural density models (ICML 2017)

- (a), (b) Same bonus formula. Section 3 lists requirements, including "the prediction gain (PG) should decay at a rate
  $n^{-1}$". PixelCNN trained best with a constant learning rate $10^{-3}$, so they multiplied the prediction gain by
  $c_n = 0.1 \thinspace n^{-1/2}$, with $n$ the global update count.
- (c) Decay forced by an explicit global schedule in the readout.
- (d) Closest precedent for forcing the rate when the network will not produce it. Their schedule is global and sits in
  the readout; the power-law and shrink-target losses act per state inside training, with no step count in the
  readout. They also tried a decaying learning rate $0.1 \thinspace n^{-1/2}$, which reached a similar loss (August note 03,
  Section 3.2).

## Tang et al. 2017 — #Exploration (NeurIPS 2017)

- (a), (b) States are hashed (SimHash) into a table of exact counts; bonus $\beta/\sqrt{n(\phi(s))}$ with $\phi$ the
  hash. Start and decay are exact by definition.
- (c) Both, with explicit counts.
- (d) Reference for "correct". With one-hot inputs, input-layer-only training gives each state its own parameter
  column, which is close to a table, but no count is stored or read; the decay must come from the training rule.

## Strehl, Littman 2008 — MBIE-EB (Journal of Computer and System Sciences 74)

- (a), (b) MBIE-EB adds $\beta/\sqrt{n(s,a)}$ to the empirical Bellman equation (their Equation 8) and initialises
  action values optimistically at $1/(1-\gamma)$; $\beta$ is one constant for all pairs. PAC-MDP bounds follow.
- (c) Both, with explicit counts.
- (d) The bonus shape our method should reproduce without counts: the same constant and the same exponent at every
  state. The ICML 2005 paper by the same authors (also downloaded) analyses MBIE only and has no MBIE-EB bonus.

## Henaff, Raileanu, Jiang, Rocktäschel 2022 — E3B (NeurIPS 2022)

- (a), (b) Bonus $\phi(s)^{\top}C^{-1}\phi(s)$ with $C = \lambda I + \sum \phi\phi^{\top}$ summed within the episode;
  "if the problem is tabular and $\phi$ is a one-hot encoding ... yielding a bonus of $1/N_e(s)$" (squared form, so
  $n^{-1}$, not $n^{-1/2}$). Features are learned by an inverse-dynamics model.
- (c) Both, by a stored matrix, reset every episode.
- (d) With one-hot features and $\lambda = 1$ the square root of a lifelong version is $1/\sqrt{1+n}$, exactly what
  the shrink-target loss produces; the shrink-target map is the rank-one update of this quantity at the visited state
  (derivation in `novelty.md`). Ours stores it in a network residual instead of a matrix (August note 03, Section 4.2).

## Fox, Choshen, Loewenstein 2018 — DORA, E-values (ICLR 2018)

- (a) Tabular E-values start at 1. With networks (Section 4.1, Atari), a logistic output is put on the last layer of
  the E-value stream and that layer's weights are initialised to 0, so "the E-values were initialized at 0.5" at
  every input — an even start by zeroing the output layer.
- (b) With $\gamma_E = 0$ each visit applies $E \leftarrow (1-\alpha)E$, so $E = (1-\alpha)^{n}$ (geometric); the count
  is read as $\log_{1-\alpha}E$ and the Atari bonus is $\beta/\sqrt{-\log E}$, $\beta = 0.05$.
- (c) Both: start by initialisation, decay by a fixed per-visit factor plus a log readout.
- (d) The closest published relative of the shrink-target loss: a learned quantity regressed each visit toward a shrunk
  copy of itself. Differences: DORA's factor is a constant, so the decay is geometric and needs a log readout with the
  learning rate as the unit of count; ours depends on the current residual and gives $n^{-1/2}$ directly as the plain
  RND residual. DORA's zeroed last layer is also prior art for the zero-head half of our first block.

## Guo et al. 2022 — BYOL-Explore (NeurIPS 2022)

- (a) The loss is the squared distance between the length-normalised prediction and the length-normalised target,
  $\lVert g/\lVert g \rVert - f/\lVert f \rVert \rVert^{2}$, with the target an exponential moving average of the
  online network. Each term lies in $[0, 4]$; for nearly orthogonal random vectors it starts near 2 at every input.
- (b) Not controlled; the reward is divided by a running standard deviation as in RND.
- (c) Neither explicitly.
- (d) Unit-length targets in a prediction-error bonus are published here. Differences: BYOL-Explore also normalises the
  prediction (a cosine distance), its target moves, and its predictor head is not zeroed; ours normalises only the
  frozen target and zeroes the predictor head, so the start is exactly 1 and the residual can shrink toward 0.

## Machado, Srinivasan, Bowling 2014 — domain-independent optimistic initialisation (arXiv 1410.4604)

- (a) With linear value functions $\theta^{\top}\phi$, an equal optimistic start at every state needs a known reward
  scale and feature vectors of constant length. Their fix: normalise rewards by the first non-zero reward and shift
  the value function so that all-zero weights already mean "optimistic"; then $\theta = 0$ gives the same start at
  every state.
- (b) Decay is whatever Sarsa($\lambda$) learning does; not controlled.
- (c) The start only.
- (d) The same two ingredients as the zero head with unit target, in linear Q-learning: zero weights for an equal
  start and constant-length features. Ours puts the constant length on the frozen target's output and the zeros on
  the predictor head of RND.

## Guei, Chen, Wu 2021 — optimistic TD learning for 2048 (IEEE Transactions on Games)

- (a) The n-tuple network (lookup tables indexed by board patterns) has its weights initialised "to very large
  values".
- (b) "Since weights tend to be reduced once the states are visited", rarely visited patterns keep high values; the
  decay follows the TD step size, with no designed rate.
- (c) The start only.
- (d) A published table-of-weights version of "start high, shrink on each visit", close in spirit to input-layer-only
  training plus a shrinking update; no square-root decay and no RND target.

## Davoodabadi, Hashemi Dijujin, Soleymani Baghshah 2024 — PreND (arXiv 2410.01745)

- (a) Not addressed beyond noting "poor target network initialization".
- (b) Names "rapid degradation of intrinsic rewards" in RND, caused by "the predictor network's very fast adaptation
  to the frozen target". Their "RND+LR" variant gives the predictor its own optimizer with the learning rate
  multiplied by 0.01; PreND also uses pre-trained target and predictor networks.
- (c) Slows the decay by a smaller learning rate; no target exponent, no per-state analysis.
- (d) Same lever as the AdaGrad or 1/t predictor (the predictor's step size), but a constant rescaling. Ours aims at a
  per-state $n^{-1/2}$ law, and the power-law and shrink-target losses change the loss, not only the step.

## Hou, An, Du 2026 — learning progress monitoring (ICLR 2026)

- (a), (b) A second "error model" predicts the dynamics model's error at its previous update; the intrinsic reward is
  that predicted earlier error minus the current error (their Equation 3), so it rewards improvement, not error.
- (c) Neither the start nor the visit-count decay of a novelty bonus.
- (d) The nearest published use of a stored earlier error as a reference. It takes a difference with the previous
  update; the initial-ratio readout divides by the error of the untrained predictor, fixed for the whole run.

## Rashid, Peng, Böhmer, Whiteson 2020 — OPIQ (ICLR 2020)

- (a) "Merely initialising the network to output optimistic Q-values is not enough, since we cannot ensure that they
  remain optimistic for novel state-action pairs": network generalisation moves the values of unvisited pairs.
- (b) Optimism is kept outside the network as $C/(N(s,a)+1)^{M}$ added to the Q-values, with explicit counts.
- (c) Both, with counts.
- (d) States the failure an even start in a network must avoid. Input-layer-only training on one-hot inputs avoids it
  by construction (a visit changes only its own column), while a fully trained predictor does not; our bonus uses no
  counts. $M = 1/2$ is the square-root shape our bonus should match.

## Even-Dar, Mansour 2003 — learning rates for Q-learning (JMLR 5)

- (a), (b) Tabular Q-learning with a per-pair step $1/k^{\omega}$, where $k$ is one plus the visits to that pair:
  $\omega \in (1/2, 1)$ converges in time polynomial in $1/(1-\gamma)$, while $\omega = 1$ can need time exponential in
  it.
- (c) Neither (not about bonuses).
- (d) Background for the AdaGrad or 1/t predictor: per-entry visit-indexed step sizes are standard in tabular
  learning, and the theory prefers decay slower than $1/k$. Ours obtains a per-state step without counts (AdaGrad on
  a one-hot column), and the shrink-target loss fixes the per-visit change directly instead of through a step size.

## Jacot, Gabriel, Hongler 2018 — Neural Tangent Kernel (NeurIPS 2018)

- In the infinite-width limit, training follows linear function-space dynamics with the tangent kernel. Our reading,
  not stated in the paper: for a fully connected network with independent Gaussian initial weights, the kernels depend
  on the inputs only through their lengths and their inner product, so on one-hot inputs every diagonal entry is the
  same and every off-diagonal entry is the same. Then the expected initial bonus and the per-state curvature are equal
  across states, and interference between states is the same for every pair.
- With input-layer-only training, a gradient step on $s$ changes only column $s$, so the tangent kernel between two
  different states is exactly zero at any width: no interference, which is the condition under which the power-law
  loss gives $n^{-1/2}$ per state. August note 01 and August note 05 cover the coordinate-input case.

## Rahaman et al. 2019 — spectral bias (ICML 2019)

- Networks fit low-frequency components of the target first. On coordinate inputs this is why RND's per-position decay
  rates differ (August note 01, Sections 2–3; August note 05, Section 4.1). On one-hot inputs every pair of states is
  equally far apart, so there is no input-space frequency to favour; the remaining non-uniformity is finite-width
  randomness and visit counts. No overlap with any block; it explains why one-hot inputs remove one source of uneven
  decay.

## Duchi, Hazan, Singer 2011 — AdaGrad (JMLR 12)

- Per-coordinate step $\eta/\sqrt{\sum_m g_m^{2}}$, the sum running over that coordinate's past gradients $g_m$;
  stated aim: "frequently occurring features very low learning rates and infrequent features high learning rates".
  With one-hot inputs, column $s$ of the input layer receives gradients only on visits to $s$, so AdaGrad gives each
  state its own step size from its own history: an implicit per-state schedule with no stored count.
- August note 02, Section 4.4: with squared error on a fixed target the accumulator converges, so the decay becomes
  geometric, not a power law. This limits the AdaGrad or 1/t predictor under squared error; it does not settle the
  power-law-loss case. The AdaGrad paper itself is a regret analysis and says nothing about exploration bonuses.

# Preconditioning and optimizer design for a bonus that decays as $n^{-1/2}$ at every position

Scope of this file: optimizers that make different directions or different training examples converge at
the *same* rate — second-order and preconditioned methods, per-example reweighting, and kernel spectral
filtering — read against the concrete target of an RND predictor
$g_n : \mathbb{R}^4 \to \mathbb{R}^{128}$ (one hidden layer of width 256, ReLU) distilled by full-batch
training onto a frozen random target $f$ at about 100 fixed maze positions $x_1,\dots,x_m$, with
per-position bonus $b_i(n) = \lVert g_n(x_i) - f(x_i) \rVert_2$ after $n$ full-batch steps.

Every rate below says which quantity decays (a loss, a squared residual, or a residual norm) and under
which assumptions. Numbers labelled "measured here" come from the small checks in
§6, which I ran while writing this file.

## 1. The problem stated exactly, and what "uniform decay" forces

Define the notation first.

- Let $m$ be the number of training positions and $d = 128$ the output width. Define the residual at
  position $i$ after $n$ steps as $r_i(n) = g_n(x_i) - f(x_i) \in \mathbb{R}^{d}$, and let $r(n)$ be the
  stacked vector of all $m d$ residual entries. The bonus is $b_i(n) = \lVert r_i(n) \rVert_2$.
- Let $J(n)$ be the Jacobian of the stacked network outputs with respect to the parameters, of shape
  $m d \times p$ where $p$ is the parameter count. Define the tangent-kernel Gram matrix
  $K(n) = J(n) J(n)^{\top}$, of shape $m d \times m d$.
- Let $\eta_n$ be the step size at step $n$, and let $\lambda_1 \ge \dots \ge \lambda_{md} \ge 0$ be the
  eigenvalues of $K$ with eigenvectors $v_j$.

For our predictor, $p = 4\cdot 256 + 256 + 256\cdot 128 + 128 = 34{,}176$ parameters against
$m d = 100 \cdot 128 = 12{,}800$ output constraints, so $p > m d$: the Jacobian can have full row rank,
which is the condition every method in §2.1 needs.

**Fact 1 (plain gradient descent, linearized).** If $K$ is held fixed (the linearized or wide-network
regime), full-batch gradient descent on $\tfrac{1}{2}\lVert r \rVert_2^2$ gives
$r(n) = \prod_{t \le n} (I - \eta_t K)\, r(0)$. Along eigenvector $v_j$, the *residual component* is
multiplied by $\prod_{t\le n}(1 - \eta_t \lambda_j)$. Different $\lambda_j$ give different factors: this
is the entire source of heterogeneous decay.

**Fact 2 (the $1/t$ schedule turns each eigenvalue into its own exponent).** With $\eta_t = a/t$,
$\prod_{t \le n} (1 - a\lambda_j / t) \approx n^{-a \lambda_j}$. So the *residual component* along mode
$j$ follows a power law whose exponent is $a\lambda_j$ — one exponent per eigenvalue, not one exponent
for the whole problem. Measured here on a 20-mode linear model with eigenvalues spread over
$[10^{-3}, 1]$ and $a = 0.5$: the fitted per-mode exponents were $-0.5000$ down to $-0.0005$, matching
$-a\lambda_j$ to four digits.

This is the precise diagnosis of the reported problem. A position's residual is a *mixture* of modes,
$r_i(n) = \sum_j c_{ij} n^{-a\lambda_j} v_j$, so:

- Each position's fitted slope is a mixture-dominated quantity that changes with the fitting window; two
  positions with different mode weights $c_{ij}$ give visibly different slopes at the same $n$.
- Asymptotically every position tends to the *slowest* exponent $a\lambda_{\min}$ present in its mixture,
  not to $-1/2$. Measured here: per-position slopes on the same 20-mode model had spread
  $\mathrm{sd} = 0.44$ over $n \in [10^2,10^3]$ and drifted from mean $-0.18$ to mean $-0.02$ as the
  window moved to $n \in [10^4, 2\cdot 10^5]$, and some positions even showed *positive* slopes because
  of sign cancellation between modes.
- Therefore an aggregate slope of $-0.503$ from plain SGD with a $1/t$ schedule is a coincidence of the
  learning-rate constant $a$, the kernel scale and the fitting window. It is not structural, and it will
  move if the learning rate, the loss normalisation (sum versus mean over positions) or the horizon
  changes. Measured here: the same network and schedule with the loss averaged rather than summed over
  the 100 positions gave an aggregate slope of $-0.055$, with per-position slopes spread over
  $[-0.098, -0.020]$.

**Fact 3 (what uniformity requires).** Requiring $\prod_{t\le n}(1-\eta_t \lambda_j)$ to be the same for
every $j$ forces $\lambda_j$ to be the same for every $j$, i.e. $K \propto I$. Any method that keeps the
freedom to choose only a scalar schedule cannot equalise positions; the equalisation has to come from a
*preconditioner* that turns the effective kernel into a multiple of the identity. Equivalently, in the
language of §4: any iterative scheme whose iterate is $r(n) = q_n(K) r(0)$ for some scalar function
$q_n$ produces per-mode factors $q_n(\lambda_j)$, and demanding $q_n(\lambda) = n^{-1/2}$ for all
$\lambda$ in the spectrum forces $q_n$ to be the constant function — which is the Newton or
Gauss-Newton iteration, not a spectral filter of $K$.

**Fact 4 (the schedule that produces exactly $n^{-1/2}$ once the rate is shared).** If every mode is
contracted by the same factor $1-\eta_t$ per step, then
$\lVert r_i(n)\rVert_2 = \lVert r_i(0) \rVert_2 \prod_{t\le n}(1-\eta_t)$, and with $\eta_t = 1/(2t)$,

$$\prod_{t=1}^{n}\Bigl(1 - \frac{1}{2t}\Bigr) = \frac{\Gamma(n + 1/2)}{\Gamma(1/2)\,\Gamma(n+1)} \longrightarrow (\pi n)^{-1/2}.$$

Measured here: the product equals $(\pi n)^{-1/2}$ to within a factor $0.99988$ at $n = 10^4$, and its
log-log slope over $n \in [10^3, 2\cdot10^4]$ is $-0.49996$. More generally $\eta_t = a/t$ gives slope
exactly $-a$ (measured: $-0.2500$, $-0.5000$, $-1.0000$ for $a = 0.25, 0.5, 1$). Two schedules that do
*not* work: a constant step gives geometric decay of the residual norm (a straight line on a log-linear
plot, not on a log-log plot), and $\eta_t = c/t^{1/2}$ gives
$\log \lVert r \rVert \sim -2c\sqrt{n}$, a stretched exponential whose apparent log-log slope keeps
steepening (measured $-11.96$ over $n\in[10^4,2\cdot10^4]$).

**Fact 5 (why the observed Adam curve saturates).** For a decoupled quadratic (a lookup table, one
parameter per position), AdaGrad-style accumulation gives
$r(n+1) = r(n)\bigl(1 - \eta / \sqrt{\textstyle\sum_{t\le n} r(t)^2}\bigr)$. Writing
$u_n = \sqrt{\sum_{t\le n} r(t)^2}$ and eliminating $n$ gives $\mathrm{d}r/\mathrm{d}u = -2\eta/r$, so
$u$ converges to a finite limit as $r \to 0$: the accumulator *saturates*, the effective step stops
shrinking, and the residual reverts to geometric decay. Measured here: $\eta = 0.1$ gave
$r(10^4) = 3.6\times10^{-140}$ and a local log-log slope of $-463$. This is a clean explanation of why a
constant-rate Adam run first looks like a steep power law and then falls off a cliff, and it says the
fix cannot be "tune Adam's learning rate".

## 2. Paper notes

### 2.1 Preconditioned and second-order methods

**Amari, "Natural Gradient Works Efficiently in Learning", Neural Computation 10(2), 1998
(doi:10.1162/089976698300017746).**
What it shows: when the parameter space carries a Riemannian metric (the Fisher information matrix
$F$), the steepest-descent direction is $F^{-1}\nabla L$, not $\nabla L$; online natural gradient
descent is Fisher efficient, i.e. asymptotically as good as batch maximum likelihood. Relevance here:
this is the origin of the idea that the *metric*, not the schedule, decides which directions move fast.
It says nothing directly about equalising per-example residuals; the per-example statement comes from
the Gauss-Newton reading in the next entry.

**Zhang, Martens, Grosse, "Fast Convergence of Natural Gradient Descent for Overparameterized Neural
Networks", NeurIPS 2019 (arXiv:1905.10961).**
This is the single most directly relevant paper for question 1. Precise content:

- The update they analyse (their eq. 3) is
  $\theta(k+1) = \theta(k) - \eta\, J^{\top} (J J^{\top})^{-1} (u - y)$, where $u$ stacks the network
  outputs on all $n$ training cases and $y$ the targets. This is the minimum-norm Gauss-Newton step: it
  is *exactly* the update that realises a prescribed change in output space.
- Condition 1 is that $J(0)$ has full row rank, equivalently that the Gram matrix
  $G(0) = J(0)J(0)^{\top}$ is positive definite. They note this forces the parameter count to be at
  least the number of output constraints, so the Fisher matrix is singular and a generalised inverse
  must be used. Condition 2 is that $J$ moves little in a ball around initialisation of radius
  $3\lVert y - u(0)\rVert_2 / \sqrt{\lambda_{\min}(G(0))}$.
- Theorem 1: under those conditions and step size $\eta \le (1-2C)/(1+C)^2$ (with $C < 1/2$ the Jacobian
  drift constant), the **squared** residual satisfies
  $\lVert u(k)-y\rVert_2^2 \le (1-\eta)^k \lVert u(0)-y \rVert_2^2$. Note the quantity: this is the
  squared error. In the exactly linear case the residual *vector* itself satisfies
  $u(k)-y = (1-\eta)^k (u(0)-y)$, so the residual *norm* decays as $(1-\eta)^k$ and the squared norm as
  $(1-\eta)^{2k}$; the theorem's $(1-\eta)^k$ on the squared error is the looser statement that also
  covers the nonlinear drift.
- The key structural point for us, which the paper states implicitly through eq. 3 and which the
  linear-case identity makes explicit: the residual is contracted by the *same scalar* in every
  direction. At $\eta = 1$ on the exactly linear model the residual is eliminated in one step.
- Theorem 3: for a two-layer ReLU network with hidden width
  $m = \Omega(n^4 / (\nu^2 \lambda_0^4 \delta^3))$ the conditions hold with high probability and the same
  $(1-\eta)^k$ bound applies with $\eta = O(1)$, whereas the comparable gradient-descent bounds require
  $\eta = O(1/n)$ and carry a $\lambda_0$ dependence.
- Theorem 4 (K-FAC): the same setup, but the proven rate is
  $\lVert u(k)-y\rVert_2^2 \le (1 - \eta/\lambda_{\max}(X^{\top}X))^k \lVert u(0)-y \rVert_2^2$, i.e. the
  rate is governed by the condition number of the *input* Gram matrix $X^{\top}X$. This is the precise
  sense in which an approximate preconditioner re-introduces non-uniform rates: K-FAC does not deliver
  the direction-independent contraction that exact natural gradient does.
- Damping: the paper's analysis is for the undamped generalised inverse. It does not analyse a Tikhonov
  damping term, so the damping question has to be answered from the practical literature below and from
  the direct measurement in §6.

**Martens, "Deep Learning via Hessian-free Optimization", ICML 2010.**
What it shows: a truncated-Newton method for deep networks in which Hessian-vector products are computed
exactly and the quadratic sub-problem is solved by conjugate gradients. The load-bearing choice for us:
Martens uses the generalised **Gauss-Newton** matrix rather than the Hessian, because the Hessian is
indefinite so the quadratic model can be unbounded below, and because bounding it by damping alone is not
practical. Damping is Levenberg-Marquardt style: a term $\lambda I$ added to the curvature matrix with
$\lambda$ adapted by the ratio of actual to predicted reduction. Relevance: it is the reference for
"Gauss-Newton on small networks, with the damping actually written down".

**Martens and Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature",
ICML 2015.**
What it shows: the Fisher matrix's layer-diagonal blocks are approximated as Kronecker products of two
much smaller matrices (activation covariance and pre-activation gradient covariance), which makes the
inverse cheap. Reported to be much faster per unit of progress than SGD with momentum while costing only
a few times a gradient. Two things to carry over: (a) K-FAC is a *block* approximation, so it does not
whiten the output-space residual, which matches Theorem 4 above; (b) K-FAC's damping is applied in the
Kronecker factors ("factored Tikhonov"), which is not the same as adding $\lambda I$ to the full Fisher —
a detail that matters if one tries to make the damping small enough to preserve uniform rates.

**Gupta, Koren, Singer, "Shampoo: Preconditioned Stochastic Tensor Optimization", ICML 2018
(arXiv:1802.09568).**
What it shows: a preconditioner per tensor dimension, each contracting over the other dimensions, giving
a structure-aware approximation to full-matrix AdaGrad at close to SGD cost per step. The theory is a
regret bound in the *stochastic convex* setting, proved with matrix trace inequalities. What it does not
show: anything about equalising per-example residual decay. Shampoo preconditions in parameter space by
dimension, not in output space by example, so for our target it belongs with K-FAC as "cheap
approximation that will help but will not give the exact shared rate".

**Duchi, Hazan, Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization",
JMLR 12:2121-2159, 2011.**
What it shows: the full-matrix AdaGrad preconditioner $G_n^{-1/2}$ with
$G_n = \sum_{t \le n} g_t g_t^{\top}$, and its diagonal approximation, with regret bounds that improve on
plain subgradient descent when the data geometry is skewed; the marketing claim is that rare but
predictive features get effectively larger steps. Relevance: the diagonal version is the ancestor of
Adam, and Fact 5 above explains why it saturates on this problem. The *full-matrix* version is closer to
what we want because $G_n^{-1/2}$ does whiten across directions, but the accumulated-square accumulator
still produces the wrong curve shape unless the schedule is imposed separately.

**Müller and Zeinhofer, "Achieving High Accuracy with PINNs via Energy Natural Gradients",
ICML 2023 (arXiv:2302.13163).**
What it shows: a natural gradient with respect to the Hessian-induced Riemannian metric of the energy
functional; the resulting update direction *in function space* is the Newton direction projected
orthogonally onto the model's tangent space. Empirically the solution error is several orders of
magnitude smaller than gradient descent, Adam or BFGS given more compute. Relevance: the cleanest
existing statement of the principle we want — pick the metric so the *function-space* increment is the
one you asked for, and let the projection handle the fact that the network cannot represent arbitrary
increments. Our §5 method M1 is the same idea specialised to a finite point set where the projection is
exact.

**Jiang, Voronin, Cyr, Southworth, "On the Convergence Behavior of Preconditioned Gradient Descent
Toward the Rich Learning Regime", arXiv:2601.03162 (January 2026, revised May 2026;
OpenReview CXlsqTAf1E).**
What it shows: preconditioned gradient descent, Gauss-Newton in particular, mitigates spectral bias;
their conjecture is that preconditioned gradient descent enables uniform exploration of parameter space
in the NTK regime, and experiments show reduced grokking delay. Relevance: an explicit, recent statement
that Gauss-Newton gives near-uniform decay across modes, which is exactly the claim question 1 asks
about. Caveat: I read the abstract and the summary, not the proofs; treat the "uniform" claim as a
conjecture plus experiments rather than a theorem.

**Geifman, Barzilai, Basri, Galun, "Controlling the Inductive Bias of Wide Neural Networks by Modifying
the Kernel's Spectrum", arXiv:2307.14531 (2023, revised 2024; OpenReview submission aD0ExytnEK — I did
not confirm a conference acceptance).**
What it shows: "Modified Spectrum Kernels", a construction that approximates a kernel with a *desired*
eigenvalue sequence, together with a preconditioned gradient descent method that follows the modified
kernel's trajectory. The claim is polynomial and in some cases exponential training speedup **without
changing the final solution**. Relevance: this is the closest thing I found to "design the
per-eigendirection rates you want", i.e. to question 4's second half. It reshapes the spectrum but does
not prescribe a decay curve in $n$; combining it with a schedule is our step.

**Korbit, Adeoye, Bemporad, Zanon, "Exact Gauss-Newton Optimization for Training Deep Neural Networks",
arXiv:2405.14402.**
What it shows: an exact (not Kronecker-factored) generalised Gauss-Newton method that uses the
Duncan-Guttman matrix identity to factorise a matrix of the size of the *mini-batch* rather than of the
parameter vector, so it is cheap exactly when the network has many more parameters than batch samples —
which is our situation (34,176 parameters against 12,800 output constraints). It adds line search,
adaptive regularization and momentum, proves convergence to a stationary point under standard conditions,
and reports competitive or better generalisation than SGD, Adam and several second-order baselines.
Relevance: the implementation reference for doing exact Gauss-Newton at our scale.

### 2.2 What the tangent-kernel literature says about the *non*-uniformity we are fighting

**Jacot, Gabriel, Hongler, "Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
NeurIPS 2018.**
What it shows: in the infinite-width limit the network function follows kernel gradient descent under a
kernel (the NTK) that is deterministic and essentially constant during training. Consequence for us:
Fact 1 above is not a modelling convenience, it is the correct description of a wide network, so the
per-mode picture applies to the real predictor.

**Arora, Du, Hu, Li, Wang, "Fine-Grained Analysis of Optimization and Generalization for
Overparameterized Two-Layer Neural Networks", ICML 2019 (arXiv:1901.08584).**
What it shows: a per-eigendirection convergence characterisation for two-layer ReLU networks — the
component of the residual along the $j$-th eigenvector of the (fixed) kernel decays like
$(1-\eta\lambda_j)^n$, and the paper uses exactly this to explain why fitting random labels (which put
mass on small-eigenvalue directions) is slow. Relevance: it is the citable form of Fact 1 for a real
network, and it says the heterogeneity is a property of the kernel spectrum, so no schedule can remove
it.

**Basri, Galun, Geifman, Jacobs, Kasten, Kritchman, "Frequency Bias in Neural Networks for Input of
Non-Uniform Density", ICML 2020 (arXiv:2003.04560).**
What it shows: with training inputs drawn from a non-uniform density $p$, and learning a pure harmonic of
frequency $\kappa$, convergence *at a point* $x$ takes time of order $\kappa^{d} / p(x)$ — the local
density enters the local convergence rate directly. Relevance: this is the theoretical statement behind
the harder extension in our problem. Under non-uniform visitation the effective kernel becomes
(kernel) times (density), so a global schedule necessarily produces different rates at differently
visited positions, and the fix must divide the density back out — which is exactly what a per-position
step $1/(2n_i)$ does.

**Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi, Barron, Ng,
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains",
NeurIPS 2020 (arXiv:2006.10739).**
What it shows: passing inputs through a random Fourier feature map turns the composed NTK into a
*stationary* kernel with a tunable bandwidth; the bandwidth directly controls the rate of convergence of
each frequency component. Relevance: the cheapest architectural lever on the kernel spectrum. Pushing
the bandwidth up makes the kernel matrix on our 100 positions closer to a multiple of the identity, which
narrows the spread of $\lambda_j$ and so narrows the spread of per-position slopes — at the cost of the
bonus generalising less to positions between the training points. That trade-off is the real design
tension for RND and is worth measuring directly.

### 2.3 Kernel spectral filtering, early stopping, and shaping a decay curve

The vocabulary: a linear estimator of the form $\hat f = g_\Lambda(K) K y$ is a *spectral filter*
estimator, and the *residual polynomial* $1 - \lambda g_\Lambda(\lambda)$ is what governs how much of
each eigendirection survives. Gradient descent with constant step $\tau$ is the Landweber iteration, with
filter $g_k(\lambda) = \tau \sum_{i<k}(1-\tau\lambda)^i$ and residual factor $(1-\tau\lambda)^k$; ridge
regression is Tikhonov, $g_\Lambda(\lambda) = 1/(\lambda + \Lambda)$; iterated Tikhonov and the
$\nu$-methods (Brakhage's accelerated Landweber, a two-term recurrence with Chebyshev-like residual
polynomials) fill in the rest.

**Yao, Rosasco, Caponnetto, "On Early Stopping in Gradient Descent Learning", Constructive Approximation
26:289-315, 2007 (doi:10.1007/s00365-006-0663-2).**
What it shows: gradient descent in a reproducing-kernel Hilbert space with **polynomially decreasing
step sizes** $\eta_t \propto t^{-\theta}$; an early-stopping rule and probabilistic bounds on the
*generalisation error* from a bias-variance trade-off, with the connections to boosting, Landweber
iteration and online learning spelled out. Relevance: the closest existing treatment of "a decaying step
schedule on a kernel method", and it is the reference that says what such schedules do to the
*statistical* error. It does not analyse the per-point residual curve, which is what we care about.

**Bauer, Pereverzev, Rosasco, "On regularization algorithms in learning theory", Journal of Complexity
23(1):52-72, 2007.**
What it shows: the general spectral-regularization family in learning theory, with the notion of a
filter's *qualification* $q$: the largest exponent such that
$\sup_\lambda |1 - \lambda g_\Lambda(\lambda)|\,\lambda^{\nu} \le \omega_\nu \Lambda^{\nu}$ for all
$\nu \le q$. Tikhonov has qualification 1 (this is the classical "saturation"), Landweber and iterated
Tikhonov have higher qualification. Relevance: this is the machinery for reasoning about *what residual
shapes a filter can produce*, and it makes the negative result in Fact 3 precise — a filter's residual
factor is a function of $\lambda$, so it cannot be constant in $\lambda$ unless the filter is
$g(\lambda) = c/\lambda$, i.e. the (regularised) inverse.

**Lo Gerfo, Rosasco, Odone, De Vito, Verri, "Spectral Algorithms for Supervised Learning", Neural
Computation 20(7):1873-1897, 2008 (doi:10.1162/neco.2008.05-07-517).**
What it shows: a catalogue of learning algorithms derived from spectral regularization of ill-posed
inverse problems — Landweber/$L^2$ boosting, $\nu$-method, iterated Tikhonov, truncated singular value
decomposition — all consistent kernel methods, with the filter of each written out. Relevance: this is
the reference to consult when asking "does a filter exist that gives shape $X$"; the answer for shape
$n^{-1/2}$ *uniformly in $\lambda$* is no, for the reason in Fact 3.

**Rudi, Carratino, Rosasco, "FALKON: An Optimal Large Scale Kernel Method", NeurIPS 2017
(arXiv:1705.10958).**
What it shows: Nyström subsampling plus a Nyström-built preconditioner inside a conjugate-gradient
solver, giving optimal statistical rates at $O(n)$ memory and $O(n\sqrt n)$ time. Relevance: not about
decay shape at all, but it is the standard answer to "the exact solve is too big" if our exact
per-position control ever has to run over thousands of distinct positions rather than 100.

**A note on the direction of the goal.** All of the statistical spectral-filter theory exists to make the
small-eigenvalue directions decay *slowly* — that is what regularisation is. Our target is the exact
opposite: we want every direction, including the smallest, to decay at the same prescribed rate, which is
the *unregularised* limit. So the theory tells us what our scheme costs (no implicit regularisation, the
fit interpolates the target function's noise-like directions as fast as its smooth ones) rather than
handing us an algorithm. For RND this may be acceptable — there is no label noise, the target is a fixed
deterministic random function — but it does mean the behaviour of the bonus *between* training positions
is controlled entirely by the norm-minimality of the update, not by any filtering.

### 2.4 Per-example reweighting and per-example step sizes

**Van der Sluis, "Condition Numbers and Equilibration of Matrices", Numerische Mathematik 14:14-23,
1969.**
What it shows: row (or column) equilibration — scaling so all row norms are equal — is within a factor of
$\sqrt{q}$ (with $q$ the maximum number of non-zeros per row) of the *best possible* diagonal scaling for
the condition number. Relevance: this is the quantitative ceiling on per-example reweighting. Reweighting
example $i$'s loss by $w_i$ replaces $K$ by $KW$ with $W$ diagonal, and the best any diagonal $W$ can do
is reduce the condition number to within $\sqrt m$ of the diagonal optimum; it cannot make $KW = I$ unless
$K$ was already diagonal. So per-example weights alone cannot equalise per-position decay rates on a
kernel with genuinely mixed eigenvectors. They can, however, remove *scale* imbalances between positions,
which is a useful partial fix.

**Wang, Yu, Perdikaris, "When and why PINNs fail to train: A neural tangent kernel perspective",
Journal of Computational Physics 449:110768, 2022 (arXiv:2007.14527).**
What it shows: the tangent kernel of a physics-informed network splits into blocks for the different loss
terms, whose eigenvalues live on very different scales; the resulting gradient flow is stiff, and the
authors propose weights computed from the kernel's eigenvalues (and, in the companion work, from
back-propagated gradient statistics) so the different loss terms converge at comparable rates.
Relevance: the clearest worked example of "reweight to equalise convergence rates", done at the level of
*loss terms*, with the tangent kernel used to set the weights. The same construction at the level of
individual positions is our method M7, and van der Sluis says how far it can go.

**McClenny and Braga-Neto, "Self-Adaptive Physics-Informed Neural Networks", Journal of Computational
Physics 474:111722, 2023 (earlier version arXiv:2009.04544, "Self-Adaptive Physics-Informed Neural
Networks using a Soft Attention Mechanism").**
What it shows: a *per-training-point* weight, trainable by gradient ascent on the same loss (a minimax
formulation), so the network is pushed to focus on the points it fits worst. Relevance: an existing,
simple implementation of per-position weights that adapt during training. Its dynamics push toward
equalising *loss levels*, which is related to but not the same as equalising decay *rates*; for us the
levels equalising is requirement (1) and the rates equalising is requirement (2), so this addresses only
half the target.

**Chen, Badrinarayanan, Lee, Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in
Deep Multitask Networks", ICML 2018 (arXiv:1711.02257).**
What it shows: task weights adjusted so each task's back-propagated gradient norm at a shared layer is
close to the average gradient norm, scaled by that task's *relative inverse training rate*
$\tilde L_i = L_i / L_i(0)$ raised to a power $\alpha$. Explicitly a mechanism to make tasks train at
similar rates. Relevance: this is a per-group version of exactly what question 3 asks about, and the
"relative inverse training rate" is the right template for a per-position weight: normalise each
position's progress by its own initial loss before comparing.

**Ren, Zeng, Yang, Urtasun, "Learning to Reweight Examples for Robust Deep Learning", ICML 2018.**
What it shows: per-example weights derived from a one-step look-ahead — weight example $i$ by the
alignment of its gradient with the gradient on a clean validation set. Relevance: shows the machinery for
per-example weights that are recomputed each step; the objective is robustness, not rate equalisation, so
the dynamics are not the ones we want, but the implementation pattern transfers.

**Loizou, Vaswani, Laradji, Lacoste-Julien, "Stochastic Polyak Step-size for SGD: An Adaptive Learning
Rate for Fast Convergence", AISTATS 2021 (arXiv:2002.10542).**
What it shows: the per-example step size $\eta_i = (f_i(\theta) - f_i^{\star}) / (c \lVert \nabla
f_i(\theta)\rVert^2)$, which under interpolation converges fast without knowing problem constants. For
squared loss with $f_i^\star = 0$ and $c = 1$, that step size drives example $i$'s loss to exactly zero
along its own gradient direction. Relevance: this is the per-example rate control we want, in the
cheapest possible form — one scalar per example, no matrix. With $c$ chosen per step, the contraction
factor applied to example $i$ can be set to $1 - \eta_t$, giving Fact 4's product. The catch is that
Polyak steps are exact only for the example being stepped on: the update also moves the other examples
(the coupling term), which is why the exact preconditioned version of §5 is still needed for strict
uniformity.

**Strohmer and Vershynin, "A randomized Kaczmarz algorithm with exponential convergence", Journal of
Fourier Analysis and Applications 15(2):262-278, 2009 (arXiv:math/0702226).**
What it shows: for a consistent overdetermined linear system, projecting onto one randomly chosen row per
step — the row chosen with probability proportional to its squared norm — converges in expectation at a
linear rate $1 - \sigma_{\min}^2 / \lVert A \rVert_F^2$ that does not depend on the number of equations.
Relevance: the Kaczmarz projection is precisely "zero out this example's residual exactly, changing the
parameters as little as possible", which is the single-example version of the min-norm Gauss-Newton step.
A damped Kaczmarz step with relaxation $\eta$ multiplies that example's residual by $1-\eta$ and is the
natural stochastic implementation of our scheme when only one position is visited at a time.

**Needell, Srebro, Ward, "Stochastic Gradient Descent, Weighted Sampling, and the Randomized Kaczmarz
algorithm", NeurIPS 2014 (arXiv:1310.5715).**
What it shows: the formal bridge between SGD and randomized Kaczmarz, and the result that importance
sampling (reweighting the *sampling distribution*) is what buys the improved dependence on average rather
than maximum smoothness. Relevance: it is the reference for the exchange rate between "sample position
$i$ more often" and "take a bigger step on position $i$" — the key fact for the non-uniform-visitation
extension, because it says a non-uniform sampling density can be undone by per-example step scaling.

**Bernstein, Wang, Azizzadenesheli, Anandkumar, "signSGD: Compressed Optimisation for Non-Convex
Problems", ICML 2018.**
What it shows: taking only the sign of each coordinate of the stochastic gradient, with convergence
guarantees matching SGD rates under a Gaussian-noise assumption, and a momentum variant matching Adam.
Relevance to the "sign-MSE loss" idea in the brief: a sign-type loss makes each position's gradient
magnitude independent of its residual, so in a decoupled model the residual decreases *linearly* in $n$
and reaches zero in finite time. That is the wrong shape, and the same argument applies to any loss whose
per-position derivative does not vanish with the residual. See the homogeneity calculation in §3.

**Even-Dar and Mansour, "Learning Rates for Q-Learning", JMLR 5:1-25, 2003.**
What it shows: for tabular Q-learning with a *per-state-action* step size that depends on that pair's own
visit count, a polynomial rate $1/n_i^{\omega}$ with $\omega \in (1/2,1)$ gives a convergence time
polynomial in $1/(1-\gamma)$, while the linear rate $1/n_i$ gives an exponential dependence on
$1/(1-\gamma)$. Relevance: the canonical precedent that per-example step sizes indexed by that example's
own visit count are a normal, analysable construction, and a caution that the linear $1/n_i$ rate has
known weaknesses when the update is noisy (in our setting the target is deterministic, so that particular
objection does not apply).

**Bungert and Burger, "Asymptotic profiles of nonlinear homogeneous evolution equations of gradient flow
type", Journal of Evolution Equations 20:1061-1092, 2020.**
What it shows: for the gradient flow of an absolutely $\alpha$-homogeneous functional, the solution decays
in finite time, exponentially, or algebraically depending on $\alpha$, and rescaled solutions converge to
self-similar profiles; the scalar model problem is $a'(t) = -\lambda a(t)^{\alpha-1}$. Relevance: this is
the proper home of the loss-exponent calculation in §3, which is the basis of method M3 — it says the
*shape* of the decay is set by the homogeneity of the loss, independently of any schedule.

**Luo, Wen, Hu, Sun, Liu, Sun, Lyu, Chen, "A Multi-Power Law for Loss Curve Prediction Across Learning
Rate Schedules", arXiv:2503.12811 (March 2025; OpenReview KnoS9XxIlK).**
What it shows: an empirical law predicting the *pretraining loss curve* of a language model from the
learning-rate schedule, combining a power law in the summed learning rate with extra power-law terms for
the decay-induced loss drop; fitted on a few schedules, it predicts curves for unseen schedules and can
be optimised over schedules to beat cosine. Relevance: the only work I found that goes in the direction
"choose a schedule to obtain a target loss curve", but it is an aggregate empirical fit for language-model
pretraining, not a per-example construction, and it does not give the pointwise control we need.

## 3. Synthesis: three independent knobs

Reading everything above together, the design decomposes cleanly.

1. **Who shares a rate** is decided by the preconditioner, in output space. Exactly one construction
   makes all positions share a rate: an update whose effect in function space is
   $\Delta r_i = -\eta_i r_i$ with the *same* $\eta_i$ for all $i$, which is the minimum-norm
   Gauss-Newton or natural gradient step of Zhang, Martens and Grosse's eq. 3. Anything block-diagonal or
   dimension-factored (K-FAC, Shampoo, diagonal AdaGrad, Adam) leaves a residual condition number — for
   K-FAC the proven rate is explicitly $\bigl(1 - \eta/\lambda_{\max}(X^{\top}X)\bigr)^k$ on the squared
   error, not $(1-\eta)^k$.
2. **What shape that shared rate has** is decided either by the step-size schedule or by the homogeneity
   of the loss, and these are interchangeable.
   - Schedule route: contract by $1 - \eta_t$ with $\eta_t = 1/(2t)$, giving exactly
     $(\pi n)^{-1/2}$ (Fact 4).
   - Loss route: use a $p$-homogeneous per-position loss $\lVert r_i \rVert_2^{p}/p$ with a *constant*
     step. In the decoupled case the flow is $\mathrm{d}b/\mathrm{d}n = -\eta\, b^{p-1}$, so
     $b(n) = \bigl((p-2)\eta n + b(0)^{-(p-2)}\bigr)^{-1/(p-2)}$ for $p > 2$: a power law with exponent
     $-1/(p-2)$. Setting $p = 4$ gives exponent $-1/2$ exactly, i.e.
     $b(n) = (2\eta n + b(0)^{-2})^{-1/2}$. For $p < 2$ (including $p = 1$, the sign loss) the residual
     reaches zero in finite $n$; $p = 2$ is the geometric case. This is the discrete-time version of the
     Bungert-Burger classification.
3. **What level the curve sits at** is decided by the initial condition, and it is *free* on the schedule
   route but *automatic* on the loss route. On the schedule route, $b_i(n) = b_i(0)(\pi n)^{-1/2}$
   inherits the spread of $b_i(0)$, so requirement (1) has to be imposed by one extra exact solve at
   $n = 0$ that sets every residual to a common norm. On the loss route with $p = 4$ the term
   $b(0)^{-2}$ becomes negligible after a few steps and the curve converges to the *same*
   $(2\eta n)^{-1/2}$ regardless of $b(0)$ — the initial spread is forgotten. Choosing $\eta = 1/2$
   makes that universal curve exactly $n^{-1/2}$, which is $1$ at $n = 1$: requirements (1) and (2) are
   then both satisfied by the same constant.

The loss route also answers the non-uniform-visitation extension for free. Because the update rule
$b \mapsto b(1-\eta b^2)$ contains no reference to a global clock, a position that is only touched on its
own visits runs the same autonomous recursion in its *own* visit count $n_i$, so
$b_i \approx (2\eta n_i)^{-1/2}$ automatically. The schedule route can also do it, but only by keeping a
counter $n_i$ per position and using $\eta_{i,n} = 1/(2 n_i)$ — which is exactly the per-state-action
step of Even-Dar and Mansour, and exactly the "undo the density" that Basri et al.'s $\kappa^d/p(x)$
result says is required.

The remaining tension is generalisation. Exact per-position control at $m$ points says nothing about the
bonus between the points; that is fixed by the minimum-norm property of the update, i.e. by the tangent
kernel's smoothness. A kernel close to a multiple of the identity gives clean decoupling and near-perfect
rate uniformity but a bonus that is spiky and does not generalise; a wide, smooth kernel generalises but
couples the positions. The Fourier-feature bandwidth of Tancik et al. and the modified-spectrum
construction of Geifman et al. are the two documented ways to move along that trade-off deliberately.

## 4. Answers to the four questions, in short

1. **Full-matrix preconditioning.** Yes for exact natural gradient or exact (min-norm) Gauss-Newton:
   Zhang, Martens and Grosse's update contracts the *whole residual vector* by the same scalar per step,
   with the theorem stated as $\lVert u(k)-y\rVert_2^2 \le (1-\eta)^k\lVert u(0)-y\rVert_2^2$ under full
   row rank of the Jacobian and small Jacobian drift; at $\eta = 1$ on a linear model the residual is
   killed in one step. No for the practical approximations: K-FAC's proven rate carries
   $\lambda_{\max}(X^{\top}X)$, and Shampoo's guarantee is a convex regret bound with no per-example
   statement at all. On damping: the literature uses Levenberg-Marquardt adaptive damping (Martens 2010)
   or factored Tikhonov (K-FAC), both chosen for stability rather than for rate uniformity, and I found
   no paper that quantifies how damping degrades uniformity. Measured here on a small network
   (40 positions, 8 outputs, 840 parameters, damping $\lambda$ added to the $md\times md$ Gram matrix,
   step $1/(2t)$): per-position log-log slopes were $-0.5006 \pm 0.0015$ at $\lambda = 10^{-6}$,
   $-0.468 \pm 0.061$ at $\lambda = 10^{-2}$, and $-0.233 \pm 0.126$ at $\lambda = 1$. So damping must
   stay well below the smallest Gram eigenvalue, and the practical recipe is to solve the Gram system by
   conjugate gradients with a small fixed $\lambda$ and check the slope spread rather than to use an
   adaptive Levenberg-Marquardt rule tuned for loss reduction.
2. **Shaping the curve with a schedule on top of a preconditioned method.** The construction works and
   is exact (Fact 4). I did **not** find prior work that does this deliberately — no paper designs an
   iteration-dependent preconditioner or schedule so that a residual follows a prescribed curve
   pointwise. The nearest things are: Yao, Rosasco and Caponnetto's polynomially decaying step sizes on
   kernel gradient descent (analysed for generalisation error, not for residual shape); the spectral
   filtering catalogue (Lo Gerfo et al., Bauer et al.), which computes what shape a given filter produces
   rather than inverting the question; and the multi-power law (arXiv:2503.12811), which predicts an
   aggregate loss curve from a schedule for language-model pretraining. This looks like a genuine gap and
   is a defensible contribution of the project.
3. **Per-example reweighting.** Reweighting position $i$'s squared loss by $w_i$ replaces the tangent
   kernel $K$ by $KW$ with $W$ diagonal; van der Sluis's equilibration theorem bounds how much a diagonal
   scaling can improve the conditioning (within $\sqrt q$ of the diagonal optimum), and no diagonal $W$
   makes $KW$ a multiple of the identity unless $K$ is already diagonal. So reweighting alone equalises
   levels, not rates. What it *does* buy: the PINN weighting literature (Wang, Yu and Perdikaris; McClenny
   and Braga-Neto) and GradNorm show the practical machinery and the right normalisation (each group's
   progress measured relative to its own initial loss). The reweighting that actually changes the *shape*
   is not inverse-loss weighting but the opposite: a weight proportional to $\lVert r_i \rVert^2$, which
   is the $p=4$ homogeneous loss of §3 and slows down the positions that are furthest behind so all
   positions land on one universal curve. Sign-type losses (signSGD's per-coordinate sign, an L1 loss on
   the residual) give the wrong shape: constant-magnitude per-position gradients make the residual reach
   zero in finite $n$.
4. **Spectral filters.** No spectral filter of the kernel can produce pointwise $n^{-1/2}$ decay. A
   filter's residual factor is a function $q_n(\lambda)$ evaluated at each eigenvalue; requiring
   $q_n(\lambda) = n^{-1/2}$ for every $\lambda$ in the spectrum forces $q_n$ to be the constant function
   $n^{-1/2}$, i.e. $q_n(K) = n^{-1/2}I$, which is not a filter of $K$ at all but the inverse-kernel
   (Newton) iteration with the $1/(2t)$ schedule. Put the other way: uniform decay in all eigendirections
   is precisely the *absence* of spectral regularisation, which is why the statistical filtering
   literature never constructs it. On designing iteration-dependent preconditioners: the closest work is
   Geifman et al.'s modified-spectrum kernels, which reshape the eigenvalues to a desired sequence and
   run preconditioned gradient descent on the reshaped kernel; that gives control over *relative* rates
   but not over the curve in $n$.

## 5. Concrete methods to implement, in the order I would try them

### M1. Exact last-layer solve with a prescribed contraction (do this first)

The predictor's last layer is linear, and this is enough to control every position exactly. Write
$h(x)\in\mathbb{R}^{256}$ for the hidden activations, let $H \in \mathbb{R}^{m\times 257}$ stack
$[h(x_i), 1]$ over the $m$ positions, and let $W \in \mathbb{R}^{257\times 128}$ hold the last layer's
weights and bias. Then the residual matrix is $R = HW - Y$ with $Y$ the stacked targets, and the update

$$\Delta W = H^{+}\,\Delta R, \qquad \Delta R = -\,\mathrm{diag}(\eta_1,\dots,\eta_m)\, R$$

realises the prescribed per-position change **exactly**, because $H H^{+} = I_m$ whenever $H$ has full row
rank — which holds generically as long as $m \le 257$. Cost: one $m \times m$ pseudo-inverse; if the first
layer is frozen, $H^{+}$ is computed once and every step is two matrix products.

Recipe:

- Step 0: one exact solve $\Delta W = H^{+}(R^{\text{target}} - R)$ with $R^{\text{target}}_i$ the unit
  vector along $R_i$, so every bonus starts at exactly 1.
- Step $n$: $\eta_i = 1/(2n)$ for all $i$ (uniform visitation) or $\eta_i = 1/(2 n_i)$ with $n_i$ the
  position's own visit count (non-uniform).

Measured here on the real size (100 positions, $4\to256\to128$, frozen first layer): bonuses at $n=1$ in
$[0.99998, 1.00003]$; per-position log-log slopes over $n\in[300,3000]$ of $-0.500000$ with standard
deviation $2\times10^{-5}$; bonuses at $n=1000$ in $[1.7847\times10^{-2}, 1.7850\times10^{-2}]$ against
the prediction $(\pi\cdot 1000)^{-1/2} = 1.7841\times10^{-2}$. Under non-uniform visitation with a
Dirichlet(0.3) visit distribution, 8 samples per step, 20,000 steps and visit counts from 50 to 17,024,
the per-position slope *against that position's own count* was $-0.4993$ with standard deviation
$0.0006$ over the 76 positions with at least 50 visits.

### M2. Full-network damped Gauss-Newton with the $1/(2t)$ schedule

The same prescription applied to all parameters: $\Delta\theta = -\eta_n J^{\top}(JJ^{\top} + \lambda
I)^{-1} r$. This is Zhang-Martens-Grosse eq. 3 with damping. Solve the $md \times md$ system with
conjugate gradients using Jacobian-vector products (never form $J$); at $md = 12{,}800$ this is a few
hundred matrix-free products per step, so it is the expensive option, but it also trains the features,
which M1 with a frozen first layer does not.

Measured here on a small network (40 positions, 8 outputs, 840 parameters): per-position slopes
$-0.4991\pm0.0076$ at $\lambda=10^{-12}$, $-0.5006\pm0.0015$ at $\lambda=10^{-6}$,
$-0.4683\pm0.0613$ at $\lambda=10^{-2}$, $-0.2332\pm0.1255$ at $\lambda=1$; plain gradient descent with
the same $1/(2t)$ schedule gave $-0.120\pm0.051$. So the whole effect is the preconditioner and the
damping is the thing that destroys it.

### M3. The $p=4$ per-position loss (no schedule, self-normalising)

Replace the per-position squared loss by $\tfrac14 \lVert r_i \rVert_2^4$ — equivalently, weight each
position's squared loss by $\lVert r_i \rVert_2^2$ (a stop-gradient weight) — and use a **constant** step
$\eta = 1/2$ on top of the exact preconditioner of M1 or M2. The function-space update becomes
$\Delta r_i = -\eta \lVert r_i\rVert^2 r_i$, so each bonus follows
$b_i(n) = (2\eta n + b_i(0)^{-2})^{-1/2} \to n^{-1/2}$ for $\eta = 1/2$, regardless of where it started.

Why this is attractive: it satisfies requirement (1) and requirement (2) with one constant and no
bookkeeping. Every position converges onto the *same* curve within a few steps even when the initial
bonuses differ by a factor of two, and under non-uniform visitation each position runs the recursion in
its own visit count with no counters in the rule.

Measured here on the real size with the M1 solve, constant $\eta = 1/2$, no initial normalisation and
initial bonuses spread over $[1.90, 3.24]$: per-position slopes over $n\in[2000, 20000]$ of $-0.4996$
with standard deviation $3\times10^{-5}$, and at $n = 20{,}000$ the bonuses spanned
$[7.0698\times10^{-3}, 7.0700\times10^{-3}]$ — a spread of $1.0000\times$ — against the prediction
$(2\cdot 0.5\cdot 20000)^{-1/2} = 7.0711\times10^{-3}$. Under the same non-uniform visitation as M1
(Dirichlet(0.3), 8 visits per step, 20,000 steps, counts from 50 to 17,024) and **with no counters
anywhere in the rule**, the slope against each position's own visit count was $-0.4915$ with standard
deviation $0.0055$, and the level check $b_i\sqrt{n_i}$ came out at $0.9943 \pm 0.0069$ against the
predicted $1$.

Implementation notes: clip $\eta\lVert r_i\rVert^2$ below 1 so a large initial residual cannot overshoot;
when several visits to the same position land in one batch, apply the contraction once per visit (compose
the factors) rather than scaling the step by the multiplicity. **The preconditioner is not optional
here.** Measured on the same network: plain full-batch gradient descent on the $p=4$ loss diverged at both
$10^{-3}$ and $10^{-2}$ (the quartic gradient is cubic in the residual, so it explodes while the residual
is still of order 1), and Adam on the $p=4$ loss at $10^{-4}$ gave per-position slopes of
$-0.940 \pm 0.113$ with a final spread of $2.12\times$ across positions. The $p=4$ loss sets the *shape*
only once the preconditioner has already decoupled the positions.

### M4. Per-position $1/(2n_i)$ schedule for non-uniform visitation

If M3's constant-step form is not wanted, keep an integer visit counter per position and use
$\eta_i = 1/(2 n_i)$ inside M1 or M2. This is the same construction as tabular Q-learning's
per-state-action step (Even-Dar and Mansour) and it is what Basri et al.'s density result says is needed.
Measured numbers are in M1 above. The counter is only needed on the *training* positions; for a state
never seen, the bonus is whatever the min-norm interpolation gives, which is the RND behaviour we want.

### M5. Cheap approximations, and what to expect from them

K-FAC (Martens and Grosse 2015) and Shampoo (Gupta et al. 2018) are the standard cheap preconditioners.
Expect them to *narrow* the spread of per-position slopes relative to Adam without closing it: K-FAC's
own analysis gives a rate carrying $\lambda_{\max}(X^{\top}X)$, and both precondition in parameter space
by layer or by tensor dimension, so neither whitens the output-space residual. Worth running as a
baseline precisely to show the gap between an approximate and an exact preconditioner; the measurement to
report is the standard deviation of per-position slopes, not the aggregate slope.

### M6. Reshape the kernel instead of inverting it

Two documented levers, both cheaper than a solve:

- A random Fourier feature map on the 2-D position before the network (Tancik et al. 2020). Raising the
  bandwidth narrows the tangent kernel toward a multiple of the identity on the 100 positions, which
  narrows the eigenvalue spread and so the slope spread. Sweep the bandwidth and plot slope spread
  against a measure of how well the bonus interpolates between positions — that curve *is* the
  uniformity/generalisation trade-off, and it is a publishable figure on its own.
- Modified-spectrum kernels with preconditioned gradient descent (Geifman et al. 2023) if a target
  eigenvalue sequence is wanted rather than a bandwidth.

### M7. Per-position weights alone (the cheap thing that will partly work)

Weight each position's squared loss by $w_i$, updated to equalise either the loss levels (McClenny and
Braga-Neto's trainable weights) or the per-position gradient norms (GradNorm's relative inverse training
rate). Expect the *levels* to equalise and the *rates* to stay heterogeneous, for the van der Sluis
reason. Useful as an ablation that isolates how much of the observed heterogeneity is scale imbalance and
how much is genuine eigenvector mixing.

### M8. Feature learning with an exact correction (the practical compromise)

Alternate: (a) one ordinary Adam step on all parameters against the plain squared loss, to keep the
features useful and the bonus generalising; then (b) one exact last-layer solve (M1 or M3) that resets
every position's residual to the value the prescribed curve demands at the current step. Step (b) costs
one $m\times m$ solve and it *overrides* whatever step (a) did to the 100 controlled positions, so the
decay curve is exact by construction while the features still move. This is the version I would take into
a real RND run.

## 6. The checks I ran while writing this

All in double precision. The scripts are stored beside this file in
`06_preconditioning_checks/` and run under
`/p/rlprojects/RND/.venvs/exploration/bin/python`: `check_math.py` (schedule products and the AdaGrad
saturation), `modes.py` (per-mode exponents on a fixed kernel), `toy_rnd.py` and `toy_rnd3.py`
(Adam / plain SGD / exact last-layer solve at the real size, uniform and non-uniform visitation),
`toy_gn.py` (full-network Gauss-Newton damping sweep), `quartic.py` and `quartic2.py` (the $p=4$ loss).
Each is a few lines and worth re-running inside the project before relying on it.

| check | setting | result |
|---|---|---|
| $\prod_{t\le n}(1-\tfrac{1}{2t})$ against $(\pi n)^{-1/2}$ | closed form | ratio $0.99999$ at $n=2\cdot10^4$; log-log slope $-0.49996$ |
| $\eta_t = a/t$ on a fixed-kernel linear model | 20 modes, eigenvalues $10^{-3}$ to $1$, $a=0.5$ | per-mode slope $= -a\lambda_j$ to 4 digits; per-position slope spread $\mathrm{sd}=0.44$ early, drifting to the slowest mode late |
| AdaGrad-style accumulation, decoupled quadratic | $\eta=0.1$ | accumulator saturates; residual reverts to geometric decay, local slope $-463$ |
| Adam, constant step, real size (100 positions, $4\to256\to128$) | $10^{-3}$, 3000 steps | aggregate slope $-0.571$; per-position mean $-0.570$, $\mathrm{sd}\,0.137$, range $[-1.02,-0.23]$ |
| plain SGD, $\eta_t=\min(a/t, 0.02)$, loss averaged over positions | $a\in\{0.2,0.5,1\}$, 5000 steps | aggregate $-0.055$ to $-0.073$; per-position $\mathrm{sd}\approx0.023$ — the achieved exponent depends on $a$ and on the loss normalisation |
| M1, exact last-layer solve, $\eta_t=1/(2t)$ | real size, frozen first layer | slopes $-0.500000\pm2\times10^{-5}$; bonus at $n{=}1$ in $[0.99998,1.00003]$ |
| M1 under non-uniform visits, $\eta_i = 1/(2n_i)$ | Dirichlet(0.3), 8 visits/step, 20k steps | slope against own count $-0.4993\pm0.0006$ over 76 positions with counts 50 to 17,024 |
| M2, full-network Gauss-Newton, damping sweep | 40 positions, 8 outputs, 840 parameters | $\lambda=10^{-6}$: $-0.5006\pm0.0015$; $\lambda=10^{-2}$: $-0.468\pm0.061$; $\lambda=1$: $-0.233\pm0.126$; plain GD: $-0.120\pm0.051$ |
| M3, $p{=}4$ loss, constant $\eta=1/2$, no schedule | real size, initial bonuses $[1.90,3.24]$ | slopes $-0.4996\pm3\times10^{-5}$; final bonuses spread $1.0000\times$ across positions |
| M3 under non-uniform visits, no counters used | Dirichlet(0.3), 8 visits/step, 20k steps | slope against own count $-0.4915\pm0.0055$; level $b_i\sqrt{n_i} = 0.9943\pm0.0069$ |
| $p{=}4$ loss WITHOUT preconditioning | real size | plain gradient descent diverged at $10^{-3}$ and $10^{-2}$; Adam at $10^{-4}$ gave $-0.940\pm0.113$, final spread $2.12\times$ |

## 7. What I could not find

- No paper that designs a schedule or an iteration-dependent preconditioner so that a residual follows a
  *prescribed* curve pointwise. Spectral filtering theory computes the curve a filter produces; the
  multi-power law predicts an aggregate loss curve from a schedule; neither inverts the problem per
  example.
- No paper that measures how Tikhonov or Levenberg-Marquardt damping degrades the uniformity of the
  Gauss-Newton contraction across examples. The damping in Martens 2010 and in K-FAC is chosen for
  stability and loss reduction. The sweep in §6 is a substitute measurement, on a tiny network.
- No analysis of per-example loss reweighting written in the tangent-kernel language with an explicit
  statement of what diagonal weights can and cannot do to the per-example rates; van der Sluis's
  equilibration bound is the right tool but I found no one who applies it to this question.

# Loss-landscape geometry and architecture choices, read against the uniform $n^{-1/2}$ bonus-decay problem

Scope: what the loss-landscape literature (starting from Li et al. 2018, which the user pointed at)
and the architecture/optimization literature actually say, and which of it bears on making the
per-position residual norm $b_i(n) = \lVert g_n(x_i) - f(x_i) \rVert_2$ decay as $n^{-1/2}$ at
*every* one of the roughly 100 maze positions at once.

Every rate below names the quantity it applies to (squared loss, residual norm, or one eigen-mode
of the residual) and the assumptions under which it holds.

---

## 0. The problem restated in the form the literature can answer

This section is not from a paper; it is the algebra that makes the rest of the notes usable. The
papers are then read against it.

Setting: predictor $g$ with parameters $\theta$, frozen target $f$, fixed point set
$x_1,\dots,x_N$ with $N \approx 100$, output dimension 128, full-batch gradient descent on
$L = \dfrac12 \sum_i \lVert g(x_i) - f(x_i) \rVert_2^2$.

Define the residual matrix $R(n) \in \mathbb{R}^{N \times 128}$ with rows
$r_i(n) = g_n(x_i) - f(x_i)$, so $b_i(n) = \lVert r_i(n) \rVert_2$. Define the empirical neural
tangent kernel matrix $K \in \mathbb{R}^{N \times N}$ with
$K_{ij} = \langle \nabla_\theta g(x_i), \nabla_\theta g(x_j) \rangle$ (summed over the 128 output
heads, which for standard parameterization act independently, so $K$ is a scalar kernel and the
128 output coordinates all follow the same $N \times N$ dynamics).

In the lazy / kernel regime (Jacot et al. 2018; Lee et al. 2019 — see §3), one gradient step with
step size $\eta$ gives, to first order,

$$R(n+1) = (I - \eta K)\, R(n).$$

Diagonalize $K = \sum_k \lambda_k v_k v_k^\top$. Then

$$r_i(n) = \sum_k \Big[\textstyle\prod_{s\le n}(1 - \eta_s \lambda_k)\Big]\, v_{k,i}\, c_k,
\qquad c_k = v_k^\top R(0) \in \mathbb{R}^{128}.$$

Three consequences, and they are the whole story:

1. **Constant step size gives exponential, not power-law, decay of $b_i$.** Each mode contracts by
   $(1-\eta\lambda_k)$ per step. On a log-log plot this looks like a slope that steepens and then
   saturates, because once the fast modes are dead only $\lambda_{\min}$ survives. This is exactly
   the reported "constant-rate Adam decays too fast (slope about $-0.8$), saturating".
2. **A $1/n$ step schedule converts exponential decay into a power law, one exponent per mode.**
   With $\eta_n = c/n$,
   $\prod_{s=n_0}^{n} (1 - c\lambda_k/s) \approx \exp(-c\lambda_k \ln(n/n_0)) = (n/n_0)^{-c\lambda_k}$.
   So mode $k$ of the **residual** decays as $n^{-c\lambda_k}$, and the **squared** loss carried by
   that mode as $n^{-2c\lambda_k}$. The aggregate slope of $-0.503$ that plain SGD with a $1/t$
   schedule produced is the slope of whichever modes dominate the total; it does not imply any
   individual position has that slope.
3. **Uniform per-position exponent $-1/2$ requires $K$ to be a scalar multiple of the identity on
   the point set.** Position $i$'s residual is a mixture $\sum_k n^{-c\lambda_k} v_{k,i} c_k$
   weighted by the eigenvector entries $v_{k,i}$. Different positions weight modes differently, so
   as long as the $\lambda_k$ differ, per-position log-log slopes differ, and each position's slope
   drifts over $n$ toward $-c\lambda_{\min}$. To make every position obey $b_i(n) \propto n^{-1/2}$
   under a global $1/n$ schedule you need $c\lambda_k = 1/2$ for every $k$ that the initial residual
   excites — i.e. $K \propto I_N$ restricted to the span of $R(0)$. Since $R(0)$ (a random target
   minus a random predictor) generically excites all modes, the requirement is
   $K \approx \dfrac{1}{2c} I_N$.

So the target of every method below is one of:

- **(A)** make $K$ close to $\kappa I_N$ (equal eigenvalues), or
- **(B)** cancel the spread of $\lambda_k$ by preconditioning the update, or
- **(C)** change the *loss* so the power law is produced per position without needing equal
  eigenvalues (see §7, method M1 — this falls out of the same ODE and does not appear in the
  papers below).

Two smaller requirements come along for free from the same picture:

- **"Starts at 1 at every position"** is a statement about $\lVert r_i(0)\rVert$, i.e. about
  initialization and input scaling, not about the optimizer.
- **Non-uniform visitation**: sampling position $i$ with probability $p_i$ replaces $K$ by
  $K \operatorname{diag}(p)$ in expectation. If $K$ is diagonal, position $i$ moves only when $i$ is
  sampled, so its residual is a function of its own visit count $n_i$ and nothing else. If $K$ is
  not diagonal, visits to $j$ move $i$, and no per-position schedule can fix that. Near-diagonality
  of $K$ is therefore the *precondition* for the harder extension, not an optional extra.

---

## 1. Loss-landscape geometry

### 1.1 Li, Xu, Taylor, Studer, Goldstein — *Visualizing the Loss Landscape of Neural Nets* (NeurIPS 2018, arXiv:1712.09913)

This is the paper behind the Virginia Tech blog post the user linked. What it actually does:

- **Filter normalization (the methodological contribution).** To plot
  $L(\theta^\star + \alpha \delta + \beta \eta)$ for random directions $\delta,\eta$, they first
  rescale each direction filter by filter: for layer $i$, filter $j$,
  $d_{i,j} \leftarrow (d_{i,j} / \lVert d_{i,j}\rVert)\,\lVert \theta_{i,j}\rVert$ (Frobenius norms).
  The reason is scale invariance: "When ReLU non-linearities are used, the network remains unchanged
  if we multiply the weights in one layer by 10, and divide the next layer by 10", and with batch
  normalization weight magnitude has no effect on the function. Without filter normalization, the
  apparent sharpness of a minimizer is an artifact of weight scale, and two minimizers cannot be
  compared. This is a plotting convention, not a training method.
- **The depth transition.** For VGG-style plain networks on CIFAR-10, "as network depth increases,
  the loss surface of the VGG-like nets spontaneously transitions from (nearly) convex to chaotic."
  Concretely: ResNet-20 without skip connections shows a "fairly benign landscape dominated by convex
  contours"; ResNet-56 without skips shows "dramatic non-convexities"; ResNet-110 without skips is
  "extremely steep". They report SGD could not train a 156-layer network without skip connections.
- **What skip connections do.** "Shortcut connections prevent the transition to chaotic behavior as
  depth increases. In fact, the width and shape of the 0.1-level contour is almost identical for the
  20- and 110-layer networks." So the claimed mechanism is *depth-independence of the landscape*, not
  a general smoothing that applies at any depth.
- **What width does.** "Wider models have loss landscapes with no noticeable chaotic behavior.
  Increased network width resulted in flat minima and wide regions of apparent convexity."
- **Hessian check.** They compute the ratio $|\lambda_{\min}|/\lambda_{\max}$ of the Hessian over the
  same 2-D plane and color it. The convex-looking regions really do have negligible negative
  curvature (for DenseNet, negative eigenvalues "less than 1% the size of positive curvatures" over a
  large region); the chaotic-looking regions really do have large negative curvature. So the pictures
  are not pure artifacts.
- **Where the non-convexity sits.** Chaotic structure is concentrated in the high-loss periphery, not
  around the minimizer; they argue "a random initialization will likely lie in the well-behaved loss
  region", which is their explanation for why initialization matters.
- **Generalization claim.** With filter normalization, visual sharpness "correlates extremely well
  with test error", including across architectures. This is a correlational claim about
  generalization, and is not about training-point convergence rates.

**Relevance to us, stated plainly.** The paper contains no statement about per-example convergence,
per-example loss decay, or uniformity across training points. Its subject is trainability of *very
deep* networks (56–110+ layers) with *narrow* layers. Our predictor is 2 layers and 256 units wide on
100 points; both of the paper's own axes (depth small, width large) put us in the regime it calls
benign. The paper is therefore evidence *against* the loss landscape being our problem, not for it.

### 1.2 Böttcher, Wheeler — *Visualizing high-dimensional loss landscapes with Hessian directions* (J. Stat. Mech. 2024, arXiv:2208.13219)

A direct caution on how far to trust §1.1's pictures. They show that under random 2-D projections,
saddle points of the original high-dimensional loss are "rarely correctly identified as such"; the
expected curvature in the projection is proportional to the *mean* curvature of the original surface,
so a saddle typically renders as a minimum, a maximum, or a flat region depending on the mean
curvature's sign. They propose projecting along the top and bottom Hessian eigenvectors instead,
noting that a plane spanned by the extreme eigenvectors is the one that actually exposes
ill-conditioning. Takeaway for us: a 2-D random-direction picture is not a measurement of
conditioning, and conditioning is the quantity our problem depends on. If we ever want a landscape
diagnostic, take the eigenvalues of $K$ (or of the Hessian in function space), not a contour plot.

### 1.3 Santurkar, Tsipras, Ilyas, Madry — *How Does Batch Normalization Help Optimization?* (NeurIPS 2018, arXiv:1805.11604)

Shows that batch normalization's benefit is not distributional stability of layer inputs ("internal
covariate shift") — they inject noise after the normalization to destroy that stability and training
still improves — but that batch normalization makes the optimization landscape smoother, i.e. improves
the Lipschitz constant of the loss and of its gradient, so gradients become more predictive of the
loss a step ahead and larger step sizes are stable. The claims are about the *loss along the
optimization path*, aggregated over the batch. There is no per-example statement. Its use to us is
indirect: a normalization layer is a cheap way to change the curvature structure, and §5 below gives
the version of that claim that is actually about the kernel spectrum.

### 1.4 Schoenholz, Gilmer, Ganguli, Sohl-Dickstein — *Deep Information Propagation* (ICLR 2017, arXiv:1611.01232)

Worth naming because the word "chaotic" is used in two different technical senses in this
literature and they should not be conflated. Here "chaos" is the mean-field order-to-chaos phase
transition of a random network's forward signal propagation: there are depth scales that bound how
far correlation information travels, "random networks may be trained precisely when information can
travel through them", and only at the edge of chaos does the relevant depth scale diverge so that
arbitrarily deep networks are trainable. This is a statement about *random initialization of deep
networks*, again about depth. Li et al.'s "chaotic" is a visual description of a 2-D slice. Neither
applies to a 2-layer predictor.

---

## 2. Does anything make convergence *uniform across training points*?

This is the question the user actually asked (Q2). Honest summary first: **I did not find a paper
that measures the spread of per-example loss-decay exponents as a function of architecture.** The
literature that bears on it does so through one of two proxies — pairwise gradient alignment, or the
spread of the kernel's eigenvalues. Both proxies are exactly the quantities §0 says matter, so the
evidence is usable, but it is indirect and none of it is in a full-batch distillation setting.

### 2.1 Sankararaman, De, Xu, Huang, Goldstein — *The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent* (ICML 2020, arXiv:1904.06963)

The most directly relevant paper on this list, because "gradient confusion" is literally the
off-diagonal of the per-example gradient Gram matrix, which is the off-diagonal of $K$.

- **Definition 2.1.** A set of objectives $\{f_i\}_{i\in[N]}$ has gradient confusion bound
  $\eta \ge 0$ at $w$ if $\langle \nabla f_i(w), \nabla f_j(w)\rangle \ge -\eta$ for all $i \ne j$.
  They also state (their §8) that all results carry over to the average-inner-product version
  $\sum_{i,j}\langle \nabla f_i,\nabla f_j\rangle / N^2 \ge -\eta$ and to the cosine-normalized
  version.
- **Theorems 3.1 / 3.2 (convergence of the *objective*, not per example).** Under Lipschitz
  smoothness and the Polyak-Łojasiewicz condition, constant-step SGD satisfies
  $\mathbb{E}[F(w_T) - F^\star] \le \rho^T (F(w_0)-F^\star) + \alpha\eta/(1-\rho)$; for smooth
  non-convex objectives, $\min_k \mathbb{E}\lVert\nabla F(w_k)\rVert^2 \le \rho(F(w_1)-F^\star)/T + \rho\eta$.
  So gradient confusion sets the size of the neighborhood SGD stalls in, and if $\eta = O(\epsilon)$
  then $T = O(\log(1/\epsilon))$ iterations suffice without shrinking the step size.
- **The illustrative special case is the one we want.** For a linear model on *orthogonal* inputs,
  $\langle \nabla f_i, \nabla f_j\rangle = \zeta_i\zeta_j\langle x_i,x_j\rangle = 0$, so "an update in
  the gradient direction $f_i$ has no effect on the loss value of $f_j$", and "SGD decouples into
  (deterministic) gradient descent on each objective term separately". That decoupled regime is
  precisely the regime in which each position can decay at its own controlled rate — and in which the
  non-uniform-visitation extension becomes possible at all.
- **Theorem 4.1 (width and depth at Gaussian initialization).** The probability that the confusion
  bound holds is at least
  $1 - \beta e^{-c_1\kappa^2\ell^2} - N^2\exp\!\big(-c_2(\ell d + \ell^2\beta)\eta^2 / (16\zeta_0^4(\beta+2)^4)\big)$
  for data on the unit sphere, with depth $\beta$, max width $\ell$, $\zeta_0 = 2\sqrt{\beta}$. Width
  $\ell$ enters positively, depth $\beta$ negatively (the $(\beta+2)^4$ in the denominator). Their
  summary: "as the depth increases (with fixed width), training a model becomes harder, while as the
  width increases (with fixed depth), training a model becomes easier."
- **Theorem 6.1 (orthogonal initialization).** For a deep *linear* network with orthogonal weight
  matrices and $\gamma = 1/\sqrt{2\beta}$ rescaling, the bound holds with probability at least
  $1 - N^2 e^{-cd\eta^2}$ — no dependence on depth $\beta$ or width $\ell$ at all.
- **Empirics.** They measure the full distribution of pairwise gradient cosine similarities (100
  minibatch pairs per epoch, batch size 128) at the end of training. Deeper CNNs have a visibly wider
  distribution with a more negative minimum; wider ones a narrower one. Adding skip connections *or*
  batch normalization individually helps but the confusion still worsens with depth; "when these
  techniques are used together, the model has relatively low gradient confusion even for very deep
  networks".

**Use to us.** This gives (i) a measurable diagnostic we can run directly — the histogram of
$K_{ij}/\sqrt{K_{ii}K_{jj}}$ over our 100 positions — and (ii) the finding that width and
normalization-plus-skip reduce the off-diagonal mass. It does *not* say anything about the spread of
the diagonal $K_{ii}$ or of the eigenvalues, which for us matters just as much.

### 2.2 Daneshmand, Joudaki, Bach — *Batch Normalization Orthogonalizes Representations in Deep Random Networks* (NeurIPS 2021, arXiv:2106.03970)

Proves that in a deep random network with batch normalization, the hidden representations of the
examples in a batch become increasingly *orthogonal to each other* with depth: the deviation from
orthogonality decays rapidly with depth down to a floor inversely proportional to width, and the
distribution of post-linear-layer representations contracts to a Wasserstein-2 ball around an
isotropic Gaussian whose radius shrinks with width. They also observe empirically that when
representations start out aligned, SGD spends many iterations orthogonalizing them before making
progress on the task.

**Use to us.** This is the mechanism that could produce $K \approx \kappa I$ directly: if the last
hidden-layer features of the $N$ points are mutually orthogonal and of equal norm, the feature Gram
matrix is $\kappa I$, and (for a linear readout) so is the kernel. The theorem is asymptotic in depth
and is about batch normalization across a batch, so at 2 layers we cannot expect the effect from
depth alone — but it argues that *explicitly* whitening the features across the 100 points (method M4
below) is doing the same thing the depth limit does, only exactly and in one step.

### 2.3 Bengio, Pineau, Precup — *Interference and Generalization in Temporal Difference Learning* (ICML 2020, arXiv:2003.06350) and Liu, Wang, Tao, Javed, White, White — *Measuring and Mitigating Interference in Reinforcement Learning* (CoLLAs 2023, PMLR v232, arXiv:2307.04887)

Both define interference as the inner product of the gradients of two examples — the same object as
gradient confusion, in the RL setting. Bengio et al. find TD learning drives parameters to a
low-interference, under-generalizing regime, opposite to supervised learning; Liu et al. define a
measure for value-based methods and show it correlates with control instability across architectures.
For us their contribution is vocabulary and the confirmation that "off-diagonal of the gradient Gram
matrix" is the standard handle for cross-example coupling, plus the reminder that *low* interference
is not automatically good (it trades against generalization between positions, which for an
exploration bonus field is a real cost, see §7 caveats).

### 2.4 Pan, Banman, White — *Fuzzy Tiling Activations* (ICLR 2021, arXiv:1911.08068)

Proposes an activation that produces a sparse, tile-coding-like representation "by design rather than
by learning", with controllable sparsity, usable online as a drop-in activation. Their motivation is
exactly that sparse representations reduce interference between updates for different inputs; they
report more stable policies and reduced need for target networks. For us this is the cheapest
concrete way to push $K$ toward diagonal: a sparse, localized hidden representation makes
$K_{ij}\approx 0$ for $i\ne j$ almost by construction.

### 2.5 Baldock, Maennel, Neyshabur — *Deep Learning Through the Lens of Example Difficulty* (NeurIPS 2021, arXiv:2106.09647)

Introduces "prediction depth" — the layer after which a $k$-NN probe's prediction stops changing —
as a per-example difficulty measure, and relates it to per-example confidence, accuracy and *speed of
learning*. The unifying observations they list include "early layers converge faster" and "networks
learn easy data and simple functions first". This is the clearest documented statement that
per-example learning speed is systematically heterogeneous in ordinary training, and that the
heterogeneity is a property of where the example sits relative to the others. It is a classification
paper with no rate statements and no architecture intervention, so it supports "non-uniform decay is
the default" but offers no fix.

---

## 3. Is a 4 → 256 → 128 network's *landscape* the problem, or is it the kernel spectrum?

### 3.1 Jacot, Gabriel, Hongler — *Neural Tangent Kernel* (NeurIPS 2018, arXiv:1806.07572) and Lee, Xiao, Schoenholz, Bahri, Novak, Sohl-Dickstein, Pennington — *Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent* (NeurIPS 2019, arXiv:1902.06720)

Establish that in the infinite-width limit the function evolves under gradient descent as a linear
model with a fixed kernel, so the training dynamics are exactly the linear recursion in §0. Lee et al.
give the finite-width version: sufficiently wide networks track their linearization throughout
training. This is what licenses the entire §0 analysis — but it also carries the caveat that it must
be *checked* at width 256, not assumed.

### 3.2 Du, Zhai, Póczos, Singh — *Gradient Descent Provably Optimizes Over-parameterized Neural Networks* (ICLR 2019, arXiv:1810.02054)

For a two-layer ReLU network with $m$ hidden units on $n$ samples: if no two inputs are parallel and
$m$ is large enough, randomly initialized gradient descent converges to a *global* minimum at a
**linear rate on the training loss** — the squared loss contracts geometrically. The mechanism they
identify is that over-parameterization plus random initialization keep every weight vector near its
initialization for all iterations, which gives a strong-convexity-like property in function space.
The rate is governed by the least eigenvalue $\lambda_0 = \lambda_{\min}(H^\infty)$ of the limiting
Gram matrix.

**Direct answer to the user's Q3.** For our architecture (two layers, one hidden layer of 256, 100
points, no two positions parallel after the input embedding), this says the landscape is *not* the
obstacle: gradient descent reaches zero training loss and the only quantity in the rate is a kernel
eigenvalue. Two honest caveats: (i) the width requirement in these proofs is enormous (Du et al. need
$m$ polynomial in $n$ with a large exponent; Arora et al. below state $m = \Omega(n^7/(\lambda_0^4\kappa^2\delta^4\epsilon^2))$)
and 256 is far below it, so the theorem is a *guide*, not a certificate at our size; (ii) our
parameter count is about $4\cdot256 + 256 + 256\cdot128 + 128 \approx 34{,}000$ against
$100 \times 128 = 12{,}800$ residual constraints, roughly $2.7\times$ overparameterized — enough to
interpolate, but not so much that laziness is guaranteed. The check to run is whether $K$ measured at
initialization and at the end of training are close (kernel drift).

### 3.3 Arora, Du, Hu, Li, Wang — *Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks* (ICML 2019, arXiv:1901.08584)

This paper supplies the exact per-mode statement §0 is built on. Their Theorem 4.1: with
$\lambda_0 = \lambda_{\min}(H^\infty) > 0$, sufficiently large width, and small enough step size,
with high probability, for all $k$,

$$\lVert y - u(k)\rVert_2 = \sqrt{\textstyle\sum_{i=1}^{n} (1-\eta\lambda_i)^{2k}\,(v_i^\top y)^2} \;\pm\; \epsilon,$$

where $H^\infty = \sum_i \lambda_i v_i v_i^\top$. Their own reading: decompose the label vector into
projections onto the kernel eigenvectors; "the $i$-th portion shrinks exponentially at ratio
$(1-\eta\lambda_i)^2$"; convergence is fast when the labels align with top eigenvectors and slow when
the projections are uniform across eigenvectors or concentrated on small eigenvalues. They verify the
eigenvalue decay of $H^\infty$ and the projection profile on MNIST/CIFAR: true labels align with the
top eigenvectors, random labels spread uniformly.

**Use to us.** This is the theorem to cite for "per-position heterogeneity is a kernel-spectrum
phenomenon". Note the setting is scalar output and the statement is about the *aggregate* residual
norm; the per-coordinate version $[y-u(k)]_j = \sum_i (1-\eta\lambda_i)^k (v_i^\top y) v_{i,j}$ is an
immediate consequence of the same linear recursion they prove tracks the network, and it is the
per-position statement we need. Also note our "labels" are a random target network's outputs, which
is the *worst* case in their taxonomy: a random target projects roughly uniformly onto all
eigenvectors, so every mode is excited and the eigenvalue spread is maximally exposed.

### 3.4 Cho, Saul — *Kernel Methods for Deep Learning* (NeurIPS 2009)

The arc-cosine kernel: $K_n(x,y) = \dfrac{1}{\pi}\lVert x\rVert^n \lVert y\rVert^n J_n(\vartheta)$
with $J_1(\vartheta) = \sin\vartheta + (\pi - \vartheta)\cos\vartheta$, where $\vartheta$ is the angle
between $x$ and $y$. The $n=1$ case is the ReLU kernel.

**This is probably the single most actionable fact in these notes for our setting.** A ReLU network
without biases is positively homogeneous of degree 1 in its input, so both the kernel and the network
output scale with $\lVert x\rVert$. With input $[x,y,0,0]$ and raw maze coordinates, the diagonal
satisfies $K_{ii} \propto \lVert x_i \rVert^2$. A position near the coordinate origin therefore has a
small $K_{ii}$, receives a proportionally small update per step, and its bonus decays *slower* than a
far-away position — a per-position rate difference produced entirely by where we put the origin, with
no learning involved. Two positions at radius 1 and radius 3 differ by a factor of 9 in their
self-rate. The same homogeneity also makes $\lVert r_i(0)\rVert$ vary with $\lVert x_i\rVert$, which
breaks requirement (1), "starts at the same value". Fix: put all inputs on a common norm (normalize
to the unit sphere, or use a sinusoidal / Fourier embedding, or at minimum include bias terms and
center the coordinates). Check this before anything else.

### 3.5 Moulines, Bach — *Non-Asymptotic Analysis of Stochastic Approximation Algorithms for Machine Learning* (NeurIPS 2011)

The classical statement matching consequence 2 of §0: a step size proportional to $1/n$ attains the
optimal rate in the strongly convex case but "is not robust to the lack of strong convexity or the
setting of the proportionality constant" — the achieved exponent depends on the product of the
constant with the curvature, so a mis-set constant degrades the rate. Their robust alternative is a
slower schedule $\eta_n \propto n^{-\alpha}$, $\alpha \in (0,1)$, with Polyak-Ruppert averaging.

**Use to us.** This is the formal reason the $1/t$ schedule produces different exponents at different
positions: the exponent is $c\lambda_k$, a product of *our* constant with a curvature we did not
choose. It also warns that any solution resting on tuning $c$ to hit $-1/2$ is fragile to anything
that shifts the spectrum (input scale, width, initialization seed).

---

## 4. Spectral bias, input density, and *per-point* rates

### 4.1 Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio, Courville — *On the Spectral Bias of Neural Networks* (ICML 2019, arXiv:1806.08734)

Deep ReLU networks fit low Fourier frequencies of the target first; they "cannot have local
fluctuations without affecting their global behavior". They also show that learning high frequencies
gets *easier* as the data manifold becomes more complex/curved. Relevant to us because a random
target network sampled at 100 scattered positions is a high-frequency target relative to the point
spacing, so the low-frequency modes of $K$ (large $\lambda$) carry little of the residual and the
slow, small-$\lambda$ modes carry a lot — the bad case in Arora et al.'s taxonomy.

### 4.2 Basri, Galun, Geifman, Jacobs, Kasten, Kritchman — *Frequency Bias in Neural Networks for Input of Non-Uniform Density* (ICML 2020, arXiv:2003.04560)

The only paper I found that states a **per-point** convergence rate as a function of the *sampling
density*, which is exactly the harder extension in our problem.

- Setting: NTK regime, two-layer ReLU with only the first layer trained, and deep fully connected
  networks; gradient descent on the squared loss; inputs on $S^{d-1}$ with a non-uniform density
  $p(x)$.
- Result (1-D, piecewise-constant density, their Theorem 1): the number of iterations to reach
  $\lVert g(x) - u^{(t)}(x)\rVert < \delta$ for a pure harmonic of frequency $\kappa$ is
  $\tilde{O}(\kappa^2/p^\star)$ with $p^\star$ the minimum density; the conjectured $d$-dimensional
  form is $O(\kappa^d/p^\star)$, and the local statement is that "convergence at a point
  $x\in S^{d-1}$ occurs in time $O(\kappa^d/p(x))$".
- Mechanism: under a piecewise-constant density the NTK eigenfunctions become
  $f(x) = a(p(x))\cos(\dfrac{q}{Z}\Psi(x) + b(p(x)))$ with $\Psi(x)=\int p(\tilde x)^{1/2}d\tilde x$ —
  i.e. locally sinusoidal with a *local* frequency scaling as $\sqrt{p_j}$ in region $j$. So the
  kernel's eigenstructure is reshaped by the density, it is not merely reweighted.
- Empirics (their Figure 7): each region converges at a time proportional to $\kappa^2/p_j$; dense
  regions converge substantially faster than sparse ones for the same target frequency.

**Use to us.** Two readings, and both matter.
- *Encouraging*: progress at a point is proportional to how often that point is sampled, which is
  qualitatively the "decay with your own visit count" behaviour we want.
- *Discouraging*: the effect is on the *time constant* only. In an exponential-decay picture,
  $b_i(t) \approx \exp(-c\,p_i t)$ — same functional form everywhere, different rate constant. What
  we want is $b_i \propto n_i^{-1/2}$, i.e. the same *exponent* with a $p_i$-dependent prefactor. In
  the $1/n$-schedule picture of §0, a density factor turns the exponent into $c\lambda_k p$-like
  quantities, which spreads the exponents rather than the prefactors. So Basri et al. tell us the
  density does the right thing to the *rate constant* and the wrong thing to the *exponent*, and that
  a per-position counter is still needed.

### 4.3 Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi, Barron, Ng — *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains* (NeurIPS 2020, arXiv:2006.10739)

Passing coordinates through $\gamma(x) = [\cos(2\pi B x), \sin(2\pi B x)]$ with $B$ drawn from a
Gaussian of scale $\sigma$ before the MLP "transforms the effective NTK into a stationary kernel with
a tunable bandwidth"; a standard coordinate MLP fails to fit high frequencies both in theory and
practice. The scale $\sigma$ is the knob, and they report the choice of $\sigma$ matters much more
than the number of features.

**Use to us — this is the most promising single architectural change.** Two properties we want fall
out at once. (i) *Stationary*: the kernel depends on $x_i - x_j$ only, so $K_{ii}$ is the **same
constant for every position**, which removes the $\lVert x\rVert^2$ heterogeneity of §3.4 and equalizes
the per-position self-rate exactly. (ii) *Tunable bandwidth*: as $\sigma$ grows, the kernel narrows
toward a delta, so $K \to \kappa I$ on any fixed point set — which is exactly condition (A) of §0, and
simultaneously the near-diagonality the non-uniform-visitation extension needs. The cost is that the
bonus field stops interpolating between grid positions (a wide-bandwidth kernel generalizes; a narrow
one memorizes), so $\sigma$ is a genuine tradeoff dial between uniform decay and a spatially smooth
bonus field, and should be swept.

### 4.4 Wang, Wang, Perdikaris — *On the eigenvector bias of Fourier feature networks* (CMAME 384:113938, 2021, arXiv:2012.10047)

Reframes spectral bias as "NTK eigenvector bias": networks learn along the dominant eigen-directions
of their limiting NTK, and a Fourier feature mapping *modulates the frequency of the NTK
eigenvectors*. They build multi-scale Fourier feature architectures (several $\sigma$ values
concatenated) so that several frequency bands are learned at comparable speed rather than
sequentially.

**Use to us.** This is the paper that states our goal in its own vocabulary — equalizing the speed at
which different eigen-directions are learned — and gives an architectural, rather than
optimizer-side, way to do it. The multi-scale variant is worth testing directly: a single $\sigma$
sets one bandwidth, and if our 100 positions are unevenly spaced, one bandwidth cannot make $K$
diagonal everywhere.

### 4.5 Wang, Yu, Perdikaris — *When and why PINNs fail to train: a neural tangent kernel perspective* (J. Comput. Phys. 449:110768, 2021, arXiv:2007.14527)

Derives the NTK of a physics-informed network, shows it converges to a deterministic kernel and
barely changes during training as width grows, and diagnoses training failure as a *discrepancy in
the convergence rates of the different loss terms*, caused by their NTK eigenvalues differing by
orders of magnitude. Their fix is an algorithm that reweights the loss terms using the NTK's own
traces so the back-propagated gradient magnitudes of the different terms are balanced, recomputed
periodically during training.

**Use to us.** This is the closest existing method to what we need, one level of granularity coarser:
they equalize convergence rates across a handful of *loss terms*; we want to equalize across 100
*positions*. The generalization is immediate — weight position $i$'s squared error by
$w_i \propto 1/K_{ii}$ (or by the trace of its own kernel block), recomputed every few hundred steps.
With $N=100$ this is cheap.

---

## 5. What residual connections and normalization do to the *kernel*, as opposed to the picture

This is the part of the architecture literature that speaks to §0 directly.

### 5.1 Belfer, Geifman, Galun, Basri — *Spectral Analysis of the Neural Tangent Kernel for Deep Residual Networks* (JMLR 25, 2024, arXiv:2104.03093)

For fully connected residual networks $x_\ell = x_{\ell-1} + \alpha V_\ell \sigma(x_{\ell-1})$ with
ReLU, in the infinite-width limit:

- With inputs uniform on $S^{d-1}$, the eigenvalues of the residual NTK with respect to the spherical
  harmonics decay polynomially in frequency as $k^{-d}$ — **the same rate as the fully connected
  NTK**, so the RKHS and the frequency bias are unchanged by skip connections.
- The behaviour depends critically on $\alpha$. For $\alpha = L^{-\gamma}$ with $0.5 < \gamma \le 1$,
  deep residual NTK converges *uniformly* to a two-layer fully connected NTK, so it does **not**
  become spiky, and they prove generalization bounds for it; deep fully connected NTK does become
  spiky (kernel concentrating near the diagonal) and generalizes poorly. For $0 \le \gamma < 0.5$,
  residual NTK becomes spiky too.
- With no bias and $0.5 < \gamma \le 1$, deep residual NTK has a *parity imbalance*: odd frequencies
  $k \ge 3$ get much smaller eigenvalues than even ones, and their experiment (depth 50,
  $\alpha = 1/50$, regressing sinusoids on $S^1$) confirms odd frequencies took much longer to fit.
  Choosing $\gamma = 0.5$, i.e. $\alpha = 1/\sqrt{L}$, removes the imbalance.

**Use to us, including an inversion worth stating.** For *our* objective, "spiky" is not a defect —
a spiky kernel is a near-diagonal kernel, which is condition (A). We do not need the bonus field to
generalize to unseen positions; we need 100 fixed positions to decay together and, in the extension,
independently. So the depth-and-$\alpha$ regime this paper warns against ($\gamma < 0.5$, or plain
deep fully connected) is the regime we might deliberately want. Also note the parity-imbalance
warning: a bias-free residual network with $\alpha = 1/L$ creates a *new* source of eigenvalue spread.
If we use residual blocks, include biases and use $\alpha = 1/\sqrt{L}$.

### 5.2 Barzilai, Geifman, Galun, Basri — *A Kernel Perspective of Skip Connections in Convolutional Networks* (ICLR 2023, arXiv:2211.14810)

Derives the Gaussian-process and NTK kernels for residual convolutional networks and bounds their
condition numbers. Their stated results: "(1) with ReLU activation, the eigenvalues of these residual
kernels decay polynomially at a similar rate compared to the same kernels when skip connections are
not used, thus maintaining a similar frequency bias; (2) however, residual kernels are more locally
biased", and "the matrices obtained by these residual kernels yield favorable condition numbers at
finite depths than those obtained without the skip connections, enabling therefore faster convergence
of training with gradient descent."

**Use to us.** This is the closest thing in the literature to "skip connections make convergence more
uniform": a better condition number of the kernel *matrix* means a smaller spread of $\lambda_k$,
which by §0 means a smaller spread of per-position exponents under a $1/n$ schedule. But note the
condition is on the whole matrix and the improvement is at *finite depth*; the paper is about
convolutional networks; and the eigenvalue *decay rate* is explicitly unchanged. At depth 2 the
effect should be small. Do not expect residual blocks alone to solve our problem.

### 5.3 Cai, Zhu, Shen, Wang, Cao — *Towards the Spectral Bias Alleviation by Normalizations in Coordinate Networks* (arXiv:2407.17834; the batch-norm-only predecessor is *Batch Normalization Alleviates the Spectral Bias in Coordinate Networks*, CVPR 2024)

The most quantitatively on-point architecture paper for us, because its claimed effect is stated
directly in terms of the *variance* of the NTK eigenvalues.

- Setting: coordinate networks — plain MLPs mapping low-dimensional coordinates to signal values
  (e.g. $\mathbb{R}^2 \to \mathbb{R}^3$), architectures like $[5\times20]$ (5 layers, width 20). This
  is structurally the same object as our predictor: a small MLP on 2-D positions.
- Their abstract's claim: "normalization techniques greatly reduces the maximum and variance of NTK's
  eigenvalues while slightly modifies the mean value", and since the largest eigenvalue is far above
  the rest, this shifts the eigenvalue distribution upward and alleviates spectral bias.
  Their reported scalings: plain MLP has mean $O(N+\sqrt N)$, variance $O((N+2\sqrt N+C)TN)$, maximum
  $O(N\sqrt{CT}+\sqrt N CT)$; batch normalization keeps the mean, reduces the variance to $O(CTN)$
  and the maximum by about $\sqrt T$; layer normalization keeps the mean, reduces the variance by a
  factor $T/C$ and the maximum by about $\sqrt C$.
- Placement: "we add one corresponding normalization layer before the ReLU activation of each
  fully-connected layer."
- They also propose combining batch and layer normalization ("cross normalization"), which is their
  best variant across image compression (up to +4.89 dB PSNR over plain ReLU on Kodak), CT and MRI
  reconstruction, shape representation, NeRF and multi-view stereo.

**Use to us.** Reducing the *variance* of the eigenvalues of $K$ is, by §0, literally reducing the
spread of the per-position decay exponents. This is a cheap, drop-in change (a LayerNorm before each
activation) in exactly our kind of network, with a mechanism stated in the right currency. Caveat:
their derivations are order-of-magnitude scalings under their own assumptions and the paper's
validation is task performance, not eigenvalue histograms on our point set — so treat the claim as a
strong hypothesis to test by measuring $K$'s spectrum with and without the normalization, not as an
established result. Second caveat: batch normalization couples the 100 positions through batch
statistics, which changes the residual dynamics away from the clean $R \leftarrow (I-\eta K)R$ form;
LayerNorm (per-example) does not, so prefer LayerNorm for our purposes.

### 5.4 Saratchandran, Chng, Lucey — *Analyzing the Neural Tangent Kernel of Periodically Activated Coordinate Networks* (arXiv:2402.04783)

Derives bounds on the *minimum* eigenvalue of the NTK at finite width for networks with periodic
(sine/cosine) activations, needing only one wide layer growing at least linearly in the number of
samples, and concludes that "periodically activated networks are notably more well-behaved, from the
NTK perspective, than ReLU activated networks", with an application to memorization capacity. A
larger $\lambda_{\min}$ with a comparable $\lambda_{\max}$ is a smaller condition number, i.e. a
narrower spread of per-position exponents.

### 5.5 Sitzmann, Martel, Bergman, Lindell, Wetzstein — *Implicit Neural Representations with Periodic Activation Functions* (NeurIPS 2020, arXiv:2006.09661)

SIREN: sine activations throughout, with an initialization that preserves the activation distribution
across depth — weights $\sim U(-\sqrt{6/n}, \sqrt{6/n})$ so each sine's pre-activation is
approximately unit-variance normal — and a first layer $\sin(\omega_0 W x + b)$ with $\omega_0 = 30$
to place the representation at the right frequency scale. Practically, $\omega_0$ plays the same role
for us as Tancik's $\sigma$: it sets how fast the kernel decorrelates two nearby positions.

### 5.6 Hayou, Doucet, Rousseau — *On the Impact of the Activation Function on Deep Neural Networks Training* (ICML 2019, arXiv:1902.06853)

Initializing on the edge-of-chaos curve propagates information deeper and speeds up training, and
smooth activations (ELU, SiLU/GELU-like) allow deeper propagation than ReLU. This is a depth
argument; at 2 layers the only part that transfers is the weaker claim that the activation changes
the kernel, which we should test empirically rather than argue from this paper.

### 5.7 Ma, Belkin — *Diving into the shallows: a computational perspective on large-scale shallow learning* (NeurIPS 2017, arXiv:1703.10622)

Shows that with smooth kernels, whose eigenvalues decay fast, only a vanishingly small part of the
function space is reachable in a polynomial number of gradient-descent iterations, and proposes
EigenPro: a preconditioner built from a small number of approximately computed top eigenvectors that
damps the largest eigenvalues, letting a much larger step size be used and accelerating the slow
directions. Their preconditioner is partial (top-$q$ eigenvalues) because $n$ is large.

**Use to us.** Our $N$ is 100. There is no reason to settle for a partial preconditioner: we can form
and invert the full $100\times100$ kernel and flatten the spectrum *exactly* (method M2 below). This
paper is the right citation for why the spectrum, not the landscape, limits what gradient descent can
reach in a given number of steps.

---

## 6. Answers to the four questions, stated plainly

**Q1 — What does Li et al. 2018 show about skip connections, and by what mechanism?** It shows,
empirically and by 2-D filter-normalized visualization backed by Hessian eigenvalue-ratio maps, that
plain (skip-free) deep networks' loss surfaces transition from near-convex to visually chaotic as
depth grows past roughly 50 layers, and that skip connections keep the surface essentially
depth-independent (the 0.1-level contour of ResNet-20 and ResNet-110 is nearly the same). Filter
normalization exists because ReLU and batch normalization make the loss invariant to per-filter weight
rescaling, so unnormalized plots compare scale, not geometry. Width has the same qualitative effect as
skips. There is no per-example claim anywhere in the paper, and the mechanism offered is about
depth-induced pathology, not about equalizing anything across data points.

**Q2 — Does a smoother landscape give more uniform per-example convergence?** No paper shows this
directly. What has evidence is the narrower claim that the interventions people associate with
"smoother landscape" also improve the *kernel conditioning* and *decorrelate per-example gradients*,
which is the mechanism that would produce uniformity: width and (batch normalization + skips) together
lower gradient confusion, i.e. the off-diagonal of the gradient Gram matrix (Sankararaman et al.);
residual kernels have better condition numbers at finite depth (Barzilai et al.); normalization layers
reduce the maximum and the variance of the NTK eigenvalues in small coordinate MLPs (Cai et al.);
batch normalization drives representations toward mutual orthogonality with depth (Daneshmand et al.).
Meanwhile the strongest documented cause of *non*-uniform per-point convergence is not the landscape
at all but the input distribution: convergence at $x$ takes time proportional to $1/p(x)$ (Basri et
al.). Treat "smoothness helps uniformity" as an unproven bridge; treat "eigenvalue spread and
off-diagonal coupling control uniformity" as the thing to measure and act on.

**Q3 — Is the landscape a problem for 4 → 256 → 128, or is it the NTK spectrum?** It is the spectrum.
The landscape results that motivate the question are about 50–150 layer networks; both of Li et al.'s
own remedies (shallower, wider) describe our network already. Du et al. 2019 says a two-layer ReLU
network of sufficient width on non-parallel inputs converges to a global minimum at a linear rate on
the squared loss, so there is no local-minimum obstacle; Arora et al. 2019 gives the exact residual
formula showing the entire rate structure is the eigen-decomposition of the kernel. Caveats to hold
onto: the width in those theorems is far larger than 256 and our overparameterization is only about
$2.7\times$, so kernel drift during training must be measured rather than assumed; and one concrete
non-spectral pathology *is* present, namely ReLU's positive homogeneity making
$K_{ii} \propto \lVert x_i \rVert^2$ for raw, unnormalized $[x,y,0,0]$ inputs (§3.4).

**Q4 — Which architecture recipes are worth testing, and what is each reported to do?** In rough order
of expected value:
1. **Input normalization / coordinate embedding onto a common norm** — removes a per-position rate
   factor of $\lVert x_i\rVert^2$ that has nothing to do with learning (Cho & Saul's arc-cosine
   kernel). Cheapest, most certain.
2. **Random Fourier feature embedding with tunable scale $\sigma$** — makes the effective kernel
   stationary (equal diagonal at every position) with a bandwidth we control, so $K$ can be pushed
   toward $\kappa I$ (Tancik et al.; Wang, Wang & Perdikaris for the multi-scale version).
3. **LayerNorm before each activation** — reported to reduce the maximum and, more importantly for
   us, the *variance* of the NTK eigenvalues in exactly this class of small coordinate MLP (Cai et
   al.). Prefer LayerNorm over BatchNorm because BatchNorm couples the 100 positions through batch
   statistics.
4. **Sine activations (SIREN) with $\omega_0$ swept** — larger minimum NTK eigenvalue than ReLU at
   finite width (Saratchandran et al.); $\omega_0$ is a decorrelation-scale knob like $\sigma$
   (Sitzmann et al.).
5. **Width** — lower gradient confusion, closer to the lazy regime (Sankararaman et al.; Lee et al.).
   Sweep 256 → 1024 and check whether the *spread* of per-position slopes narrows.
6. **Residual MLP blocks** — better kernel-matrix condition number at finite depth without changing
   the eigenvalue decay rate (Barzilai et al.); at depth 2 the effect should be small, and if used,
   include biases and scale the residual branch by $1/\sqrt{L}$ to avoid the parity imbalance Belfer
   et al. prove for bias-free $\alpha = 1/L$ residual networks.
7. **GELU vs ReLU** — the smooth-activation literature (Hayou et al.) is a depth argument; at depth 2
   expect little, but GELU does remove the exact positive homogeneity, so it is a cheap control.

---

## 7. Concrete methods to implement in this setting

Ordered by how directly they attack §0's requirement. M0 is a measurement, not a method, and should
come first because it decides between the rest.

### M0 — Measure the kernel before changing anything (half a day, decides everything else)

With $N \approx 100$ the empirical kernel is a $100\times100$ matrix and is fully computable.

- Form $J \in \mathbb{R}^{(N\cdot128) \times P}$, the Jacobian of the 128 outputs at the 100 points
  (use `torch.func.jacrev`/`vmap`, or 128 backward passes per point). Then $K = J J^\top$ collapsed
  over output heads: $K_{ij} = \dfrac{1}{128}\sum_{o} \langle \nabla_\theta g_o(x_i), \nabla_\theta g_o(x_j)\rangle$.
- Report: the eigenvalue spectrum $\lambda_1 \ge \dots \ge \lambda_N$ and its spread
  $\lambda_{\max}/\lambda_{\min}$; the diagonal $K_{ii}$ against $\lVert x_i\rVert$ (this is the
  §3.4 check); the histogram of normalized off-diagonals $K_{ij}/\sqrt{K_{ii}K_{jj}}$ (the gradient
  confusion diagnostic of Sankararaman et al.); the projections $\lVert v_k^\top R(0)\rVert$ (the
  Arora et al. diagnostic); and the eigenvector localization $\max_i |v_{k,i}|$ per mode.
- Repeat at the end of training and compare — that is the kernel-drift check that decides whether the
  lazy analysis in §0 is even valid at width 256.
- **Predicted result if §0 is right:** the per-position log-log slope of $b_i$ over a window should be
  close to $c \cdot (\text{effective } \lambda \text{ at } i)$, and plotting measured slope against
  $\sum_k \lambda_k v_{k,i}^2 / \sum_k v_{k,i}^2$ should give a straight line. If it does, the rest of
  this section applies. If it does not, the network is not lazy and the analysis needs redoing.

### M1 — Quartic loss in the residual norm: the power law per position, with a constant step size

*Derived from §0's ODE; not taken from any paper below. Flag it as untested.*

For a loss of the form $L = \sum_i \psi(b_i)$, the residual at position $i$ obeys, when $K$ is
diagonal, $\mathrm{d}b_i/\mathrm{d}n = -\eta K_{ii}\,\psi'(b_i)$. Asking for $b(n) = A n^{-\alpha}$
forces $\psi'(b) \propto b^{1+1/\alpha}$, i.e. $\psi(b) \propto b^{2+1/\alpha}$.

- For $\alpha = 1/2$: $\psi(b) = b^4$, i.e. **use $L = \dfrac14\sum_i \lVert g(x_i)-f(x_i)\rVert_2^4$**
  with a constant step size. Then
  $b_i(n)^2 = \big(b_i(0)^{-2} + 2\eta K_{ii} n\big)^{-1}$, so $b_i(n) \to (2\eta K_{ii} n)^{-1/2}$
  for every $i$ — the exponent is $-1/2$ at every position by construction, and the only per-position
  difference is the prefactor $(2\eta K_{ii})^{-1/2}$, which M3/M5 equalize.
- Contrast with ordinary MSE ($\psi = b^2$), which gives exponential decay of $b_i$ at constant step
  size and needs a global $1/n$ schedule to become a power law — with the per-mode exponent
  $c\lambda_k$ that is the whole problem.
- **Why this also addresses the harder extension**: the update to $b_i$ happens only on steps where
  $i$ is sampled, so if $K$ is close to diagonal, $b_i \approx (2\eta K_{ii} n_i)^{-1/2}$ — decay in
  position $i$'s *own* visit count $n_i$, with no schedule and no counters. That is exactly the
  count-based bonus. It requires the near-diagonal kernel of M5/M6.
- Practical notes: the gradient is $\lVert r_i\rVert^2 r_i$, so it vanishes fast as $r_i$ shrinks —
  use float64 or rescale, and expect the effective step size to need retuning by orders of magnitude
  versus MSE. Also, with Adam the per-parameter normalization destroys the $\lVert r_i\rVert^2$
  factor that produces the power law, so this method requires plain SGD (or at most momentum), not
  Adam.
- Generalization: to target exponent $\alpha$, use $\psi(b)=b^{2+1/\alpha}$; $\alpha=1$ gives a cubic
  loss. This gives a single knob for "which count-based bonus rate do we want".

### M2 — Function-space preconditioning (Gauss-Newton / kernel whitening): uniform by construction

Instead of $\theta \leftarrow \theta - \eta J^\top r$, use
$\theta \leftarrow \theta - \eta J^\top (K + \delta I)^{-1} r$. The function-space update becomes
$r \leftarrow (I - \eta K(K+\delta I)^{-1}) r \approx (1-\eta) r$: every eigen-mode contracts at the
*same* rate, so every position's residual contracts at the same rate whatever $K$'s spectrum is.
Combined with $\eta_n = 1/(2n)$, every position gets $b_i(n) \propto n^{-1/2}$ exactly.

- Cost: forming $K$ costs $N$ (or $N \times 128$) Jacobian-vector products per update and a
  $100\times100$ solve — trivial at our size, and $K$ can be recomputed every $T$ steps rather than
  every step if it drifts slowly (M0 tells us).
- This is the exact, full-spectrum version of what EigenPro (Ma & Belkin) does approximately with the
  top-$q$ eigenvectors because their $n$ is large; and it is the per-position version of the
  NTK-based loss reweighting Wang, Yu & Perdikaris use across PINN loss terms.
- Damping $\delta$ is the only real hyperparameter, and it controls how much of the small-eigenvalue
  end gets amplified (and hence how much noise gets amplified).
- **Expected result:** the cleanest possible per-position uniformity. Use it as the *ceiling* — the
  best any method could do — even if it is too heavy for the final recipe.

### M3 — Per-position loss weights $w_i \propto 1/K_{ii}$ (the cheap approximation of M2)

Weight position $i$'s contribution to the loss by $1/K_{ii}$, recomputed every few hundred steps.
This equalizes the *diagonal* of the effective kernel without touching the off-diagonals, so it fixes
the largest and simplest source of per-position rate difference (including the $\lVert x_i\rVert^2$
effect of §3.4) at almost no cost. This is a direct transposition of the NTK-based weight balancing in
Wang, Yu & Perdikaris (2021) from "loss terms" to "positions". Combine with M1: M3 fixes the
prefactor, M1 fixes the exponent.

### M4 — Train only the readout on whitened random features

Freeze the first layer, train only the $256 \to 128$ readout. The dynamics are then exactly linear
least squares with $K = \Phi\Phi^\top$ where $\Phi \in \mathbb{R}^{100\times256}$ is the fixed hidden
representation. Now $K$ can be controlled exactly:

- Whiten $\Phi$ across the 100 positions (ZCA on $\Phi$, or an orthogonal $\Phi$ from a QR of the
  random features, which is possible since $256 > 100$). Then $K = I$ exactly, and every position
  decays at the identical rate for free.
- This is the finite, exact form of what Daneshmand et al. prove batch normalization approaches
  asymptotically in depth.
- Cost: the bonus field is then a linear function of fixed features, which limits how well it can
  track a target network — but for the *decay-rate* research question this is the setting where the
  answer is provable, and it is the right control experiment.

### M5 — Fourier feature embedding with swept bandwidth (best architectural change)

Replace the input $[x, y, 0, 0]$ with $\gamma(x) = [\cos(2\pi B x), \sin(2\pi B x)]$,
$B \in \mathbb{R}^{m\times 2}$ with entries $\sim \mathcal{N}(0, \sigma^2)$, and feed $\gamma(x)$ to
the same MLP (Tancik et al.). Two effects we need at once:

- Every position gets the same $\lVert \gamma(x_i)\rVert$ and the same $K_{ii}$ (the kernel becomes
  stationary), so requirement (1), "starts at the same value", and the equal-diagonal condition are
  both met by construction.
- $\sigma$ tunes the bandwidth: as $\sigma$ grows, $K_{ij} \to 0$ for $i \ne j$ and $K \to \kappa I$,
  which is condition (A) and also the near-diagonality M1's extension needs.
- Sweep $\sigma$ over several orders of magnitude and, at each, record: the spread of per-position
  log-log slopes; $\lambda_{\max}/\lambda_{\min}$ of $K$; and how smooth the resulting bonus field is
  between grid positions. There is a real tradeoff here (uniformity versus spatial generalization of
  the bonus) and it should be reported as a curve, not collapsed to one setting.
- Multi-scale variant (Wang, Wang & Perdikaris): concatenate features from two or three $\sigma$
  values, useful if the 100 positions are unevenly spaced so one bandwidth cannot serve all of them.

### M6 — Sparse / localized hidden representation

Fuzzy tiling activations (Pan, Banman & White) or any localized encoding (radial basis units, tile
coding) make the hidden representation sparse and localized, so $K_{ij} \approx 0$ for distant
positions almost by construction. This is the interference-reduction route to the same
near-diagonal kernel as M5, and it is the one the RL literature already uses. Expect it to look
tabular in the limit — which for a count-based bonus is a feature, not a bug, but it removes any
generalization between positions.

### M7 — LayerNorm before each activation

Add a LayerNorm before every ReLU (the placement Cai et al. use). Reported effect: reduced maximum
and reduced *variance* of the NTK eigenvalues, with the mean roughly unchanged. Prefer LayerNorm over
BatchNorm here: BatchNorm ties the 100 positions together through batch statistics, which invalidates
the clean per-position dynamics of §0, whereas LayerNorm acts per example. Measure the eigenvalue
histogram of $K$ with and without, via M0 — this is a claim we can verify directly on our own point
set in an afternoon.

### M8 — Per-position step size driven by that position's visit count (the harder extension)

Once $K$ is near diagonal (M5/M6), give position $i$ its own counter $n_i$ and its own effective step
$\eta_i(n) = c/(K_{ii}\, n_i)$ — implemented as a per-position loss weight $c/(K_{ii} n_i)$ divided by
the global step size. Then position $i$'s residual satisfies $b_i \propto n_i^{-c}$, and $c = 1/2$
gives the count-based bonus rate against $i$'s own visit count, exactly as required. Note this is the
*only* way to get the exponent right under non-uniform visitation with an MSE loss — Basri et al.'s
result says the sampling density changes the rate *constant*, not the exponent, so a global schedule
cannot do it. M1 (quartic loss) achieves the same thing without counters and is the more elegant
route if it works; M8 is the reliable fallback.

### M9 — Residual MLP block, if depth is increased

Only worth trying if depth goes past 2–3 layers. Use $x_\ell = x_{\ell-1} + \alpha V_\ell \sigma(x_{\ell-1})$
**with biases** and $\alpha = 1/\sqrt{L}$: Belfer et al. prove that a bias-free residual network with
$\alpha = L^{-\gamma}$, $0.5 < \gamma \le 1$, has a parity imbalance in which odd frequencies get much
smaller eigenvalues and take much longer to fit — a *new* source of non-uniformity that we would be
importing. Barzilai et al. is the reason to expect a better condition number at finite depth.

---

## 8. What none of these papers answers

- No paper measures the distribution of per-training-point decay *exponents* under a decaying step
  size, or how architecture changes that distribution. Every uniformity claim above is mediated by a
  proxy (eigenvalue spread, condition number, gradient-cosine histogram). The measurement in M0 does
  not appear to exist in the literature for any setting, which means it is worth doing and worth
  reporting.
- No paper studies distillation to a *random* target with a $1/n$ schedule and asks for a prescribed
  power law. The closest analogue is the PINN loss-balancing line (Wang, Yu & Perdikaris), which
  equalizes across a handful of loss terms rather than across many individual points.
- The non-uniform-visitation case has one relevant result (Basri et al.) and it is about the rate
  constant, not the exponent. The per-position-counter idea (M8) and the quartic-loss idea (M1) are
  ours to test, not something to cite.
- Nothing in this literature addresses the "starts at exactly 1 everywhere" requirement; that is an
  initialization-and-normalization matter (equal input norms, or dividing by $\lVert r_i(0)\rVert$ at
  readout), not an optimization one.

---

## 9. Suggested order of experiments

1. **M0** on the current setup: kernel spectrum, diagonal against $\lVert x_i\rVert$, off-diagonal
   histogram, eigenvector localization, kernel drift. Confirm or refute that the per-position slope is
   predicted by the kernel. Everything else is conditional on this.
2. **Input normalization** (§3.4) alone, then re-run M0. This may remove a large fraction of the
   observed heterogeneity for free.
3. **M2** (full Gauss-Newton preconditioning) as the ceiling experiment: if it does not give uniform
   $n^{-1/2}$, the lazy picture is wrong and the analysis must be redone before trying anything else.
4. **M1** (quartic loss, constant step size) — the cheapest route to a per-position power law and the
   one that also solves the non-uniform-visitation case. Test with plain SGD, not Adam.
5. **M5** (Fourier features, $\sigma$ sweep) and **M7** (LayerNorm) as the two architectural changes
   with the most direct published support for narrowing the eigenvalue spread; report the spread of
   per-position slopes as the outcome metric, not the aggregate slope.
6. **M8** on top of the winner, for the non-uniform visitation extension.
7. **M4** (whitened frozen features) as the provable control that the target rate is attainable at
   all in this setting.

---

## 10. Reference list (all verified to exist; identifier given where found)

| # | Paper | Venue / identifier |
|---|---|---|
| 1 | Li, Xu, Taylor, Studer, Goldstein. Visualizing the Loss Landscape of Neural Nets | NeurIPS 2018; arXiv:1712.09913 |
| 2 | Böttcher, Wheeler. Visualizing high-dimensional loss landscapes with Hessian directions | J. Stat. Mech. 2024; arXiv:2208.13219 |
| 3 | Santurkar, Tsipras, Ilyas, Madry. How Does Batch Normalization Help Optimization? | NeurIPS 2018; arXiv:1805.11604 |
| 4 | Schoenholz, Gilmer, Ganguli, Sohl-Dickstein. Deep Information Propagation | ICLR 2017; arXiv:1611.01232 |
| 5 | Sankararaman, De, Xu, Huang, Goldstein. The Impact of Neural Network Overparameterization on Gradient Confusion and SGD | ICML 2020; arXiv:1904.06963 |
| 6 | Daneshmand, Joudaki, Bach. Batch Normalization Orthogonalizes Representations in Deep Random Networks | NeurIPS 2021; arXiv:2106.03970 |
| 7 | Bengio, Pineau, Precup. Interference and Generalization in Temporal Difference Learning | ICML 2020; arXiv:2003.06350 |
| 8 | Liu, Wang, Tao, Javed, White, White. Measuring and Mitigating Interference in Reinforcement Learning | CoLLAs 2023 (PMLR v232); arXiv:2307.04887 |
| 9 | Pan, Banman, White. Fuzzy Tiling Activations | ICLR 2021; arXiv:1911.08068 |
| 10 | Baldock, Maennel, Neyshabur. Deep Learning Through the Lens of Example Difficulty | NeurIPS 2021; arXiv:2106.09647 |
| 11 | Jacot, Gabriel, Hongler. Neural Tangent Kernel | NeurIPS 2018; arXiv:1806.07572 |
| 12 | Lee, Xiao, Schoenholz, Bahri, Novak, Sohl-Dickstein, Pennington. Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent | NeurIPS 2019; arXiv:1902.06720 |
| 13 | Du, Zhai, Póczos, Singh. Gradient Descent Provably Optimizes Over-parameterized Neural Networks | ICLR 2019; arXiv:1810.02054 |
| 14 | Arora, Du, Hu, Li, Wang. Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks | ICML 2019; arXiv:1901.08584 |
| 15 | Cho, Saul. Kernel Methods for Deep Learning | NeurIPS 2009 |
| 16 | Moulines, Bach. Non-Asymptotic Analysis of Stochastic Approximation Algorithms for Machine Learning | NeurIPS 2011 |
| 17 | Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio, Courville. On the Spectral Bias of Neural Networks | ICML 2019; arXiv:1806.08734 |
| 18 | Basri, Galun, Geifman, Jacobs, Kasten, Kritchman. Frequency Bias in Neural Networks for Input of Non-Uniform Density | ICML 2020; arXiv:2003.04560 |
| 19 | Tancik et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains | NeurIPS 2020; arXiv:2006.10739 |
| 20 | Wang, Wang, Perdikaris. On the eigenvector bias of Fourier feature networks | CMAME 384:113938 (2021); arXiv:2012.10047 |
| 21 | Wang, Yu, Perdikaris. When and why PINNs fail to train: a neural tangent kernel perspective | J. Comput. Phys. 449:110768 (2021); arXiv:2007.14527 |
| 22 | Belfer, Geifman, Galun, Basri. Spectral Analysis of the Neural Tangent Kernel for Deep Residual Networks | JMLR 25 (2024); arXiv:2104.03093 |
| 23 | Barzilai, Geifman, Galun, Basri. A Kernel Perspective of Skip Connections in Convolutional Networks | ICLR 2023; arXiv:2211.14810 |
| 24 | Cai, Zhu, Shen, Wang, Cao. Towards the Spectral bias Alleviation by Normalizations in Coordinate Networks | arXiv:2407.17834 (predecessor: Batch Normalization Alleviates the Spectral Bias in Coordinate Networks, CVPR 2024) |
| 25 | Saratchandran, Chng, Lucey. Analyzing the Neural Tangent Kernel of Periodically Activated Coordinate Networks | arXiv:2402.04783 |
| 26 | Sitzmann, Martel, Bergman, Lindell, Wetzstein. Implicit Neural Representations with Periodic Activation Functions | NeurIPS 2020; arXiv:2006.09661 |
| 27 | Hayou, Doucet, Rousseau. On the Impact of the Activation Function on Deep Neural Networks Training | ICML 2019; arXiv:1902.06853 |
| 28 | Ma, Belkin. Diving into the shallows: a computational perspective on large-scale shallow learning (EigenPro) | NeurIPS 2017; arXiv:1703.10622 |

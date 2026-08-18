# Why prediction error decays at different rates at different input points, and what makes the rates equal

Notes for the RND decay-rate problem: a predictor $g$ (4 inputs, one hidden layer of 256, output
128) trained by full-batch mean-squared-error distillation against a frozen random target $f$ on a
fixed set of about 100 maze positions, where the wanted behaviour is that the per-position residual
norm starts at the same value at every position and then falls as (number of steps) to the power
$-1/2$ at every position.

---

## 1. Setting and notation

Define the following, all used throughout these notes.

- $N$ is the number of fixed training positions (about 100 here). $x_1, \dots, x_N$ are those
  positions, each fed to the network as a 4-vector.
- $d_{\text{out}}$ is the output width of target and predictor (128 here).
- $E(n) \in \mathbb{R}^{N \times d_{\text{out}}}$ is the residual matrix after $n$ full-batch
  optimizer steps; its row $i$ is $g_n(x_i) - f(x_i)$.
- $b_i(n) = \lVert g_n(x_i) - f(x_i) \rVert_2$ is the bonus at position $i$, i.e. the 2-norm of row
  $i$ of $E(n)$. This is the quantity the project wants to behave as $n^{-1/2}$.
- $K \in \mathbb{R}^{N \times N}$ is the neural tangent kernel Gram matrix on the training
  positions, $K_{ij} = \langle \partial g(x_i)/\partial\theta, \partial g(x_j)/\partial\theta
  \rangle$ evaluated at the initial parameters (per output channel).
- $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_N \ge 0$ are the eigenvalues of $K$ and
  $v_1, \dots, v_N$ its orthonormal eigenvectors; $v_{j,i}$ is entry $i$ of $v_j$.
- $\eta$ is the step size; $\eta_n$ when it changes with the step index.
- $n_i$ is the number of times position $i$ has been visited (used only in Section 7, the
  non-uniform-visitation extension); $p_i$ is its sampling probability.

In the linearized (kernel) regime the residual obeys

$$E(n+1) = (I - \eta K)\thinspace E(n), \qquad\text{so}\qquad E(n) = (I - \eta K)^n E(0).$$

Every rate quoted below is tagged with the quantity it describes, because the literature mixes at
least four: the squared loss summed over training points, the 2-norm of the whole residual vector,
the coefficient of one eigen-direction, and the function error at one input point. They have
different exponents (the first is the square of the second, up to a constant).

---

## 2. Question 1 — why $E(n) = (I - \eta K)^n E(0)$ makes the decay rate position-dependent

### 2.1 The exact per-position formula

Expand $E(0)$ in the eigenbasis of $K$. Write $c_j = E(0)^\top v_j \in \mathbb{R}^{d_{\text{out}}}$
for the coefficient of mode $j$. Then

$$E(n) = \sum_{j=1}^{N} (1 - \eta\lambda_j)^n \thinspace v_j \thinspace c_j^\top ,
\qquad
g_n(x_i) - f(x_i) = \sum_{j=1}^{N} (1 - \eta\lambda_j)^n \thinspace v_{j,i} \thinspace c_j .$$

For a random-network-distillation residual at initialization the entries of $E(0)$ are close to
independent and identically distributed with some variance $\sigma^2$, so the $c_j$ are close to
mutually orthogonal with $\lVert c_j \rVert^2 \approx d_{\text{out}}\sigma^2$ for every $j$. Under
that approximation the cross terms drop and

$$\boxed{\thickspace b_i(n)^2 \thickspace \approx\thickspace d_{\text{out}}\sigma^2 \sum_{j=1}^{N} w_{ij}\thinspace (1-\eta\lambda_j)^{2n},
\qquad w_{ij} \thickspace =\thickspace v_{j,i}^2 . \thickspace }$$

Because $V$ is orthogonal, $\sum_j w_{ij} = 1$ for every $i$ and $\sum_i w_{ij} = 1$ for every $j$.
So:

1. **Every position has its own probability distribution $w_{i\cdot}$ over the eigenvalues of
   $K$.** The residual at position $i$ is a mixture of $N$ geometric sequences whose ratios are
   $(1-\eta\lambda_j)^2$, mixed with weights $w_{ij}$ that depend on where the eigenvectors put
   their mass.
2. **The aggregate residual uses the uniform mixture.** Averaging the boxed formula over $i$ and
   using $\sum_i w_{ij} = 1$ gives
   $\frac{1}{N}\sum_i b_i(n)^2 \approx \frac{d_{\text{out}}\sigma^2}{N}\sum_j (1-\eta\lambda_j)^{2n}$,
   i.e. all modes weighted equally. This is why the aggregate can look clean while individual
   positions do not: they are averages over *different* distributions.
3. **Position-dependence has exactly two sources.** Heterogeneity requires both a spread of
   eigenvalues $\lambda_j$ (otherwise every mixture collapses to the same single geometric
   sequence) and eigenvectors whose mass is not spread evenly over positions (otherwise
   $w_{ij} = 1/N$ for all $i,j$ and every position has the same mixture). Removing either one makes
   the per-position curves identical. This separation matters for choosing an intervention and is
   made explicit in Section 8.
4. **At $n = 0$ all positions start equal**, $b_i(0)^2 \approx d_{\text{out}}\sigma^2$, because
   $\sum_j w_{ij} = 1$. The "starts at the same value" requirement is therefore satisfied to within
   a $O(1/\sqrt{d_{\text{out}}})$ fluctuation already, and can be made exact by the trick in
   Section 9.1.

### 2.2 What a single position's curve looks like when $K$ has a wide spectrum

Take logarithms of the boxed formula. The local slope of $\log b_i$ against $\log n$ is

$$\frac{d \log b_i(n)}{d \log n} \thickspace =\thickspace -\thinspace n \cdot \frac{\sum_j w_{ij}\thinspace \eta\lambda_j\thinspace
e^{-2\eta\lambda_j n}}{\sum_j w_{ij}\thinspace e^{-2\eta\lambda_j n}}
\thickspace =\thickspace -\thinspace n\thinspace \eta\thinspace \langle \lambda \rangle_{i,n},$$

where $\langle \lambda \rangle_{i,n}$ is the mean of $\lambda$ under the distribution $w_{i\cdot}$
re-weighted ("tilted") by $e^{-2\eta\lambda n}$ (using $\log(1-\eta\lambda)\approx-\eta\lambda$ for
small steps). Three consequences.

- **A constant step size gives no power law at a single position.** As $n$ grows the tilt suppresses the
  large eigenvalues and $\langle\lambda\rangle_{i,n}$ drifts down towards the smallest $\lambda_j$
  carrying weight at position $i$. The curve is a sum of exponentials: it looks like a straight line
  in log-log over a window and then bends. Over many decades this is exactly the "saturating"
  behaviour reported for the constant-step run.
- **The apparent slope over a window is set by the eigenvalues that are "active" in that window.**
  Position $i$ is in a different part of the spectrum than position $l$ whenever
  $w_{i\cdot} \ne w_{l\cdot}$, so at a fixed $n$ the two have different tilted means and different
  slopes. That is the observed heterogeneity, stated as a formula.
- **A single position whose weight sits on one eigenvalue decays as a clean exponential**; a
  position whose weight is spread over a wide range of $\lambda$ decays as a curve whose log-log
  slope steepens then flattens. Positions near the boundary of the maze, or isolated ones, tend to
  carry weight on the small eigenvalues (their eigenvector mass is localized), and are the last to
  decay.

### 2.3 Why a $1/n$ step-size schedule turns exponentials into power laws — and why the exponents still differ per mode

With $\eta_k = a/k$ the factor for mode $j$ becomes

$$\prod_{k=k_0}^{n} \Bigl(1 - \frac{a\lambda_j}{k}\Bigr) \thickspace \sim\thickspace C_j \thinspace n^{-a\lambda_j},$$

which is the standard Robbins–Monro product. So the residual becomes

$$b_i(n)^2 \thickspace \approx\thickspace d_{\text{out}}\sigma^2 \sum_j w_{ij}\thinspace \tilde C_j \thinspace n^{-2a\lambda_j},$$

**a mixture of power laws whose exponents are $-a\lambda_j$, one per eigenvalue.** This is the
precise reason the $1/t$ schedule produced an aggregate slope near $-0.5$ while positions still
disagreed:

- the aggregate is $\frac{1}{N}\sum_j \tilde C_j n^{-2a\lambda_j}$, a fixed mixture, and if the
  bulk of the spectrum sits near a value $\bar\lambda$ with $a\bar\lambda \approx 1/2$ the aggregate
  traces a near-straight line of slope $-0.5$;
- position $i$'s mixture emphasises different $\lambda_j$, so its local slope is
  $-a\langle\lambda\rangle_{i,n}$ with the tilt now $n^{-2a\lambda}$, which differs across $i$;
- asymptotically **every** position converges to the same slope $-a\lambda_{\min}$ over the modes
  it touches, but that limit is reached only after the faster modes have died, which for a spectrum
  spanning orders of magnitude is far past any realistic training horizon.

**A uniform slope of $-1/2$ at every position and every $n$ requires
$a\langle\lambda\rangle_{i,n} = 1/2$ for all $i$ and $n$.** The only way to get that identically is
$\lambda_j = \lambda$ for all $j$ — a flat spectrum, $K = \lambda I$ — with $a = 1/(2\lambda)$.
Every intervention in Sections 5–9 is an approximation to that condition, or a way to fake it.

### 2.4 Why Adam decays too fast and then saturates

Adam divides each parameter's step by a running root-mean-square of its own gradient, so the
effective step per parameter is roughly constant in magnitude regardless of how small the gradient
has become. In the eigen-picture this acts like a step size that grows as the residual shrinks:
small-eigenvalue modes are pushed at nearly the same rate as large ones early on (a steeper
aggregate slope than $-1/2$, consistent with the reported $-0.8$), and then the dynamics hits the
floor set by the gradient-noise/epsilon term and the residual stops moving (the saturation). Adam
is not a fixed linear operator on the residual, so the boxed formula does not apply to it at all;
none of the papers below analyse Adam in this setting. Treat the Adam observation as an empirical
fact, not as something the kernel theory explains.

---

## 3. Question 2 — per-paper notes: what is known about the neural-tangent-kernel spectrum of small ReLU networks on low-dimensional inputs

### 3.1 Jacot, Gabriel, Hongler (2018), "Neural Tangent Kernel: Convergence and Generalization in Neural Networks", NeurIPS 2018, arXiv:1806.07572

- **What it shows.** For a fully connected network in the standard parameterization, as the widths
  go to infinity the kernel $K$ defined above converges to a deterministic limit that depends only
  on the architecture and the activation, and stays constant during gradient-descent training. In
  that limit, training the network on squared loss is equivalent to kernel gradient descent in
  function space with that kernel.
- **The claim relevant to us.** The abstract states that convergence is fastest along the largest
  kernel principal components of the input data. That is the sentence the whole spectral-bias
  literature elaborates: the eigen-decomposition of $K$, not the network, determines what is
  learned first.
- **Caveat for our setting.** Our predictor has one hidden layer of 256 units and 100 training
  points. That is only mildly wide. The linearization is a model, not a guarantee, and should be
  checked empirically (Section 10.1) before any conclusion is drawn from it.

### 3.2 Arora, Du, Hu, Li, Wang (2019), "Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks", ICML 2019, arXiv:1901.08584

- **The exact theorem (Theorem 4.1, quoted).** With $\lambda_0 = \lambda_{\min}(H^\infty) > 0$,
  initialization scale $\kappa = O(\epsilon\delta/\sqrt{n})$, width
  $m = \Omega\negthinspace \left(\dfrac{n^7}{\lambda_0^4\kappa^2\delta^4\epsilon^2}\right)$ and step size
  $\eta = O(\lambda_0/n^2)$, with probability at least $1-\delta$ over the initialization, for all
  $k = 0,1,2,\dots$ the residual satisfies
  $\lVert y - u(k)\rVert_2 = \sqrt{\sum_{i=1}^{n} (1-\eta\lambda_i)^{2k} (v_i^\top y)^2} \pm \epsilon$.
  Here $u(k)$ is the vector of network outputs on the training points at step $k$, $y$ the labels,
  and $\lambda_i, v_i$ the eigen-system of the limiting Gram matrix $H^\infty$.
- **Which quantity.** The **2-norm of the residual over the whole training set** — an aggregate, not
  a per-point quantity. Our per-position formula in Section 2.1 is the row-wise version of the same
  algebra (their proof sketch derives $\tilde u(k) - y = -\sum_i (1-\eta\lambda_i)^k (v_i^\top y) v_i$,
  which is exactly what we take rows of).
- **Their reading of it.** Labels aligned with top eigenvectors converge quickly; labels with
  uniform projections onto eigenvectors, or aligned with small-eigenvalue eigenvectors, converge
  slowly. They verify it on MNIST and CIFAR: real labels concentrate on top eigenvectors, random
  labels spread uniformly, and hand-built worst-case labels (the eigenvector of $\lambda_{\min}$)
  train visibly slower.
- **Why this matters here.** A random-network-distillation target is closest to their "random
  labels" case: it has no alignment with the top eigenvectors, so its projections are spread across
  the whole spectrum, which is precisely the case with the widest spread of per-mode rates.
- **Assumptions to keep in mind.** Two-layer ReLU, only the first layer trained, extreme width
  requirement, a step size bound $\eta = O(\lambda_0/n^2)$ that is far smaller than anything used in
  practice, and squared loss.

### 3.3 Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio, Courville (2019), "On the Spectral Bias of Neural Networks", ICML 2019, arXiv:1806.08734

- **What it shows.** Using Fourier analysis of deep ReLU networks (which are piecewise linear, so
  their Fourier transform has an explicit form), the magnitude of the Fourier components of the
  represented function decays with frequency; the networks "cannot have local fluctuations without
  affecting their global behavior". Empirically, when fitting a sum of sinusoids, low frequencies
  are fitted first and high frequencies later.
- **The second, less-quoted claim.** Learning high frequencies gets *easier* as the geometry of the
  data manifold gets more complex — the bias is a property of the network composed with the data
  embedding, not of the network alone.
- **Relevance and limits.** This is the qualitative statement of the phenomenon and predates the
  kernel-eigenvalue account. It gives no rate for a residual at a point and no eigenvalue formula;
  do not cite it for a numeric decay exponent.

### 3.4 Basri, Jacobs, Kasten, Kritchman (2019), "The Convergence Rate of Neural Networks for Learned Functions of Different Frequencies", NeurIPS 2019, arXiv:1906.00425

This is the paper that gives closed-form eigenvalues for a small ReLU network on a low-dimensional
input, which is the closest published setting to ours.

- **Setting.** Two-layer ReLU network, first layer trained, inputs on the unit circle $S^1$ or on
  $S^d$, uniformly spaced. Because the kernel is then a convolution, its eigenfunctions are the
  Fourier series (circle) or the spherical harmonics (sphere).
- **Eigenvalues without bias**, for frequency $k$ on the circle:
  $a_k = 1/\pi^2$ for $k=0$; $1/4$ for $k=1$; $\dfrac{2(k^2+1)}{\pi^2 (k^2-1)^2}$ for even
  $k \ge 2$; and **exactly $0$ for odd $k \ge 3$**. A bias-free two-layer ReLU network cannot
  represent or learn odd frequencies at or above 3 — an exact null space, not a slow direction.
- **Eigenvalues with bias**, same setting:
  $\dfrac{1}{2\pi^2}+\dfrac18$ for $k=0$; $\dfrac{1}{\pi^2}+\dfrac18$ for $k=1$;
  $\dfrac{k^2+1}{\pi^2(k^2-1)^2}$ for even $k\ge2$; $\dfrac{1}{\pi^2 k^2}$ for odd $k\ge2$. All
  frequencies now pass, and asymptotically $\lambda_k \sim \dfrac{1}{\pi^2 k^2}$.
- **The convergence-time prediction.** Combining with Arora's Theorem 4.1: to bring the coefficient
  of eigen-direction $i$ down to $\bar\delta$ needs $t_i > -\log(\bar\delta+\epsilon)/(\eta\bar\lambda_i)$,
  so on the circle the time grows as $k^2$. **Which quantity: the number of gradient steps to reduce
  the error of a pure single-frequency target to 5% of its initial value.** Measured exponents in
  their experiments: $O(k^{2.15})$ shallow no bias, $O(k^{1.93})$ shallow with bias, $O(k^{1.94})$
  a 5-hidden-layer network, $O(k^{2.11})$ a 10-layer residual network — that is, **depth and skip
  connections did not change the exponent.**
- **Higher dimension.** On $S^d$ the eigenvalues are obtained via the Funk–Hecke theorem and decay
  roughly as $k^{-d}$; measured convergence-time exponents on $S^2$ were $O(k^{2.74})$,
  $O(k^{2.87})$, $O(k^{3.13})$ — consistent with $k^{d}$ with $d=3$ counting the ambient dimension
  of the sphere in their convention. **The spread of eigenvalues, and hence of per-mode rates, gets
  exponentially worse with input dimension.**
- **Direct consequence for us.** Our input is effectively 2-dimensional (a maze position padded to
  4 dimensions). The condition number of $K$ over 100 points therefore spans roughly
  $k_{\max}^2$ to $k_{\max}^3$ where $k_{\max} \approx \sqrt{N} \approx 10$: two to three orders of
  magnitude. That is exactly the width of spectrum needed to produce visibly different per-position
  slopes.

### 3.5 Basri, Galun, Geifman, Jacobs, Kasten, Kritchman (2020), "Frequency Bias in Neural Networks for Input of Non-Uniform Density", ICML 2020, arXiv:2003.04560

The single most relevant paper for the non-uniform-visitation extension.

- **What changes with non-uniform density.** With training points drawn from a density $p$, the
  relevant operator is no longer the kernel $k$ but the density-weighted operator
  $\int k(x,z)\thinspace p(z)\thinspace f(z)\thinspace dz = \lambda f(x)$ (their Eq. 7). It is symmetrized by
  $\tilde k(x,z) = p^{1/2}(x)k(x,z)p^{1/2}(z)$, so the eigenvalues are real.
- **Proposition 1 (eigenfunctions).** For a piecewise-constant density on the circle, the
  eigenfunctions have the form $f(x) = a(p(x))\cos\negthinspace \bigl(qZ\Psi(x)+b(p(x))\bigr)$ with
  $\Psi(x) = \int_{-\pi}^{x}\sqrt{p(\tilde x)}\thinspace d\tilde x$. On a region where $p = p_j$ is constant
  this is a cosine of frequency proportional to $\sqrt{p_j}$. The eigenfunctions satisfy the
  ordinary differential equation $f''(x) = -\dfrac{p(x)}{\pi\lambda} f(x)$.
- **Theorem 1 (their main rate).** For a piecewise-constant density on $S^1$ and a target of
  frequency $\kappa$, the number of gradient-descent iterations needed to reach
  $\lVert g(x)-u^{(t)}(x)\rVert < \delta$ is $\tilde O(\kappa^2/p_{\min})$, where $p_{\min}$ (their $p$-star) is the
  **minimum** density over the input space and $\tilde O$ hides logarithms. Empirically on $S^{d-1}$
  the exponent becomes $\kappa^d$, giving the abstract's $O(\kappa^d/p(x))$ per point.
- **Which quantity.** The number of iterations for the network's *function* error to fall below a
  fixed threshold, region by region. Not a per-step decay exponent.
- **Theorem 2.** The Arora residual formula holds for deep finite-width fully connected networks
  too: $\lVert y - u(t)\rVert = \sqrt{\sum_i (1-\eta\lambda_i)^{2t}(v_i^\top y)^2} \pm \epsilon$,
  under a width requirement $m \ge \Omega(n^{24}L^{12}\log^5 m/(\delta^8\tau^6))$ — astronomically
  large, so treat it as a structural statement rather than a usable bound.
- **The sentence that matters most for the RND extension.** Because the local convergence rate is
  proportional to the local density $p(x)$, the error at $x$ falls like
  $\exp(-c\thinspace p(x)\thinspace t)$, and $p(x)\thinspace t$ is (in expectation) exactly the number of samples drawn at $x$.
  **So the density-weighted kernel already produces "each point decays on its own visit-count
  clock", in the exponential-decay sense.** What it does not give is a *power law* in the visit
  count, which is what the RND goal needs; that requires the step-size or preconditioning
  intervention of Section 9.
- **Their empirical finding about depth.** Under non-uniform density the eigenfunctions of the
  deep-network kernel look indistinguishable from the two-layer ones, so the $O(\kappa^d/p_{\min})$
  picture carries over to deeper networks. Depth changes constants, not the mechanism.

### 3.6 Bietti, Mairal (2019), "On the Inductive Bias of Neural Tangent Kernels", NeurIPS 2019, arXiv:1905.12173

- Studies the reproducing-kernel Hilbert space of the neural tangent kernel: smoothness,
  approximation and stability properties, plus stability of convolutional kernels to image
  deformations.
- It is the standard reference (cited as such by Basri et al. 2020 and by Geifman et al. 2024) for
  the statement that on the sphere the neural-tangent-kernel eigenvalues for spherical harmonics of
  frequency $k$ decay as $k^{-d}$. **I did not verify that formula inside the paper's own text**,
  only that two later papers attribute it there; treat the attribution as second-hand.

### 3.7 Cao, Fang, Wu, Zhou, Gu (2019/2021), "Towards Understanding the Spectral Bias of Deep Learning", IJCAI 2021, arXiv:1912.01198

- **What it shows.** The training process of an over-parameterized deep network decomposes along the
  eigenfunctions of the neural tangent kernel; each direction has its own convergence rate,
  determined by its eigenvalue. On the unit sphere, lower-degree spherical harmonics are easier to
  learn.
- This is the deep-network generalization of Arora's per-eigendirection statement. It confirms that
  the "one rate per eigen-direction" structure — the root cause of position-dependence — is not an
  artifact of the two-layer analysis.

### 3.8 Yang, Salman (2019), "A Fine-Grained Spectral Perspective on Neural Networks", arXiv:1907.10599

- Derives fast algorithms for computing the eigenvalues of the conjugate kernel and the neural
  tangent kernel when the data is uniform on the Boolean cube, and shows the spectrum is the same in
  high dimension for isotropic Gaussian or uniform-on-sphere data.
- **Use for us.** It is the practical reference for asking how architecture choices — depth, the
  activation, the ratio of weight to bias variance at initialization — move the eigenvalue
  spectrum, with runnable code (`NNspectra`). If we want to search architectures for a flatter
  spectrum rather than guess, this is the tool-shaped paper.

### 3.9 Murray, Jin, Bowman, Montúfar (2023), "Characterizing the spectrum of the NTK via a power series expansion", ICLR 2023, arXiv:2211.07844

- **What it shows.** A power-series expansion of the infinite-width neural tangent kernel for
  arbitrarily deep feedforward networks, with coefficients expressed through the Hermite
  coefficients of the activation function and the depth. Faster decay of the activation's Hermite
  coefficients gives faster decay of the kernel coefficients. The largest eigenvalue takes an
  $\Omega(1)$ fraction of the trace, and there are $O(1)$ outlier eigenvalues.
- **Two facts to carry.** (i) The activation function directly controls how fast the spectrum
  decays — so the choice of activation changes the heterogeneity directly, and is not a detail.
  (ii) **The top eigenvalue always carries a constant fraction of the trace**, which means a
  perfectly flat spectrum is not reachable by architecture choice alone under standard
  initialization; the flat-spectrum condition of Section 2.3 has to be bought by preconditioning or
  by feature design, not by depth.
- Geifman et al. 2024 cite this paper for the further statement that with some activations the
  eigenvalue decay can even be exponential (worse than the polynomial $k^{-d}$ of ReLU).

### 3.10 Bordelon, Canatar, Pehlevan (2020), "Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks", ICML 2020, arXiv:2002.02561

- Analytic expressions for the generalization error of kernel regression as a function of the number
  of samples, decomposed **per spectral mode**: as the training set grows, kernel machines and wide
  networks fit successively higher modes of the target.
- **Relevance.** It is the sample-count analogue of the time-count story: the same eigenvalue
  ordering governs both "which modes are learned first in time" and "which modes are learned first
  as data accumulates". Useful if we want to reason about the maze positions as a growing sample set
  rather than a fixed batch.

### 3.11 Velikanov, Yarotsky (2021), "Universal scaling laws in the gradient descent training of neural networks", arXiv:2105.00507 (the NeurIPS 2021 version is titled "Explicit loss asymptotics in the gradient descent training of neural networks")

This paper explains where a power-law-looking curve comes from **even at a constant step size**, and
it is important that we do not misattribute our slopes.

- **Setting.** Lazy/linearized training, loss $L(t) = \frac12\sum_n e^{-2\lambda_n t}|c_n|^2$ where
  $\lambda_n$ are the eigenvalues of the kernel integral operator and $c_n$ the expansion
  coefficients of the initial error.
- **Assumptions.** Power-law eigenvalues $\lambda_n \sim \Lambda n^{-\nu}$ and power-law tail sums of
  the coefficients $s_n = \sum_{k\ge n}|c_k|^2 \sim K n^{-\kappa}$.
- **Theorem 1 (their Eq. 14 / B.2).** The loss satisfies
  $L(t) \sim \dfrac{K}{2} \Gamma\bigl(\kappa/\nu+1\bigr) (2\Lambda t)^{-\kappa/\nu}$,
  so the exponent is $\xi = \kappa/\nu$.
- **Which quantity: the LOSS**, i.e. one half of the squared residual, aggregated. In terms of the
  residual 2-norm the exponent is $\kappa/(2\nu)$. A residual-norm slope of $-1/2$ corresponds to
  $\kappa/\nu = 1$.
- **The warning for us.** A straight line of slope about $-0.5$ in a log-log plot of the aggregate
  residual is **not** evidence that a $1/t$ step schedule is doing what we think. A power-law
  spectrum plus a constant step size produces the same straight line. The two mechanisms are
  distinguished by their per-position behaviour and by how the slope reacts to changing $a$ in
  $\eta_n = a/n$: under the schedule mechanism the slope should scale linearly with $a$; under the
  spectrum mechanism it should not.

### 3.12 Bahri, Dyer, Kaplan, Lee, Sharma (2024), "Explaining Neural Scaling Laws", PNAS 121, arXiv:2102.06701

- Identifies four scaling regimes (variance-limited and resolution-limited, each in dataset size and
  model size). In the large-width limit the resolution-limited exponents are obtained from the
  spectrum of the associated kernel; they present evidence of a duality relating the width and
  dataset exponents.
- **Use for us.** Background support for "power-law kernel spectrum produces power-law error"; the
  same caution as Section 3.11 applies about which quantity is being scaled (test loss against
  dataset or model size, not residual against training step).

---

## 4. Summary answer to Question 2

For a small ReLU network on a low-dimensional input, the neural-tangent-kernel spectrum is
**polynomially decaying and wide**, and the eigenvectors are **oscillatory functions of position**:

- On the circle with bias: $\lambda_k \sim 1/(\pi^2 k^2)$; without bias, odd frequencies at or above
  3 are exactly in the null space (Basri et al. 2019).
- On the $d$-sphere: $\lambda_k \sim k^{-d}$ (Basri et al. 2019 empirically; attributed to Bietti &
  Mairal 2019 and Bietti & Bach 2020 by later work).
- The eigenvectors are (discretized) Fourier modes or spherical harmonics under uniform sampling —
  so under uniform sampling the eigenvectors are *delocalized*, $w_{ij} \approx 1/N$, and the
  per-position curves are nearly identical even though the spectrum is wide.
- **Under non-uniform sampling the eigenfunctions become locally-frequency-modulated cosines whose
  local frequency scales as $\sqrt{p(x)}$ and whose amplitude depends on $p(x)$** (Basri et al.
  2020, Proposition 1). This *localizes* the eigenvectors: mass concentrates in dense regions for
  high modes, in sparse regions for low modes. That is precisely how $w_{ij}$ stops being uniform
  and per-position rates start to differ.
- Depth and skip connections do not change the frequency exponent (Basri et al. 2019, measured).
  The activation does (Murray et al. 2023).
- The top eigenvalue always carries a constant fraction of the trace (Murray et al. 2023), so the
  spectrum is never flat by default.

**Implication for the 100 maze positions.** They are not uniformly spread on a sphere: the maze has
walls, corridors and a boundary, so the point set has non-uniform effective density and non-trivial
geometry. Both ingredients for eigenvector localization are present. This, more than the eigenvalue
spread alone, is my best guess for the observed per-position heterogeneity — and it is directly
testable (Section 10.1).

---

## 5. Question 3 — interventions that flatten or condition the spectrum

### 5.1 Fourier features — Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi, Barron, Ng (2020), "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020, arXiv:2006.10739

The closest published intervention to what we need, and it is in exactly our regime (dense
coordinates in low dimension).

- **The mechanism they state (their Eq. 4).** With $K = Q\Lambda Q^\top$,
  $Q^\top(\hat y_{\mathrm{train}}^{(t)} - y) \approx -e^{-\eta\Lambda t}Q^\top y$, so the $i$-th
  component of the absolute error in the kernel eigenbasis decays approximately exponentially at
  rate $\eta\lambda_i$. **Which quantity: the coefficient of one eigen-direction of the training
  residual.** For a conventional multilayer network the eigenvalues decay fast, so high-frequency
  components converge extremely slowly.
- **The mapping.**
  $\gamma(v) = [a_1\cos(2\pi b_1^\top v), a_1\sin(2\pi b_1^\top v), \dots,
  a_m\cos(2\pi b_m^\top v), a_m\sin(2\pi b_m^\top v)]^\top$, giving
  $k_\gamma(v_1,v_2) = \gamma(v_1)^\top\gamma(v_2) = \sum_j a_j^2\cos(2\pi b_j^\top(v_1-v_2))
  = h_\gamma(v_1-v_2)$.
- **The two properties they want, quoted in substance.** (i) The composed kernel becomes
  **stationary (shift-invariant)** — a function of $v_1 - v_2$ only — so it acts as a convolution
  over the input domain instead of a dot-product kernel on a sphere. (ii) Its **bandwidth is
  tunable** through the $a_j$ and the scale of the $b_j$: a wider kernel (slower spectral falloff)
  trains high frequencies faster, but too wide causes high-frequency artifacts, and they tune
  between underfitting and overfitting empirically (their Section 5).
- **Why property (i) is the important one for us, and is under-appreciated.** On a *uniformly
  spaced* point set a stationary kernel gives a circulant Gram matrix, whose eigenvectors are the
  discrete Fourier modes. Those eigenvectors have $|v_{j,i}| = 1/\sqrt{N}$ for **every** $i$ and
  $j$, so $w_{ij} = 1/N$ exactly: **every position gets the identical mixture over the spectrum, and
  hence the identical decay curve, regardless of how wide the spectrum is.** Stationarity buys
  per-position uniformity directly, without flattening anything. This is the second of the two
  routes identified in Section 2.1(3).
- **The catch in our setting.** Maze positions are not a full uniform grid — walls remove points and
  the boundary breaks the shift symmetry — so the Gram matrix is only approximately circulant and
  the eigenvectors only approximately delocalized. Expect improvement, not exactness.
- **The second catch.** Property (ii) alone does not make the spectrum *flat*; it moves the falloff.
  A wide-bandwidth Fourier feature scale pushes the Gram matrix towards a multiple of the identity
  (adjacent points decorrelate), which does flatten it — at the cost of the network no longer
  generalizing between nearby positions at all. For a count-based bonus that trade is arguably
  correct; for reinforcement-learning use it removes the smoothing that makes random network
  distillation better than a table. The Fourier feature scale moves continuously between the two.

### 5.2 Wang, Wang, Perdikaris (2021), "On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks", Computer Methods in Applied Mechanics and Engineering 384:113938, arXiv:2012.10047

- Shows through neural-tangent-kernel analysis that these networks are biased towards learning along
  the dominant eigen-directions of their limiting kernel, and builds architectures with
  **multi-scale** random Fourier features (several feature blocks at different scales, concatenated)
  so that several bands of the spectrum are boosted at once.
- **Use for us.** If one single Fourier scale over-flattens (removes all generalization) and another
  under-flattens, the multi-scale construction is the documented way to cover a range; it is a
  cheap thing to try after the single-scale version.

### 5.3 Sitzmann, Martel, Bergman, Lindell, Wetzstein (2020), "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020, arXiv:2006.09661

- Replaces the activation with $\sin$, with a matched initialization scheme derived from the
  activation statistics; represents images, audio, wavefields and their derivatives much better than
  ReLU networks.
- **What it does and does not show.** It is an architecture/initialization result with an empirical
  case, not a theorem about the eigenvalue spectrum. It is a reasonable alternative to Fourier
  features (the first layer of a sine network is essentially a learned Fourier feature layer), but
  cite it for the empirical improvement, not for a spectral guarantee.

### 5.4 Hong, Siegel, Tan, Xu (2022), "On the Activation Function Dependence of the Spectral Bias of Neural Networks", arXiv:2208.04924

- Explains the spectral bias of ReLU networks by a connection to finite element methods, and
  **predicts and then verifies empirically that replacing the activation with a piecewise-linear
  B-spline (the "hat" function) removes the spectral bias**; networks with the hat activation train
  significantly faster with both stochastic gradient descent and Adam.
- **Why this is interesting for us.** The hat function is compactly supported, so its induced kernel
  is local: the Gram matrix over well-separated points moves towards diagonal. This is the same
  effect as a narrow-bandwidth kernel, obtained by changing one line of the model. It is a
  one-parameter experiment worth running.
- **Caveat.** I did not find a formal eigenvalue-decay theorem in the abstract; the removal of
  spectral bias is stated as a prediction verified empirically.

### 5.5 Cai, Zhu, Shen, Wang, Cao (2024), "Towards the Spectral bias Alleviation by Normalizations in Coordinate Networks", arXiv:2407.17834

- Measures what batch normalization and layer normalization do to the neural-tangent-kernel
  eigenvalues of coordinate-based networks: they **greatly reduce the maximum eigenvalue and the
  variance of the eigenvalues while only slightly changing the mean**, which shifts the eigenvalue
  distribution upward as a whole.
- **Reading this honestly.** Reducing the variance of the eigenvalues is exactly "flattening the
  spectrum", which is the condition in Section 2.3. This is the most direct empirical support for
  "add a normalization layer" as an intervention. But it is an empirical measurement on
  coordinate-based image/signal fitting, not a theorem, and the paper is a preprint. Treat it as a
  cheap experiment to run (add layer normalization inside the predictor) with a measurable
  prediction: the eigenvalue variance of the empirical kernel should drop, and the per-position
  slopes should tighten.

### 5.6 Depth, width, residual connections, input normalization

- **Depth and skip connections: no measured effect on the frequency exponent.** Basri et al. 2019
  measured $O(k^{1.93})$ for a shallow network with bias, $O(k^{1.94})$ for a 5-hidden-layer
  network, and $O(k^{2.11})$ for a 10-layer residual network on the circle. Do not expect depth to
  fix heterogeneity.
- **Bias terms matter.** Without bias, odd frequencies at or above 3 are in the exact null
  space of the two-layer ReLU kernel (Basri et al. 2019, their Theorem 4). A position whose residual
  has mass on a null direction never decays at all. **Check that our predictor has bias terms** —
  this is a five-second check with a potentially decisive effect.
- **Width.** Width moves the finite-width network towards the kernel regime; it does not change the
  limiting spectrum's shape. More width should make the per-position curves more predictable (better
  agreement with the boxed formula), not more uniform.
- **Input normalization / input geometry.** Murray et al. 2023 make the kernel coefficients depend
  on the *effective rank of the input Gram matrix*; Rahaman et al. 2019 show that a more complex
  data manifold makes high frequencies easier. Our inputs are $[x, y, 0, 0]$ — two constant
  coordinates that contribute nothing but do change the norm and the effective input geometry.
  **Test: drop the two zero coordinates, and separately test centering and scaling the maze
  coordinates to the unit circle or unit square.** These are free changes with a real chance of
  moving the spectrum.

### 5.7 Preconditioning — the interventions that actually give a guarantee

These two papers are the strongest published support for making the decay rate identical at every
point, because they change the dynamics matrix directly instead of hoping architecture does it.

#### 5.7.1 Geifman, Barzilai, Basri, Galun (2024), "Controlling the Inductive Bias of Wide Neural Networks by Modifying the Kernel's Spectrum", Transactions on Machine Learning Research (02/2024), arXiv:2307.14531

- **The update rule (their Eq. 7).** $w_{t+1} = w_t - \eta\thinspace \nabla_w f(X,w_t)^\top S\thinspace r_t$, where
  $r_t$ is the residual vector on the $n$ training points and $S \in \mathbb{R}^{n\times n}$,
  $S \succ 0$, is a preconditioner **on the output side**, not on the parameters. Cost per
  iteration is $n^2$, not $p^2$ — for us $n = 100$, so this is free.
- **Theorem 4.1.** Under their assumptions, for
  $\eta_0 < \dfrac{2}{\lambda_{\min}(KS)+\lambda_{\max}(KS)}$ and width above a threshold, with high
  probability over initialization, $r_t = (I-\eta_0 K S)^t y \pm \xi(t)$ with
  $\lVert\xi\rVert_2 \le \epsilon$.
- **Why this is the key result.** It says the residual dynamics can be made $(I - \eta_0 KS)^t$ for
  a preconditioner of our choosing, with a proof for a **finite-width** network. Choosing
  $S \approx K^{-1}$ gives $r_t = (1-\eta_0)^t y$: **every coordinate of the residual — every
  position — decays at exactly the same geometric rate**, which is condition (b) of Section 2.3 met
  exactly rather than approximately.
- **They also prove consistency**: preconditioning does not change the global minimum reached, only
  the trajectory. And they introduce Modified Spectrum Kernels, a way to construct a kernel with
  prescribed eigenvalues $g(\lambda_i)$ from an existing kernel — the tool for "give me the spectrum
  I want" rather than "give me a faster one".

#### 5.7.2 Zhang, Martens, Grosse (2019), "Fast Convergence of Natural Gradient Descent for Over-Parameterized Neural Networks", NeurIPS 2019, arXiv:1905.10961

- **The update rule (their Eq. 3).** With more parameters than samples, the natural-gradient update
  with the generalized inverse is $\theta(k+1) = \theta(k) - \eta\thinspace J^\top (JJ^\top)^{-1}(u-y)$.
  Note $JJ^\top$ **is** the neural-tangent-kernel Gram matrix on the training points, so this is
  exactly the $S = K^{-1}$ choice of the previous paper.
- **Theorem 1 (quoted).** Under Condition 1 (the Jacobian has full row rank at initialization) and
  Condition 2 (the Jacobian is stable in a neighbourhood of the initialization, with constant
  $C < 1/2$), with step size $\eta \le \dfrac{1-2C}{(1+C)^2}$, for $k = 0,1,2,\dots$ the residual
  satisfies $\lVert u(k)-y\rVert_2^2 \le (1-\eta)^k \lVert u(0)-y\rVert_2^2$.
- **Which quantity, and why it is the sharpest statement here.** It is the aggregate squared
  residual, and the rate $(1-\eta)$ contains **no eigenvalue of the kernel at all**. In the exactly
  linearized case the same algebra holds coordinate-wise: $u(k)-y = (1-\eta)^k (u(0)-y)$, so **every
  position's residual is multiplied by exactly $(1-\eta)$ per step.** Combine with the step schedule
  of Section 2.3 and every position follows $n^{-a}$ with the same $a$.
- They also analyse K-FAC (an approximate natural gradient) and show it converges linearly too,
  requiring more over-parameterization; and they argue the faster convergence does not cost
  generalization.
- **Caveats.** Conditions 1 and 2 are the linearization assumptions in disguise; with 100 points and
  a 256-unit hidden layer, Condition 1 (full row rank of a $100\times p$ Jacobian) is easy, and
  Condition 2 is the one to check empirically. Also $(JJ^\top)^{-1}$ needs a damping term
  $(JJ^\top + \mu I)^{-1}$ in practice, and this project has already seen a float32
  Cholesky fail on an accumulating covariance matrix — keep this solve in float64.

### 5.8 Per-point loss weights — Chen, Howard, Stinis (2024/2025), "Self-adaptive weights based on balanced residual decay rate for physics-informed neural networks and deep operator networks", Journal of Computational Physics, arXiv:2407.01613

This is the only paper I found whose *stated objective* is the same as ours: make the residual decay
rate the same at every training point.

- **The problem they identify**, in their words: the failure of plain physics-informed networks
  arises from the significant discrepancy in the convergence rate of residuals at different training
  points, where the slowest rate dominates the overall convergence. They measure decay rates varying
  over several orders of magnitude across points.
- **Their measurement of a per-point decay rate.** Define $R$ as that point's residual, $M$ as an
  exponential moving average of $R^4$ updated at step $n$ as
  $M_n = \beta_c M_{n-1} + (1-\beta_c) R_n^4$ with smoothing factor $\beta_c$, and
  $\varepsilon = 10^{-14}$. Their "inverse residual decay rate" is
  $\text{irdr} = \dfrac{R^2}{\sqrt{M+\varepsilon}}$.
  The interpretation: the quantity is below 1 when the residual is
  falling, above 1 when it is rising, near 1 when it is flat, and for an exponentially decaying
  residual $R = R_0 e^{-\lambda n}$ it is inversely proportional to $\lambda$ for large $\lambda$.
- **Their intervention.** Set the per-point loss weight proportional to that quantity:
  $w^{\text{ref}}_n = \text{irdr}_n / \text{mean}(\text{irdr}_n)$, then smooth it,
  $w_n = \beta_w w_{n-1} + (1-\beta_w) w^{\text{ref}}_n$. Slowly-decaying points get more weight.
- **Why it works, in our language.** Weighting the per-point loss by $w_i$ changes the residual
  dynamics matrix from $K$ to $K\thinspace \text{diag}(w)/N$ (they state this explicitly). It is
  preconditioning with a **diagonal** $S = \text{diag}(w)$ — the cheap, gradient-free cousin of
  Section 5.7, needing no kernel and no matrix solve.
- **What they report.** The distribution of per-point decay rates becomes significantly more
  uniform, and the worst (slowest) rate improves markedly; the weights stay bounded, training
  variance drops, and the cost is negligible.
- **Limitations for us.** A diagonal preconditioner can only equalize rates to the extent that the
  off-diagonal coupling of $K$ is weak; there is no theorem that a diagonal $S$ can make $KS$ have a
  flat spectrum. And their target is "all points decay at least as fast as the fastest", not "all
  points follow a prescribed power law".

### 5.9 Wang, Yu, Perdikaris (2021), "When and why PINNs fail to train: A neural tangent kernel perspective", Journal of Computational Physics 449:110768, arXiv:2007.14527

- Derives the neural tangent kernel of a physics-informed network and shows it converges to a
  deterministic kernel that stays constant in the infinite-width limit; uses the eigenvalues of that
  kernel to explain the discrepancy in convergence rates between loss terms, and proposes weights
  computed from the kernel eigenvalues to balance those rates.
- **Relevance.** This is the group-level (not per-point) version of Section 5.8, and it is the paper
  that established "compute the kernel, read the eigenvalues, set weights to equalize rates" as a
  method. Our problem is the per-point version of the same recipe.

---

## 6. Question 4 — is there work on uniform per-point convergence rather than fast aggregate convergence?

Short answer: **very little, and what exists is in the physics-informed-network literature, not in
the neural-tangent-kernel theory literature.**

- **The theory papers all report aggregate quantities.** Arora et al. 2019 Theorem 4.1, Basri et al.
  2020 Theorem 2, Geifman et al. 2024 Theorem 4.1 and Zhang et al. 2019 Theorem 1 are all statements
  about $\lVert y - u(k)\rVert_2$, the 2-norm over the whole training set. The per-position
  statement is obtained by taking rows, as in Section 2.1, but no paper I found states or studies
  it.
- **Basri et al. 2020 is the one paper that gives a genuinely per-point rate**: convergence at
  $x$ takes time $O(\kappa^d/p(x))$. It is a per-point statement in the input space and it is the
  right shape for the non-uniform-visitation extension. But it is a statement about the *time to
  reach a threshold*, not about a decay exponent, and its route to a per-point result is through
  regions of constant density, not through arbitrary point sets.
- **Chen, Howard, Stinis (Section 5.8) is the closest match to the actual goal.** They measure a
  per-point decay rate, observe that the distribution of those rates spans orders of magnitude, and
  propose a weighting that makes the distribution uniform. Their objective is uniformity of the
  rate, exactly ours. What they do not do is prescribe *which* rate, and they never ask for a power
  law.
- **The uniform-versus-pointwise distinction is classical in nonparametric statistics** (uniform
  consistency rates of kernel regression estimators over the input space, versus pointwise rates)
  but that literature is about the estimator's statistical error as data accumulates, not about a
  gradient-descent trajectory, and I did not find a bridge to the neural-tangent-kernel setting.
- **Nobody appears to have asked our exact question**: find a training procedure such that the
  residual at every input point follows a prescribed power law with a prescribed exponent. That is
  a gap, and it is a reasonable thing to write up if the construction in Section 9 works.

---

## 7. The non-uniform-visitation extension, worked out

Suppose position $i$ is sampled with probability $p_i$ and each step does a stochastic-gradient
update on that one point. The parameter update is $\Delta\theta = -\eta J_i^\top e_i$, so the
function value at position $l$ changes by $\Delta e_l = -\eta K_{li} e_i$. Taking the expectation
over which point is drawn,

$$\mathbb{E}[e(n+1)] \thickspace =\thickspace (I - \eta K P)\thinspace e(n), \qquad P = \text{diag}(p_1,\dots,p_N).$$

Read three consequences off this.

1. **If $K$ is (close to) a multiple of the identity, $K = cI$, the dynamics decouple exactly:**
   $e_i(n+1) = (1-\eta c\thinspace p_i)\thinspace e_i(n)$, so $e_i(n) \approx e^{-\eta c\thinspace p_i n} = e^{-\eta c\thinspace n_i}$
   where $n_i = p_i n$ is position $i$'s own visit count. **The clock at each position becomes its
   own visit count, automatically.** No counters, no bookkeeping.
2. **A global step-size schedule cannot do this**, exactly as the project already suspected: with a
   global $\eta_n = a/n$ and $K = cI$, the exponent becomes $a c\thinspace p_i$, i.e.
   $b_i \sim n^{-acp_i}$ — a power law in the *global* step count with a position-dependent
   exponent, which is not the wanted $n_i^{-1/2}$. To get $n_i^{-1/2}$ the step size must be
   $a/n_i$, i.e. per position.
3. **A preconditioner supplies the per-position counters for free.** With a linear head on frozen
   features $\varphi(x)\in\mathbb{R}^m$ and the update
   $\theta \leftarrow \theta - \eta A^{-1}\varphi_i e_i$ where
   $A = \mu I + \sum_{\text{visits}} \varphi\varphi^\top$, the residual at the visited point is
   multiplied by $(1 - \eta\thinspace \varphi_i^\top A^{-1}\varphi_i)$. If the features are orthonormal over
   the point set then $\varphi_i^\top A^{-1}\varphi_i = \dfrac{1}{\mu + n_i}$ exactly, so the update
   is $e_i \leftarrow \bigl(1-\dfrac{\eta}{\mu+n_i}\bigr) e_i$, and after $n_i$ visits
   $b_i \sim n_i^{-\eta}$.
   Setting $\eta = 1/2$ gives $b_i \sim n_i^{-1/2}$ at every position, under any visitation
   distribution. **The matrix $A$ is the same matrix that defines the elliptical bonus already
   implemented in this project** (`rnd_elliptical` / `EllipticalBonus`); the quantity
   $\varphi^\top A^{-1}\varphi$ is the squared elliptical bonus. So the preconditioner and the
   bonus are the same object used two ways, and the "generalized count" $1/\varphi^\top A^{-1}\varphi$
   is what replaces $n_i$ when features are not orthonormal.
4. **Basri et al. 2020 says the density enters the rate linearly even without a diagonal kernel.**
   Their $O(\kappa^d/p(x))$ means the local convergence rate is proportional to $p(x)$, so the
   exponential clock at $x$ is already $p(x)\thinspace t = n_x$. What the general (non-diagonal) kernel does
   *not* give is the power-law shape; that has to come from the step-size schedule or the
   preconditioner.

---

## 8. Synthesis: three conditions, and what changes each one

The goal — $b_i(0)$ equal for all $i$, and $b_i(n) \propto n^{-1/2}$ for all $i$ — decomposes into
three independent conditions. Fixing one does not fix the others, which is why partial interventions
have produced partial results.

1. **Condition A — same starting value.**
   - What it requires: $b_i(0)$ identical across $i$.
   - What changes it: normalize the target's output to unit norm at each point, and centre the predictor so
     its output is exactly 0 at step 0 (Section 9.1). Also holds approximately for free when
     $d_{\text{out}}$ is large (Section 2.1, item 4).
2. **Condition B — same decay curve shape at every position.**
   - What it requires: either $w_{ij}$ independent of $i$ (delocalized eigenvectors) or $\lambda_j$
     independent of $j$ (flat spectrum).
   - What changes it, for delocalization: a stationary kernel through Fourier features (Tancik et al.) on an
     evenly spread point set.
   - What changes it, for flatness: output-side preconditioning with $S \approx K^{-1}$ (Geifman et al.;
     Zhang et al.), normalization layers (Cai et al.), narrow local kernels (hat activation,
     high-scale Fourier features).
3. **Condition C — that common curve is $n^{-1/2}$.**
   - What it requires: the single effective rate $\rho$ satisfies $\eta_n \rho = 1/(2n)$.
   - What changes it: the step-size schedule $\eta_n = a/n$ with $a$ calibrated so $a\rho = 1/2$; or exact
     preconditioning, where $\rho = 1$ by construction so $a = 1/2$ needs no calibration.

The current state of the project is: condition A approximately holds, condition C is being attempted
by the $1/t$ schedule, and **condition B is the one that is missing**. That matches the reported
symptom exactly — a correct aggregate exponent with heterogeneous per-position exponents.

---

## 9. Concrete method ideas, in the order I would try them

Each is stated so it can be implemented directly in the small setting (100 fixed positions,
$4 \to 256 \to \text{ReLU} \to 128$).

### 9.1 Make the start exactly equal (do this first, it is free and removes a confound)

- Replace the target by a normalized target $\tilde f(x) = f(x)/\lVert f(x)\rVert_2$, so
  $\lVert \tilde f(x_i)\rVert = 1$ at every position.
- Replace the predictor output by a **centred** predictor
  $\tilde g_\theta(x) = g_\theta(x) - g_{\theta_0}(x)$, where $g_{\theta_0}$ is a frozen copy of the
  predictor at initialization. Then $\tilde g$ outputs exactly 0 at step 0 without zeroing any
  weights (zeroing the last layer would kill the trunk's gradient and change the kernel; the
  centred form leaves the kernel unchanged).
- Result: $b_i(0) = 1$ exactly at every position, and requirement (1) is met by construction rather
  than by luck.
- This is the same construction as the frozen prior network in Osband, Aslanides, Cassirer (2018),
  "Randomized Prior Functions for Deep Reinforcement Learning", NeurIPS 2018, arXiv:1806.03335, used
  there to give an ensemble a prior; here it is used only to fix the starting value.

### 9.2 Output-side preconditioned full-batch gradient descent (the method with a theorem)

- Compute the empirical kernel Gram matrix $K$ on the 100 positions once at initialization:
  $K_{ij} = \langle \nabla_\theta g(x_i), \nabla_\theta g(x_j)\rangle$ (per output channel, averaged
  over channels). With 100 points this is a $100\times100$ matrix and a handful of backward passes.
- Precondition on the output side, exactly Geifman et al.'s Eq. 7 with
  $S = (K+\mu I)^{-1}$: the parameter update becomes
  $\Delta\theta = -\eta_n \sum_{i,l} \nabla_\theta g(x_i)\thinspace S_{il}\thinspace e_l$.
- Set the step schedule $\eta_n = \dfrac{1}{2n}$ (no calibration needed: with $S = K^{-1}$ the
  effective rate is 1, so $a = 1/2$ directly gives $n^{-1/2}$).
- **Predicted behaviour**: $b_i(n) = \prod_{k}(1-\tfrac{1}{2k}) \sim n^{-1/2}$ at every position
  simultaneously, with no heterogeneity, backed by Geifman et al. Theorem 4.1 for finite width and
  Zhang et al. Theorem 1 for the exactness of the $(1-\eta)$ rate.
- **Practical notes.** Damping $\mu$ matters (start at $10^{-3}\thinspace \text{trace}(K)/N$ and sweep);
  the solve must be float64 (this project has already hit a float32 Cholesky failure on an
  accumulating covariance matrix); recompute $K$ occasionally if the network leaves the linearized
  regime, and log $\lVert K_t - K_0\rVert_F$ so we know whether it did.
- **Cost.** Negligible at $N=100$. The method's real limitation is that it needs the whole point set
  in a batch; the online version is 9.4.

### 9.3 Per-point loss weights that equalize measured decay rates (the cheap version)

- Implement the balanced-residual-decay-rate weighting of Chen, Howard, Stinis: per position keep an
  exponential moving average of $b_i^4$, form
  $\text{irdr}_i = b_i^2/\sqrt{\overline{b_i^4}+10^{-14}}$, normalize by its mean over positions,
  smooth with $w \leftarrow \beta_w w + (1-\beta_w)w^{\text{ref}}$, and weight position $i$'s
  contribution to the loss by $w_i$.
- **Predicted behaviour**: the spread of per-position slopes tightens substantially; the common
  slope is then set by the global step schedule as before.
- **Advantages.** No kernel, no matrix solve, works online, works with any optimizer, and it can be
  layered on top of any of the other ideas. **Disadvantage.** It equalizes the *measured* rate, and
  only through a diagonal preconditioner, so there is no guarantee it reaches the prescribed
  $-1/2$; it also introduces two smoothing constants to tune ($\beta_c$, $\beta_w$).
- Add a variant worth testing: instead of the moving-average ratio, weight directly by the *target
  over actual* slope error, $w_i \propto \exp\bigl(\gamma\thinspace [\text{slope}_i - (-1/2)]\bigr)$ where
  $\text{slope}_i$ is a windowed regression of $\log b_i$ on $\log n$. This targets the prescribed
  exponent rather than uniformity alone, which is what the published method lacks.

### 9.4 Linear head on frozen features with recursive-least-squares preconditioning (the version that extends to non-uniform visitation)

- Freeze the trunk (or use frozen random Fourier features $\varphi(x)$ of the maze coordinate,
  per Tancik et al. Eq. 5), and train only a linear head $W\varphi(x)$ against the target.
- Maintain $A = \mu I + \sum_{\text{visits}} \varphi\varphi^\top$ in float64 and update
  $W \leftarrow W - \eta\thinspace (A^{-1}\varphi_i)\thinspace e_i^\top$ with $\eta = 1/2$.
- **Predicted behaviour** (derivation in Section 7, item 3): $b_i \sim n_i^{-1/2}$ where $n_i$ is
  position $i$'s own visit count, exactly, for any visitation distribution — the hard extension
  solved rather than approximated.
- **Why it is worth taking seriously beyond this toy setting.** $A$ is the same matrix the elliptical
  bonus already builds, so the machinery exists in the repository; and $\varphi^\top A^{-1}\varphi$
  is the generalized inverse count, which degrades gracefully when features overlap (nearby
  positions share visits, which is the generalization one wants in reinforcement learning).
- **Cost.** $A$ is $m\times m$ for feature width $m$; use the Sherman–Morrison rank-one update so no
  inverse is ever recomputed, or keep $m$ small (a few hundred) and refactorize periodically.

### 9.5 Make the kernel stationary: Fourier features on the input

- Map the maze position through
  $\gamma(v) = [\cos(2\pi Bv), \sin(2\pi Bv)]$ with $B$ drawn from an isotropic Gaussian of scale
  $\sigma$, then feed $\gamma(v)$ to the same network. Sweep $\sigma$ over about three orders of
  magnitude.
- **Predicted behaviour**: as $\sigma$ grows the Gram matrix moves from a wide-spectrum dot-product
  kernel to a near-diagonal one; per-position slopes should tighten monotonically, and at large
  $\sigma$ the bonus stops generalizing between neighbouring positions at all. The sweep maps out
  the trade-off explicitly, and the useful setting is the smallest $\sigma$ at which the slope
  spread is acceptable.
- **Caveat already noted.** The maze point set is not a uniform grid, so the exact circulant argument
  (identical mixture weights $w_{ij}$ at every position) is only approximate. Expect the largest
  residual heterogeneity at wall-adjacent and boundary positions; check them specifically.

### 9.6 Cheap architecture and input changes to run alongside (each one line)

- **Confirm the network has bias terms.** Without them the two-layer ReLU kernel has an exact null
  space at odd frequencies at or above 3 (Basri et al. 2019), and any residual mass there never
  decays.
- **Drop the two constant zero input coordinates.** They contribute nothing and change the input
  geometry that the kernel's power series depends on (Murray et al. 2023).
- **Centre and scale the maze coordinates** (to the unit square, and separately onto the unit
  circle) and compare the empirical eigenvalue spread.
- **Add layer normalization inside the predictor** and measure the change in the variance of the
  empirical kernel eigenvalues; Cai et al. 2024 report a large reduction in the maximum eigenvalue
  and in the eigenvalue variance.
- **Swap ReLU for the hat activation** (Hong et al. 2022), which is compactly supported and should
  move the kernel towards diagonal.

---

## 10. Diagnostics to run before choosing a method

### 10.1 The decisive one-hour experiment

1. Compute the empirical kernel Gram matrix $K$ on the 100 positions at initialization and
   eigendecompose it.
2. Plot the eigenvalues $\lambda_j$ on a log scale — this reads off how wide the spectrum is, and
   therefore how much of the heterogeneity is condition-B-eigenvalue.
3. Plot, for each position $i$, the mixture weights $w_{ij} = v_{j,i}^2$ against $j$ — this reads off
   how localized the eigenvectors are, and therefore how much of the heterogeneity is
   condition-B-eigenvector. If the $w_{i\cdot}$ curves lie on top of each other, the eigenvectors are
   delocalized and the eigenvalue spread alone is not the problem.
4. Predict each position's residual curve from the boxed formula of Section 2.1 with the observed
   step schedule, and overlay it on the measured $b_i(n)$. **If the prediction matches, the whole
   linearized account applies and Section 9.2 will work. If it does not match, the network is
   leaving the linearized regime and the kernel interventions are the wrong tool** — in which case
   Section 9.3 (measured rates, no kernel) is the fallback that does not care.

### 10.2 Distinguish the two power-law mechanisms

Run the same experiment at $\eta_n = a/n$ for $a$ and $2a$. Under the step-schedule mechanism of
Section 2.3 the exponents should scale linearly with $a$ (slope $-a\lambda$). Under the power-law-
spectrum mechanism of Velikanov & Yarotsky the aggregate exponent $\kappa/(2\nu)$ does not depend on
$a$ at all. Knowing which one is producing the $-0.503$ decides whether calibrating $a$ can change
anything at all.

### 10.3 Report the right quantity

Every plot should say whether it shows the loss (half the squared residual), the aggregate residual
2-norm, or a per-position residual norm — the exponents differ by a factor of two between the first
and the others, and several papers quoted above report the first while our goal is stated in the
third.

---

## 11. What the cited work does not say (so we do not over-claim)

- No paper proves a **per-input-point** decay exponent for gradient descent on a fixed finite point
  set. The per-position formula in Section 2.1 is a direct row-wise reading of Arora et al.'s
  Theorem 4.1, but I did not find it written down anywhere.
- No paper analyses Adam in this framework. The $-0.8$-and-saturate observation has no citation
  behind it.
- Basri et al.'s $O(\kappa^d/p(x))$ is a **time-to-threshold** for a pure single-frequency target
  under a piecewise-constant density on a sphere. It is not a decay exponent, and the maze point set
  is neither a sphere nor piecewise-constant in density.
- Cai et al. (normalization flattens the eigenvalue spread) is an empirical preprint measurement on
  coordinate-based signal fitting, not a theorem.
- Hong et al.'s claim that the hat activation removes spectral bias is a prediction from a
  finite-element analogy plus empirical verification; I did not find an eigenvalue-decay theorem
  for it.
- Bietti & Mairal's $k^{-d}$ eigenvalue decay is quoted here on the authority of Basri et al. 2020
  and Geifman et al. 2024, not from the paper's own text.
- The natural-gradient and preconditioned-gradient theorems both assume the network stays near its
  initialization (their Condition 2 / assumptions 1–4). With a 256-unit hidden layer and 100 points
  this is plausible but must be checked, which is what diagnostic 10.1 step 4 does.

---

## 12. References

All of the following were confirmed to exist at the listed identifier during this literature search.

1. Jacot, A., Gabriel, F., Hongler, C. "Neural Tangent Kernel: Convergence and Generalization in
   Neural Networks." NeurIPS 2018. arXiv:1806.07572.
   <https://arxiv.org/abs/1806.07572>
2. Arora, S., Du, S. S., Hu, W., Li, Z., Wang, R. "Fine-Grained Analysis of Optimization and
   Generalization for Overparameterized Two-Layer Neural Networks." ICML 2019. arXiv:1901.08584.
   <https://arxiv.org/abs/1901.08584>
3. Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y.,
   Courville, A. "On the Spectral Bias of Neural Networks." ICML 2019. arXiv:1806.08734.
   <https://arxiv.org/abs/1806.08734>
4. Basri, R., Jacobs, D., Kasten, Y., Kritchman, S. "The Convergence Rate of Neural Networks for
   Learned Functions of Different Frequencies." NeurIPS 2019. arXiv:1906.00425.
   <https://arxiv.org/abs/1906.00425>
5. Basri, R., Galun, M., Geifman, A., Jacobs, D., Kasten, Y., Kritchman, S. "Frequency Bias in
   Neural Networks for Input of Non-Uniform Density." ICML 2020, PMLR 119:685–694. arXiv:2003.04560.
   <https://arxiv.org/abs/2003.04560>
6. Bietti, A., Mairal, J. "On the Inductive Bias of Neural Tangent Kernels." NeurIPS 2019.
   arXiv:1905.12173. <https://arxiv.org/abs/1905.12173>
7. Cao, Y., Fang, Z., Wu, Y., Zhou, D.-X., Gu, Q. "Towards Understanding the Spectral Bias of Deep
   Learning." IJCAI 2021. arXiv:1912.01198. <https://arxiv.org/abs/1912.01198>
8. Yang, G., Salman, H. "A Fine-Grained Spectral Perspective on Neural Networks." arXiv:1907.10599.
   <https://arxiv.org/abs/1907.10599>
9. Murray, M., Jin, H., Bowman, B., Montúfar, G. "Characterizing the spectrum of the NTK via a
   power series expansion." ICLR 2023. arXiv:2211.07844. <https://arxiv.org/abs/2211.07844>
10. Bordelon, B., Canatar, A., Pehlevan, C. "Spectrum Dependent Learning Curves in Kernel Regression
    and Wide Neural Networks." ICML 2020, PMLR 119:1024–1034. arXiv:2002.02561.
    <https://proceedings.mlr.press/v119/bordelon20a.html>
11. Velikanov, M., Yarotsky, D. "Universal scaling laws in the gradient descent training of neural
    networks." arXiv:2105.00507. <https://arxiv.org/abs/2105.00507>
    (NeurIPS 2021 version: "Explicit loss asymptotics in the gradient descent training of neural
    networks.")
12. Bahri, Y., Dyer, E., Kaplan, J., Lee, J., Sharma, U. "Explaining Neural Scaling Laws."
    PNAS 121, 2024. arXiv:2102.06701. <https://arxiv.org/abs/2102.06701>
13. Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U.,
    Ramamoorthi, R., Barron, J. T., Ng, R. "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains." NeurIPS 2020. arXiv:2006.10739.
    <https://arxiv.org/abs/2006.10739>
14. Wang, S., Wang, H., Perdikaris, P. "On the eigenvector bias of Fourier feature networks: From
    regression to solving multi-scale PDEs with physics-informed neural networks." Computer Methods
    in Applied Mechanics and Engineering 384:113938, 2021. arXiv:2012.10047.
    <https://arxiv.org/abs/2012.10047>
15. Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., Wetzstein, G. "Implicit Neural
    Representations with Periodic Activation Functions." NeurIPS 2020. arXiv:2006.09661.
    <https://arxiv.org/abs/2006.09661>
16. Hong, Q., Siegel, J. W., Tan, Q., Xu, J. "On the Activation Function Dependence of the Spectral
    Bias of Neural Networks." arXiv:2208.04924. <https://arxiv.org/abs/2208.04924>
17. Cai, Z., Zhu, H., Shen, Q., Wang, X., Cao, X. "Towards the Spectral bias Alleviation by
    Normalizations in Coordinate Networks." arXiv:2407.17834. <https://arxiv.org/abs/2407.17834>
18. Geifman, A., Barzilai, D., Basri, R., Galun, M. "Controlling the Inductive Bias of Wide Neural
    Networks by Modifying the Kernel's Spectrum." Transactions on Machine Learning Research,
    February 2024. arXiv:2307.14531. <https://arxiv.org/abs/2307.14531>
19. Zhang, G., Martens, J., Grosse, R. "Fast Convergence of Natural Gradient Descent for
    Over-Parameterized Neural Networks." NeurIPS 2019. arXiv:1905.10961.
    <https://arxiv.org/abs/1905.10961>
20. Chen, W., Howard, A. A., Stinis, P. "Self-adaptive weights based on balanced residual decay rate
    for physics-informed neural networks and deep operator networks." Journal of Computational
    Physics, 2025. arXiv:2407.01613. <https://arxiv.org/abs/2407.01613>
21. Wang, S., Yu, X., Perdikaris, P. "When and why PINNs fail to train: A neural tangent kernel
    perspective." Journal of Computational Physics 449:110768, 2021. arXiv:2007.14527.
    <https://arxiv.org/abs/2007.14527>
22. Osband, I., Aslanides, J., Cassirer, A. "Randomized Prior Functions for Deep Reinforcement
    Learning." NeurIPS 2018. arXiv:1806.03335. <https://arxiv.org/abs/1806.03335>

# Novelty check, first draft (2026-11-09 PT)

Recall from `notes.md`: $s$ is a state, $n(s)$ its number of predictor updates, $r(s)$ the residual between predictor
and frozen random target, and $\rho(s) = \lVert r(s) \rVert$ the bonus (written $b(s)$ in `notes.md`). The goal is $\rho(s) = 1$ before training at
every state and $\rho(s) \propto n(s)^{-1/2}$ at every state, with no stored counts. The papers are listed in
`sources.md`; per-paper details are in `notes.md`.

## Main findings

1. **The problem is published; the claim must be about the mechanism.** DRND (Yang et al. 2024) names both halves
   ("initial bonus inconsistency" and "final bonus inconsistency"), and the coin-flip network (Lobel et al. 2023) and
   Random Distribution Distillation (Fang et al. 2025) both state that RND's rate of decay with visits "depends on the
   learning dynamics". All three reach an approximately even start and a $n^{-1/2}$ decay by regressing onto random
   targets (fresh coin flips, one of $N$ networks, or Gaussian draws). What none of them does is get the rate from a
   fixed, deterministic random target through the shape of the training rule.
2. **The shrink-target map is the elliptical bonus's own update.** Mapping $\rho \mapsto \rho/\sqrt{1+\rho^{2}}$ at
   the visited state is exactly the rank-one update of the elliptical bonus $\sqrt{\phi^{\top}\Lambda^{-1}\phi}$ used
   by E3B, by anti-concentrated bonuses and by linear bandits (derivation 1 below). With one-hot inputs and no
   interference it gives exactly $1/\sqrt{1+n(s)}$, the tabular count bonus. So the bonus values are not new; what
   was not found is storing that bonus as an RND residual through moved regression targets, with no matrix and no
   count. The nearest learned-counter precedent is DORA's E-values (Fox et al. 2018), which shrink by a constant
   factor per visit (geometric) and need a logarithm to read a count.
3. **The even start has published parts.** Machado, Srinivasan and Bowling (2014) state that an equal optimistic
   start for a linear value function needs constant-length features, and get it from all-zero weights. A zeroed
   output layer giving the same starting value at every input is in DORA (E-values start at 0.5); length-normalised
   targets in a prediction-error bonus are in BYOL-Explore (Guo et al. 2022); the coin-flip network scales its prior
   so the average squared output is 1. The exact combination (zero predictor head, frozen random target divided by
   its own length, inside RND) was not found, but it is a small step from these.
4. **No published power-law or non-squared distillation loss for an exploration bonus was found** in the papers read:
   every RND variant here uses squared error (BYOL-Explore uses squared error between normalised vectors). The fact
   that gradient descent on a quartic gives $t^{-1/2}$ is elementary; its use to set a per-state bonus exponent was
   not found. It is the first-order version of the shrink-target map (derivation 2).
5. **Input-layer-only training on one-hot inputs makes the predictor a per-state table.** The one-hot case is analysed
   in published work only with linear models (coin-flip Appendix B; DRND's toy). Training only the one-hot layer of a
   nonlinear RND predictor was not found, but a reviewer will read it as a tabular method; the novelty claim should
   rest on the training rule (power-law or shrink-target), with this block presented as the setting that removes
   interference.

## Table: closest published work per building block

| block | closest published work found | what it does | how ours differs |
|---|---|---|---|
| zero head with unit target | Machado et al. 2014; DORA (Fox et al. 2018, Section 4.1); BYOL-Explore (Guo et al. 2022); coin-flip network prior (Lobel et al. 2023, Section 3.5.2); Zanger et al. 2026, Proposition 4.1; DRND (Yang et al. 2024) for the problem statement | Machado et al. get the same optimistic start at every state of a linear value function from zero weights, and note that this needs constant-length features. DORA zeroes the last layer of the E-value network under a logistic output, so every input starts at 0.5. BYOL-Explore divides both prediction and moving target by their lengths (cosine-type loss). The coin-flip prior is scaled so the average over the data of each squared output is 1. Zanger et al. zero the final layer of the target, not the predictor, to make the error a posterior. DRND narrows the uneven start by averaging 10 targets. | Each half is published separately. Not found: zero predictor head together with a frozen random target divided by its own length at each state, inside RND, so the start is exactly 1 at every state. Small novelty; present it as a construction, not a contribution. |
| initial-ratio readout | DRND second bonus term; Random Distribution Distillation statistic; Never Give Up lifelong multiplier; learning progress monitoring (Hou et al. 2026) | DRND divides by the per-state variance of its 10 targets; Random Distribution Distillation divides by the target variance (a constant in its deployed bonus); Never Give Up subtracts a running mean and divides by a running standard deviation pooled over all states. Hou et al. subtract the current error from a learned prediction of the error at the previous update. | Not found: dividing by the realised error of the untrained predictor at the same state (also not found by August note 04, Section 4.1). See the interaction note: this readout conflicts with the power-law and shrink-target losses. |
| power-law loss | none found in exploration; nearest are the squared-error RND variants and Ostrovski et al. 2017 | Every RND variant read uses squared error. In the coin-flip paper (Appendix E.2), fixed zero targets with the random prior leave states visited more than once with almost no bonus. Ostrovski et al. force the rate by multiplying prediction gain by a global $n^{-1/2}$ schedule in the readout. | Not found. Ours sets the exponent per state through the loss power during training, with no step count in the readout. The constant still depends on each state's curvature (derivation 2). |
| shrink-target loss | elliptical bonus update (E3B, Henaff et al. 2022; Ash et al. 2022; linear bandits); DORA E-values (Fox et al. 2018) | The elliptical bonus at a visited input follows exactly the same per-visit map (derivation 1). DORA regresses a learned value toward a constant fraction of itself each visit and reads a count through a logarithm. | Same bonus values as the elliptical or count bonus, so the values are not new. Not found: producing them as an RND residual against a frozen random target by regressing toward the predictor's own moved output, with no stored matrix, count or log readout. |
| input-layer-only training | coin-flip network Appendix B; DRND toy experiment; E3B one-hot remark; Guei et al. 2021; Rashid et al. 2020 | The first three note that with one-hot inputs a linear model or an elliptical bonus reduces to tabular counts. Guei et al. initialise the weights of a table-indexed value network very high so that visited entries shrink and rare ones stay high. Rashid et al. state that an optimistic network start does not stay optimistic for unvisited pairs because of generalisation, and move the optimism into explicit counts. | Not found for a nonlinear RND predictor trained only in its one-hot input layer. Honest reading: it makes the predictor a per-state table of embeddings, which removes interference between states (the condition the power-law loss needs) and also removes generalisation to unvisited states. |
| AdaGrad or 1/t predictor | PreND (Davoodabadi et al. 2024); Bellemare et al. 2016; Ostrovski et al. 2017; Even-Dar and Mansour 2003; Duchi et al. 2011; August note 02, Section 4.4 | PreND slows RND's "rapid degradation of intrinsic rewards" by giving the predictor a learning rate 0.01 times smaller. Even-Dar and Mansour analyse per-pair steps of one over a power of the visit count in tabular Q-learning. Bellemare et al. tie count recovery to the density model's step size (a $1/n$ step gives the empirical distribution). Ostrovski et al. kept PixelCNN's learning rate constant (it gave the lowest loss) and moved the decay schedule into the readout. AdaGrad gives rare features larger steps. August note 02: AdaGrad on a fixed-target squared error converges geometrically, not as a power law. | Not found as an RND design: AdaGrad on a one-hot input layer gives each state its own step from its own gradient history, an implicit per-state schedule. With squared error it does not give a power law; a global 1/t ties the decay to global time, not to $n(s)$. |
| whole method | coin-flip network; DRND; Random Distribution Distillation; DORA | Same two goals, reached by random targets or by a learned counter with a log readout. | Deterministic frozen target; the rate comes from the training rule. Not found as a whole. |

## Derivations used in the table

These are algebra only (no numerical runs).

**Derivation 1: the shrink-target map equals the elliptical bonus update.** Define the feature vector $\phi$ of the
visited state, a ridge constant $\lambda > 0$, the matrix $\Lambda_n = \lambda I + \sum_{m=1}^{n} \phi_m \phi_m^{\top}$
over the first $n$ visited feature vectors $\phi_m$, and the squared elliptical bonus $u = \phi^{\top}\Lambda_n^{-1}\phi$
at the state about to be visited. Adding $\phi\phi^{\top}$ and applying the Sherman–Morrison formula gives

$$\phi^{\top}\Lambda_{n+1}^{-1}\phi = u - \dfrac{u^{2}}{1+u} = \dfrac{u}{1+u},$$

so the bonus $\sqrt{u}$ maps to $\sqrt{u}/\sqrt{1+u}$, the shrink-target map. With one-hot $\phi$, $\Lambda_n$ is
diagonal with entry $\lambda + n(s)$ for state $s$, the bonus is $1/\sqrt{\lambda + n(s)}$, and the update at $s$
leaves every other state unchanged. With $\lambda = 1$ the start is 1 and the bonus is $1/\sqrt{1+n(s)}$. The same
recursion written for the residual is $\rho_n^{-2} = \rho_0^{-2} + n$.

**Derivation 2: the power-law loss is the first-order version.** Define $\eta$ as the step size and $\kappa(s)$ as the
change in $g(s)$ produced by a unit gradient step taken on $s$, assuming for now that the tangent kernel at $s$ acts
on the output as $\kappa(s)$ times the identity. With no interference, one step on $\frac{1}{4}\lVert r \rVert^{4}$
(gradient $\lVert r \rVert^{2} r$ in output space) changes the residual length by
$\rho \mapsto \rho \thinspace (1 - \eta\kappa\rho^{2})$, so $\rho^{-2}$ grows by about $2\eta\kappa$ per update and

$$\rho_n \approx \left(\rho_0^{-2} + 2\eta\kappa(s) \thinspace n\right)^{-1/2}.$$

With $\eta\kappa = 1/2$ this matches the shrink-target map to first order in $\rho^{2}$. For a general power $q$, the
same argument gives $\rho_n \propto n^{-1/(q-2)}$, so $q = 4$ gives the exponent $-1/2$ at every state, while the
constant depends on $\kappa(s)$. When the kernel at $s$ is not a multiple of the identity (for input-layer-only
training it is the product of the frozen upper layers' Jacobian with its transpose, which differs between states),
the same equations written per eigen-direction give: the component on the smallest eigenvalue $\kappa_{\min}(s)$
decays as $(2\eta\kappa_{\min}(s) n)^{-1/2}$ and every other component decays faster, so the exponent stays $-1/2$ and
the constant is set by $\kappa_{\min}(s)$. The shrink-target loss sets the change in output space directly, so its
constant does not depend on the kernel as long as the regression reaches its moved target. This matches August note
05 (method M1) and August note 06 (method M3), which also measured that plain gradient descent on the quartic loss
diverged on the coordinate-input maze network without a preconditioner.

**Interaction note: initial-ratio readout with these two losses.** Both recursions forget the starting value:
$\rho_n \approx n^{-1/2}$ (shrink target) or $(2\eta\kappa n)^{-1/2}$ (power law) for large $n$, whatever $\rho_0$ was.
Dividing by $\rho_0(s)$ then gives an even start but a late bonus of about $n^{-1/2}/\rho_0(s)$, uneven by the
spread of the starting values. With these losses the even start has to come from the zero head with unit target
($\rho_0 = 1$ exactly), not from the readout.

**AdaGrad reading.** With the power-law loss the gradient on a state's input column is proportional to $\rho^{3}$; if
$\rho \propto n^{-1/2}$, the sum of squared gradients converges, AdaGrad's step settles to a constant, and the
$n^{-1/2}$ law is kept, with a constant set by the state's early history. With squared error the sum also converges
and the decay is geometric (August note 02, Section 4.4). This is a reading of the formulas, not a measurement.

## Other candidates seen in the extra search (abstract pages only, not downloaded)

The extra search used about 45 queries (general web search, the arXiv listing, OpenReview search, and the lists of
papers citing the coin-flip network and DRND). These were judged further from the blocks than the downloaded ones:

- Martin, Sasikumar, Everitt, Hutter, "Count-Based Exploration in Feature Space for Reinforcement Learning", IJCAI
  2017 (arXiv 1706.08090): pseudo-counts from feature visit frequencies, bonus by a hand-written inverse square root.
- Khan, Nielsen, Tangkaratt, Lin, Gal, Srivastava, "Fast and Scalable Bayesian Deep Learning by Weight-Perturbation
  in Adam", ICML 2018 (arXiv 1806.04854): uncertainty from Adam's per-weight scaling vector; the reading that on
  one-hot inputs it acts as a visit counter is ours, not the paper's.
- Lobel, Gottesman, Allen, Bagaria, Konidaris, "Optimistic Initialization for Exploration in Continuous Control",
  AAAI 2022: values kept optimistic by distance to visited data, not by count.
- Chandak et al., "Representations and Exploration for Deep Reinforcement Learning using Singular Value
  Decomposition", ICML 2023 (arXiv 2305.00654): pseudo-counts read from learned representations.
- Zanger et al., "Universal Value-Function Uncertainties", ICLR 2026 (arXiv 2505.21119): RND-style error with a
  temporal-difference-trained network.

## Not found, after the searches recorded in `sources.md` and above

- A paper dividing an RND bonus by the untrained predictor's error at the same state.
- A paper training an exploration-bonus predictor with a non-squared power of the residual to set the decay
  exponent.
- A paper regressing an RND predictor toward its own residual shrunk by a residual-dependent factor.
- A paper training only the one-hot input layer of a nonlinear RND predictor.

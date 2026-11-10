# Diagnosis notes

Times are Pacific. Scripts: `diagnosis/frontier_profile.py`, `diagnosis/count_nonfinite.py`.

## 1. The power-law loss diverges inside RL (found 2026-11-09 22:25)

### What happened

- The first full candidate, unit start + power-law loss q = 4 + plain SGD 0.3 + length readout, scored 709K on the
  snake grid with coordinate-PPO, with the goal found by 84% of seeds. The power-law loss alone (SGD 0.1) scored 60K.
- The candidate's bonus is not-a-number from about 65K transitions on (`frontier_profile.py`: the median over copies
  is finite at 16,384 transitions and not-a-number at 65,536).
- Count of copies whose bonus became non-finite, in the units finished by 22:25 (`count_nonfinite.py`):

| idea | case | non-finite copies | earliest checkpoint |
|---|---|---|---|
| unit start + power-law q = 4, SGD 0.3 | snake, coordinate-PPO | 53 and 59 of 224 (two entropies) | 4,096 |
| power-law q = 4, SGD 0.1 | straight, coordinate-PPO | 29 and 28 of 224 (4 of 32 seeds at every weight: one network draw of the eight) | 1,024 |
| power-law q = 4, SGD 0.1 | snake, coordinate-PPO | 2 and 0 of 224 | 65,536 |
| shrink target, SGD 1 | straight, coordinate-PPO | 1 and 0 of 224 | 8,192 |
| unit start + shrink target, SGD 3 | straight, tabular-PPO | 0 of 448 | none |
| control, unit start, separate Adam, bias start | every finished case | 0 | none |

### Mechanism

Define $r$ as the target minus the predictor output of one cell, $\rho = \lVert r \rVert$, $\eta$ as the SGD learning
rate, $\kappa$ as the size of the predictor's own-cell tangent kernel, and $\phi$ as the fraction of a minibatch's kept
samples that are this cell. One SGD step on the minibatch-mean loss moves the cell's residual to

$$r' = r\thinspace\bigl(1 - \eta\thinspace\kappa\thinspace\phi\thinspace s(\rho)\bigr)$$

where $s(\rho)$ is the loss's own step factor.

- **Power-law loss, q = 4:** $s(\rho) = 2\rho^2$. The step grows with the square of the error, so it has no upper
  limit. The residual overshoots once $\eta\kappa\phi \cdot 2\rho^2 > 2$, and an overshoot makes $\rho$ larger, so
  the next step is larger still.
- **Shrink-target loss:** $s(\rho) = 1 - 1/\sqrt{1+\rho^2}$, which lies between 0 and 1 for every $\rho$. It is the
  same law for small errors ($s \approx \rho^2/2$) with a bounded step.
- **Why the bench did not show it:** on the bench every cell is a small share of each minibatch ($\phi$ about 1/16
  to 1/32). In RL the policy sits in a few cells at the start, so $\phi$ is close to 1 and the step is 16 to 32 times
  larger. A network draw with a large starting error diverges at the first checkpoint (the straight-grid row above).

### Verdict

A property of the loss, not a code defect. The power-law form is dropped as the method's loss. Its finished rows are
reported with their non-finite counts and are not evidence about the two properties.

### Fix

- **The loss:** the shrink-target form is the method's loss.
- **Other exponents:** the shrink map is generalized to $\rho \to \rho\thinspace(1+\rho^{p})^{-1/p}$, which keeps the
  step factor between 0 and 1 for every $p$ and gives $\rho_n = (\rho_0^{-p} + n)^{-1/p}$: exponent $-1/p$, so
  $p = 2$ is $-1/2$, $p = 1$ is $-1$ and $p = 4$ is $-1/4$.
- **Bench:** it gains a visitation law in which the predictor trains on rollouts that sit in one or two cells, as
  early RL does, so this failure shows before RL units are sent.
- **Report:** the screen tables list each idea's copies with a non-finite bonus.
- **Queue:** 8 power-law units that had not started are held in `queue/deferred_unstable_power_law/`.

### Gate for the fix

No copy of any shrink-target idea may have a non-finite bonus in RL, in all four cases; checked with
`count_nonfinite.py` as units finish.

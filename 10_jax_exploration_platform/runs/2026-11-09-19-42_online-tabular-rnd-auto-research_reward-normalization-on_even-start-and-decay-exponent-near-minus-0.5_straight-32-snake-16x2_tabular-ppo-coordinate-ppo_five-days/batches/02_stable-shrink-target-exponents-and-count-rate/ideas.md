# Batch 2: what the count rule loses under an RND bonus's handicaps, and the stable shrink-target family

Written 2026-11-09 23:12 PT, before any unit of the batch had finished. Times are Pacific. Exact settings:
`rl_ideas.json`. Every unit runs with CleanRL's reward normalization on, at policy learning rates 1e-3 and 1e-4,
entropy 0.001 and 0.01, and weights 2^-6 to 2^6 in steps of 4×, on seeds 0-31, stopped at 2^22 transitions.

## Why this batch

Batch 1 (partial, 23:00 PT) gave two facts:
1. **The power-law loss is unstable in RL** (`diagnosis/diagnosis.md`). The shrink-target loss is the stable form.
2. **The stable full candidate is not ahead yet.** On the straight grid with tabular-PPO, even start + shrink target
   (SGD 3) scores 14.9K. The fast-decay even-start variants score 8.0K to 8.6K, and the original study's uniform count
   field scored 8.1K (128 seeds). The candidate has both properties in RL (start 1.00, slope −0.53) and still trails
   the count rule it imitates.

So something other than the two properties separates an RND bonus from the count rule. Two structural differences
are known:
- **Update timing:** an RND bonus is fixed during a rollout of 128 steps and changes only at the update, while the
  count rule's bonus falls at every arrival.
- **Count rate:** the shrink-target predictor counts each visit as about a quarter to a half of a visit (bench).

## Part 1: the uniform count field with those handicaps (10 diagnostic rows)

These are count-rule runs, not RND variants. Define $m$ as a cell's earlier visits and $c$ as the count rate. The
bonus is $(1 + c\thinspace m)^{-a}$ with $a = 0.5$ unless named.

| row | earlier visits $m$ counted | $c$ | $a$ | question |
|---|---|---|---|---|
| count rule | before this arrival (per step) | 1 | 0.5 | reference; equals the original uniform field |
| per rollout | before this rollout | 1 | 0.5 | does fixing the bonus during a rollout cost performance? |
| per step, rate 0.5 and 0.25 | per step | 0.5, 0.25 | 0.5 | does a slower count cost performance? |
| per rollout, rate 0.5 and 0.25 | per rollout | 0.5, 0.25 | 0.5 | both handicaps: the best a shrink-target RND bonus could do |
| per rollout, rate 4 | per rollout | 4 | 0.5 | does a faster count repair the per-rollout form? |
| per rollout, exponents −1, −2, −0.25 | per rollout | 1 | 1, 2, 0.25 | which exponent is best once the bonus is fixed during a rollout? |

Predictions:
- If "per rollout" is clearly worse than the count rule, the update timing is the main loss, and the method needs a
  correction for visits inside the rollout.
- If "per rollout, rate 0.5" matches the batch 1 candidate, the candidate already does what its form allows, and the
  remaining gap to the count rule is the form.
- If the per-rollout rows prefer a faster exponent (−1 or −2), that explains why fast-decay variants do well on the
  straight grid.

## Part 2: the stable shrink-target family (4 RND ideas)

| idea | bench: slope, straight / snake | bench: non-finite under one-cell rollouts | question |
|---|---|---|---|
| unit start + shrink target, SGD 1 | −0.46 / −0.41 | 0 of 16 | the safest learning rate (batch 1 runs SGD 2 and 3) |
| unit start + shrink exponent −1, SGD 1 | −0.93 / −0.86 | 0 of 16 | is −0.5 better than −1 for an RND bonus? |
| unit start + shrink exponent −0.25, SGD 4 | −0.24 / −0.22 | 0 of 16 | is −0.5 better than −0.25? |
| unit start + exact-step shrink target (no learning rate) | −0.31 / −0.25 | 0 of 16 | stable by construction; it under-counts when a minibatch holds many cells |

The exact-step form is not yet at −0.5 on the bench with one step per minibatch (−0.33 with two). It is included
because it cannot run away at any setting.

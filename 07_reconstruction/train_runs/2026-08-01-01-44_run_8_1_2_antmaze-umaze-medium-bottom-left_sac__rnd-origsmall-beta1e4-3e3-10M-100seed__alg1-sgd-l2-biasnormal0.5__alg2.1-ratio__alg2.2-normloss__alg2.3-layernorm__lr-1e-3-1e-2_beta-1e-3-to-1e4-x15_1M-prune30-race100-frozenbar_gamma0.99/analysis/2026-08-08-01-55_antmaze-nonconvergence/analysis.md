# Why the AntMaze runs do not converge

Snapshot 2026-08-08. One question: SAC + RND (and the four train-run-1.2 algorithms) never
reach goal-reaching performance on AntMaze UMaze / Medium — why?

**One-paragraph answer.** The runs do not fail to explore — they fail to KEEP a goal-reaching
policy, and mostly never build one. Reaching the goal takes so long on AntMaze (median 462 of
the 700-step cap on UMaze, 752 of 1000 on Medium) that at discount $\gamma=0.99$ a success
improves the start-state value by under 1% of the return scale, while the intrinsic bonus the
agent actually optimizes never decays away. The successes that do happen are exploration
by-products (their steps-to-goal never shortens over training), and on the 10⁷-step runs they
are eventually evicted from the 10⁶-transition replay buffer, after which the success rate
decays to exactly zero while maze coverage sits at 100%.

## 1. Data and denominators

One-pass extraction (`code/extract_summaries.py`) over the three sweeps' completed per-run
records; every number below re-derives from `data/summaries.jsonl.gz` via `code/make_numbers.py`.

| source | records kept (completed) | of files seen |
|---|---|---|
| train run 1.1 (8 envs × 5 arms × 15 weights, 1M steps) | 14,488 | 15,474 |
| train run 1.2 (4 arms task S at 1M + RND task R at 10M) | 6,254 | 8,717 |
| lr addendum (RND at Adam 1e-3 / 1e-2, 1M steps) | 2,297 | 3,185 |

Verification gates (all pass; `make_numbers.py` hard-fails otherwise):
run-1.1's published last-100 success rates reproduce (0.0619 vs 0.061 UMaze at β=10⁴ n=84;
0.0152 vs 0.015 Medium at β=3×10³ n=82); the task-R whole-run rewards reproduce to the cent
(−695.29 / −996.75); the reward identity (success at step $k$ earns $-(k-1)$, a timeout earns
$-T$) holds with 0 violations across all 20,742 shift(−1) records.

Notation used throughout: define $T$ as the episode cap, $k$ as the step at which a successful
episode terminates, $\gamma=0.99$ the discount, $E$ an episode's extrinsic return, $I$ its
$\beta$-scaled intrinsic sum, and the success rate per block as successful episodes over
episodes in that block.

## 2. The phenomenon: find-then-forget at 10M, never-learn at 1M

Figure `plots/f1_phenomenon.png`. Task-R baseline (the run-1.1 winner per env, 10⁷ steps):

- UMaze (n=17): per-1M-block median success 0.033, 0.059, 0.028, 0.004, 0.002, 0.0007, then
  exactly 0 in blocks 7–10. Median last-success step 5.56M. Every seed succeeds at least once.
- Medium (n=20): rises to ~0.02 by block 4–5, exactly 0 from block 8. Median last success 8.76M.
- Maze-cell coverage reaches 100% at a median of 0.05M steps (UMaze) / 0.45M (Medium) and stays
  there — the top panels decay while the blue lines are flat at 100.
- The 10⁷ run's final reward (−695.29) is WORSE than the same configuration's own 10⁶-step
  value in run 1.1 (−689.85): the second 9M steps subtracted performance.

At 1M steps (every sweep arm): median whole-run success ≈ 0 for every configuration of every
arm; the best single episodes (−46..−148) show the goal is touched but never becomes policy.

## 3. Mechanisms, ranked by evidence

### 3.1 Credit assignment: a success is worth <1% of the return scale (Figure `plots/f4_credit.png`)

$\Delta_\gamma = (\gamma^{k} - \gamma^{T})/(1-\gamma)$ — the discounted start-state advantage
of succeeding at step $k$ over timing out — against a return scale of
$(1-\gamma^{T})/(1-\gamma) \approx 100$:

| environment | median $k$ | cap $T$ | $\Delta_\gamma$ | share of return scale |
|---|---|---|---|---|
| PointMaze UMaze (converges) | 105 | 300 | **29.9** | **31%** |
| AntMaze UMaze | 462 | 700 | 0.87 | 0.9% |
| AntMaze Medium | 752 | 1000 | 0.048 | 0.05% |

Same code, same reward shift, same $\gamma$: the signal a success carries is 34–620× weaker on
AntMaze than on the PointMaze control that converges. This is the cleanest single explanation,
and it also orders the two AntMaze environments correctly (Medium ≪ UMaze in success).

### 3.2 The objective never becomes the task (Figure `plots/f3_objective.png`)

Median per-block $\sum|\beta I| / \sum|E|$ on the task-R baseline (reward normalization ON):
UMaze 3.3 → 1.7 across ten 1M blocks; Medium ≈ 1.0 throughout. After 10⁷ steps — with coverage
at 100% since 0.5M — the bonus still matches or outweighs the task. On the reward-norm-OFF
task-S arms the ratio at the winning weights is 10–10³. A fixed goal-reaching policy is not a
fixed point of this objective. (All ratios are medians over episodes: the first episodes of a
reward-norm-ON run have ratios ~10³ — a warm-up artifact that would dominate a mean.)

### 3.3 Successes are stumbled upon, not learned

Task-R UMaze steps-to-goal: first-half median 488, second-half 436 — essentially flat, never
approaching the ~191 its own best episodes (p10) demonstrate is possible, and LATER than the
~350 a memoryless constant-hazard reference predicts. The success-rate collapse happens with no
change in $k$. These are exploration by-products, not an emerging policy.

### 3.4 Replay eviction locks the forgetting in (consequence, not trigger)

`buffer_size` = 10⁶ (SB3 default, never overridden) on 10⁷-step runs. The success rate halves
while the buffer still holds ~41 successful episodes — so eviction does not start the decay
(3.1–3.3 do) — but once the last success leaves the buffer, no gradient update ever sees a goal
transition again; the extrinsic Q-target becomes a constant −1/step and recovery is impossible.

### 3.5 Terminal-set geometry, and episodes burned by a fallen ant

The 0.45 m goal radius is hard-coded while AntMaze cells are 4 m (`maze_size_scaling=4` vs
PointMaze's 1): the goal ball covers ≈4% of a cell vs ≈64%. And `ant_maze_v5.py:294` discards
the inner Ant's termination flag, so a flipped ant runs out the 700/1000-step clock at −1/step
with no reset. (No trajectories are logged, so the flip frequency is not measurable after the
fact.)

### 3.6 This is the known-hard setting, run without the ingredients that solve it

ExPLORe — the paper the environment choice follows — runs RLPD: SAC with a 50/50 replay mix of
OFFLINE goal-reaching prior data and 20 gradient steps per environment step. These runs: SB3
SAC from scratch, one environment, 1 gradient step per step, no prior data.

## 4. What the failure is NOT

- **Not exploration.** Coverage 100% from 0.05M/0.45M steps while success decays to zero.
- **Not RND bonus quality.** RND is the BEST explorer in run 1.1 (winner coverage 100%); the
  oracle count arms are WORSE explorers there (winner-table coverage 64–69% — their bonus
  saturates as counts grow) and score exactly zero success. No bonus in the sweep converts
  coverage into policy.
- **Not the step budget.** 10× more steps made the final policy worse.
- **Not the predictor learning rate, though it helps.** Addendum best configs (whole-run
  success / median $k$): UMaze 0.144 / 406 at Adam 10⁻² vs 0.066 / 476 at 10⁻³ vs run-1.1's
  0.061 / ~449 at 10⁻⁴; Medium 0.033 / 696 vs 0.013 / 780. Monotone in the rate on every
  metric — consistent with a faster predictor acting as an implicit bonus anneal — but 14%
  success at $k≈400$ of a 700 cap is still not convergence.

## 5. Levers, ranked by the mechanism they target

1. $\gamma \to 0.999$ (or n-step returns): moves UMaze's $\Delta_\gamma$ share from ~0.9% to
   ~55% at the observed $k$ [3.1];
2. a bonus that anneals, or smaller $\beta$ at the faster predictor rate [3.2];
3. `buffer_size` ≥ total steps on long runs (a 10⁷ Ant buffer is ~5–6 GB) [3.4];
4. gradient steps per env step > 1 (the RLPD ingredient; ×4–20 compute) [3.6];
5. deterministic standalone evaluation to separate the exploration policy from what the critic
   could exploit (diagnostic);
6. log trajectories or visit maps so 3.5's flip cost becomes measurable.

## 6. Reproduction

```
cd code
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python extract_summaries.py --jobs 8
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python make_numbers.py
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python make_figures.py
```
Extraction ≈ 5 minutes at 8 workers; everything downstream reads `data/summaries.jsonl.gz`.

# Interim comparison of the five arms, read at the halfway point

Written 2026-08-10, with the campaign at 40.8% of its 300,000,000,000-step budget and no run
finished. Regenerate with:

```bash
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python \
  analysis/code/make_interim_table_and_plot.py --sweep_dir <this run folder>
```

## The measurement, and why it is fixed to a step

No run has a final score, so nothing here can use a run's last logged row. A run's last step reflects
the speed of the node it happened to land on as much as its arm: the fastest node in the campaign
delivers about 3,900 steps per second and the slowest about 2,500, so after four days the same arm
spans a wide range of depths. Reading arms at different depths would rank node allocations rather
than algorithms.

Both artifacts therefore fix the step first. The table reads **1,000,000,000 steps**, half of the
2,000,000,000 planned. Logging is every 200 policy updates, or 3,276,800 steps, so no row falls
exactly on 1e9 and the row used is the last at or before it, 999,424,000. When the campaign finishes
this becomes the full 2,000,000,000 and nothing else changes.

The quantity is `train/mean_extrinsic_reward` — the trailing mean over each run's last 200 finished
episodes, which is what the trainer logs and the only performance number it records.

## The one number, at 1e9 steps

| arm | seeds | mean ± s.e. | median | worst seed | best seed |
|---|---|---|---|---|---|
| Arm 1 original | 19 | 6682.7 ± 758.9 | 6852.5 | 400.0 | 13494.0 |
| Arm 2 no RND clip | 17 | 6799.0 ± 618.8 | 6852.0 | 1773.0 | 10951.5 |
| Arm 3 full batch | 18 | **7419.9 ± 725.3** | 7080.0 | 469.0 | 14491.0 |
| Arm 4 shallower | 22 | 5886.4 ± 581.6 | 6385.5 | 400.0 | 11230.5 |
| Arm 5 all three | 19 | <u>7210.6 ± 533.4</u> | 7053.0 | **4584.0** | 12350.0 |
| CleanRL published, 1 seed | 1 | 4759.0 | — | — | — |

Bold marks the best value in a column and underline the second best. The seeds column counts the runs
that had reached 1e9 at the time of reading; it differs across arms only because of node speed, which
is independent of the arm.

## The published reference

CleanRL's own run of this file is drawn beside the arms and tabulated below them: their benchmark run
`openrlbenchmark/cleanrl/1wm98fjm` (`MontezumaRevenge-v5__ppo_rnd__1__1657607857`), one seed, pulled
from wandb into `data/reference/cleanrl_ppo_rnd_montezuma.json`. Its configuration is the one this
campaign reproduces, checked field by field: 2,000,000,000 total steps, 128 environments, 128 rollout
steps, update proportion 0.25, max gradient norm 0.5, learning rate 1e-4, intrinsic and extrinsic
coefficients 1 and 2.

**Their per-episode returns are re-averaged over a trailing 200 episodes here.** They log a trailing
mean over 20; this project logs 200. Without re-averaging, the reference would look noisier than the
arms for a reason that has nothing to do with performance. Their wandb history carries one
`charts/episodic_return` row per finished episode — the same event this project records — so the
re-average is exact rather than an approximation.

The reference reads **4,759 at 1e9 steps** and **7,050 at the end of its own 2e9-step run**. Every arm
is above it at 1e9, and two things stop that being a claim of improvement: it is one seed against 17
to 22, and the curve shows the reference flat from about 0.3e9 to 1.0e9 and then climbing steeply, so
1e9 catches it immediately before its own jump. By its end it is inside the range the arms occupy.
The honest reading is that this reproduction tracks CleanRL's.

The original RND paper is not plotted. OpenAI's release is code only, so the paper's Figure 7 would
have to be digitised off the page, and CleanRL's run is both a closer match to what is being
reproduced here and available as real data.

## The plot

`plots/interim_arm_curves.pdf` — mean reward against steps, one line per arm, shaded standard error.
**A curve is drawn only while at least five of that arm's seeds have reached the step.** Past that the
mean is a handful of the fastest nodes' runs and stops meaning what the axis says. The lower panel
gives the seed count behind every point, so the truncation is visible rather than implied. Each arm's
curve ends where its fifth-deepest seed does:

| arm | curve ends at | seeds there |
|---|---|---|
| Arm 1 original | 1,186,201,600 | 5 |
| Arm 2 no RND clip | 1,179,648,000 | 5 |
| Arm 3 full batch | 1,169,817,600 | 5 |
| Arm 4 shallower | 1,327,104,000 | 5 |
| Arm 5 all three | 1,320,550,400 | 5 |

The seed count falls from about twenty to five between 1.0e9 and 1.2e9, and the curves get visibly
noisier there. That is the honest picture of what the data supports, which is why the count panel is
part of the figure rather than a footnote.

## Individual seeds of arm 1

`plots/arm1_individual_seeds.pdf` — the **first three seeds by seed number** of arm 1, drawn
individually against arm 1's mean over all of its seeds. First, not best, so the choice cannot flatter
the arm. Seed 1 has no data: its run was requeued on 2026-08-06 when a collaborator cancelled the job
holding it, and its marker is still in `queue/pending/` because no worker slot has freed since. So the
first three with data are seeds 2, 3 and 4.

| seed | reaches | last value | best |
|---|---|---|---|
| 2 | 1,156,710,400 | 5,915.5 | 6,011.5 |
| 3 | 1,150,156,800 | 8,397.5 | 9,053.5 |
| 4 | 1,137,049,600 | 1,090.5 | 1,301.0 |

**The mean resembles none of them.** Seed 3 reaches 6,700 by 0.32e9 steps and then sits there for two
thirds of the run; seed 2 is still under 1,200 at 0.7e9 and climbs to 5,900 only afterwards; seed 4
stays at about 400 — one room's key and door — for essentially the whole run so far. The smooth
average passes through the middle and describes no single run.

Each curve is a staircase rather than a slope: a plateau while the agent exploits the rooms it knows,
then a step up when exploration opens a new one. Averaging across seeds turns staircases with
different step times into a smooth ramp, which is a property of the averaging and not of any agent.

This is why the campaign runs 30 seeds per arm, and why the arms are not yet separated: with per-seed
outcomes spanning 400 to 8,400 inside a single arm, a mean over eighteen seeds still carries a
standard error of several hundred, and any difference between arms has to clear that.

## What the numbers say

1. **All five arms learn the task.** Every arm reaches a trailing mean between 5,900 and 7,400 by 1e9
   steps, against the 400 that the first room's key and door pay. The reproduction works and the
   auto-reset correction did not break it.
2. **The early separation has closed.** At 2e7 steps the two arms that leave the RND predictor
   unclipped led every other arm on the fraction of seeds that had scored at all — 50% against 17%.
   That measured how quickly a seed found its *first* reward, not how much it eventually collects, and
   by 1e9 steps almost every seed in every arm has found it. An early lead on a sparse-reward task is
   a lead in first-reward timing, and it should not have been read as more than that.
3. **The arms are not separated in the mean at 1e9 steps.** The spread between highest and lowest is
   1,533 while the standard errors are 560 to 795, so every pair overlaps.
4. **The one thing that does stand out is the worst seed, not the mean.** Arm 5's weakest seed scores
   4,584; arms 1 and 4 each have a seed still at 400, one room's worth, and arm 3 has one at 469. So
   arm 5 — the arm with all three departures — is the only one with no seed that failed to get past
   the first room. Whether that is a real difference in reliability or the luck of eighteen seeds is
   what the remaining half of the campaign is for.

The gradient statistics confirm the arms do what they are meant to, and they have moved since the
early ticks. Read at 2026-08-10 over the last five logged rows of every run:

| arm | joint norm | policy norm | predictor norm | predictor share of joint sq | predictor clip fires | mean scale on predictor |
|---|---|---|---|---|---|---|
| Arm 1 original | 0.545 | 0.537 | 0.083 | 2.1% | 47% | 0.879 |
| Arm 2 no RND clip | 0.601 | 0.596 | 0.078 | 1.5% | 0% | 1.000 |
| Arm 3 full batch | 0.598 | 0.591 | 0.079 | 1.7% | 56% | 0.836 |
| Arm 4 shallower | 0.606 | 0.603 | 0.053 | 0.8% | 58% | 0.838 |
| Arm 5 all three | 0.534 | 0.527 | 0.068 | 2.0% | 0% | 1.000 |

Arms 2 and 5 never scale the predictor, which is the intervention; arms 1, 3 and 4 scale it on about
half of optimizer steps. No run has recorded a non-finite norm. The early reading — a scale of about
0.38 firing on 92–97% of steps — was taken at 2e7 steps, when gradients were far larger; at 1e9 the
clip still fires often but cuts much less. So the intervention is real and well measured; what is not
yet established is that it changes the final score.

## Where this appears in the writeup

`07_reconstruction/development_document/RND_development_document.tex`, section 9: Table 81 is the
table above, Figure 29 is the arm-mean plot and Figure 30 the individual arm-1 seeds. Both are regenerated by the script named at the top; the
LaTeX tabular block is written to `analysis/interim_arm_table.tex` and pasted into the document, so
re-running the script and re-pasting is the whole update path.

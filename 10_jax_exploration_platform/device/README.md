# device

How many copies one process holds on each card model, per workload. The method, the card list and the calculation
are the shared skill `/p/rlprojects/.claude/skills/estimate-copies-per-gpu/` (`SKILL.md`, `card_facts.json`,
`scripts/capacity.py`); this folder holds the platform's series runner and its records.

| path | what it holds | written by |
|---|---|---|
| `series.py` | the doubling-series runner: one workload on one card at 64, 128, 256, ... copies, one process per size, two warm-up calls, the median round | by hand |
| `probes/` | one record per doubling series, `<workload>__<card>__<YYYY-MM-DD-HH-MM>.json` (time in Pacific); `probes/logs/<series>/` holds the output of each size | `series.py` |
| `observed/` | `<workload>.jsonl`, one line per finished unit: card, copies, seconds per iteration, peak memory | the fleet, when a unit finishes |
| `estimates/` | `<workload>.json`, generated: per card the copies, the environments per copy, the seconds per iteration, the peak memory, the status and the source | `capacity.py rough`, `estimate`, `correct` |
| `ledger.md` | generated: one table per workload with the five throughput columns (copies, seconds per iteration, total steps per second, steps per second per copy, hours per million steps per copy) | `capacity.py table` |

Status of a card in an estimates file:

1. `rough`: from the UVA survey alone, or from the survey scaled by one H100 NVL series; no measurement on that card.
2. `measured`: its own doubling series (the three key cards: H100 NVL, RTX 5080, RTX A4500).
3. `interpolated`: a card the survey holds, read off the key cards' fit at its relative speed.
4. `proxy`: a card the survey does not hold, copied from the card `card_facts.json` names, with its factor.
5. `corrected`: refitted from real units that ran on that card.

Tables show `interpolated` and `proxy` as "calculated".

## Commands

`series.py` runs on a compute node, or on serval05 or serval02 card 1 through their queue, never on a login node
(it refuses one). `capacity.py` uses the standard library only and runs anywhere.

```bash
S=/p/rlprojects/.claude/skills/estimate-copies-per-gpu/scripts
P=/p/rlprojects/RND/10_jax_exploration_platform

# one doubling series (inside a job or a queue unit that holds one card)
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python $P/device/series.py \
        --configuration-file $P/configs/experiments/<recorded file>.toml \
        --environment-family pointmaze --map large --environments-per-policy 4 \
        --update-style full_batch --bonus rnd_next_state --card "H100 NVL" --out-dir $P/device/probes

# every card from the three key series; then the cards that have real units; then the ledger
python3 $S/capacity.py estimate --workload "<workload>" --series <three series files> \
        --out $P/device/estimates/<workload>.json
python3 $S/capacity.py correct --estimates $P/device/estimates/<workload>.json \
        --observed $P/device/observed/<workload>.jsonl
python3 $S/capacity.py table --estimates $P/device/estimates/*.json > $P/device/ledger.md
```

Any trainer option beyond the four of the workload goes to `scripts/run_training.py` unchanged through
`--trainer-argument=<option>` (repeatable); the series then needs `--workload "<name in words>"`, because the option
changes the workload.

## Estimates so far

| file | what it is |
|---|---|
| `estimates/rough_from_survey_pointmaze_full_batch.json` | `rough` from the UVA survey alone (PointMaze Large, RND, one full-batch update per rollout, 4 environments per copy; survey of 2026-08-16). No measurement of the current trainer. Written 2026-11-12 PT |

# runs

One folder per training run, named `YYYY-MM-DD-HH-MM_<slug>` with the timestamp in Pacific Time and
every load-bearing knob spelled out in the slug (no shorthand, no opaque identifiers — the readable
sentence about the run lives in `manifest.yaml`).

**A run is one sweep.** One folder may span many jobs on many machines. Each job writes its own
shard, and the run-level files are always aggregates over the shards — no single job ever writes
them.

```
<YYYY-MM-DD-HH-MM_slug>/
├── experiment_background.md   # written by hand at launch: purpose, hyperparameters, changes, git state
├── manifest.yaml              # machine-readable identity: run id, git, component specs, compiled shape, seeding
├── config_resolved.yaml       # the full configuration after defaults and overrides
├── command.txt                # the exact launch command
├── queue/                     # a sweep's work units: pending/ running/ done/ failed/
├── slurm/                     # only when slurm was used
│   ├── submitted_jobids.txt   #   every job id, appended at submit time
│   └── jobs/<jobid>_<node>/   #   per submission: job.sbatch, assignment.json, hardware.json, job.log
├── data/<unit_id>.jsonl       # per-unit shards, one line per record, flushed as written
├── metrics.jsonl              # aggregated from the shards
├── summary.json               # end-of-run aggregate, including the throughput numbers
├── logs/                      # launcher and monitor logs (per-job logs live with their submission)
└── analysis/                  # per the analysis-folder rule
```

A single-machine run uses the same shape with one shard and a trivial aggregate; `queue/` and
`slurm/` are simply absent.

**No checkpoints.** Model state is not saved unless someone explicitly asks for it, so a unit that
dies part way cannot be continued — re-running it starts a new attempt in the same shard, and a unit
whose shard already ends in a `unit_complete` record is skipped.

Which files git keeps: `experiment_background.md`, `manifest.yaml`, `config_resolved.yaml` and
`command.txt` are committed. The shards, `metrics.jsonl`, `summary.json` and `logs/` are ignored —
all four can be regenerated from the shards, and the shards are large.

The two scripts: `../scripts/run_training.py` runs one unit and writes its shard;
`../scripts/aggregate_run.py` rebuilds `metrics.jsonl` and `summary.json` from every shard.

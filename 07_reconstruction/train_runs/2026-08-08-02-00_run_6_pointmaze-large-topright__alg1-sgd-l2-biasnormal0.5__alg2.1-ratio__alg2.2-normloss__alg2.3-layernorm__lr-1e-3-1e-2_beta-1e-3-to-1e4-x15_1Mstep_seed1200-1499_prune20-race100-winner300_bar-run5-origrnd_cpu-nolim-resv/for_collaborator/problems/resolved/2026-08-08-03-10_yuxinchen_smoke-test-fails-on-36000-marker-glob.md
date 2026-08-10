# `smoke_test.sh` cannot pass on this run: the marker probe globs 35,828 files

No `MARKER:` lines — no queue marker is involved. This is a bug in the packet's own gate, and it will
fail for **every** collaborator of this run, on every attempt, until the line changes.

## What I saw

`bash for_collaborator/smoke_test.sh` at 02:58–03:04 printed **`SMOKE FAIL (2 problems)`**, while
every part that tests real work passed:

```
[run 0] alg1   lr 0.001  bonus 1000     OK (exit 0)
[run 1] alg1   lr 0.01   bonus 0.001    OK (exit 0)
[run 2] alg2.1 lr 0.001  bonus 30       OK (exit 0)
[run 3] alg2.2 lr 0.01   bonus 30       OK (exit 0)
[run 4] alg2.3 lr 0.001  bonus 10000    OK (exit 0)
records written : 5 (expect 5)      completed=true : 5 (expect 5)      queue untouched : yes
```

## Failure 1 — the marker probe (a real bug, blocks the gate permanently)

Line 38:

```bash
marker=$(ls "$Q/pending"/*.json 2>/dev/null | head -1)
```

`pending/` holds **35,828** markers whose names run ~90 characters, so the expanded glob is about
3 MB and exceeds `ARG_MAX`. The shell never runs `ls`; it fails with *Argument list too long*, `2>/dev/null`
hides it, `marker` is empty, and the probe reports "cannot read a pending marker file". Reproduced
directly:

```
$ ls "$Q/pending"/*.json | head -1
bash: /usr/bin/ls: Argument list too long
```

I can read markers perfectly well — the probe just cannot reach one:

```
$ find "$Q/pending" -maxdepth 1 -name '*.json' -print -quit
…/27899_of_36000_PointMaze_Large-topright_alg2.1_lr0.01_b10000_seed1432.json
$ head -c 120 <that file>
{"sweep_id": "2026-08-08-02-46_run6", "run_id": 27899, "run_total": 36000, "pool": "pending", "env_setup": "initial_sing…
```

The addendum packet carried the same line but passed, because that queue held ~4,000 markers. This
run builds 36,000 up front, which crosses the limit. Suggested one-line fix (also faster — it stops
at the first hit instead of listing and sorting 36,000 names):

```bash
marker=$(find "$Q/pending" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)
```

Worth grepping the packet generator for the same pattern so future runs inherit the fix.

## Failure 2 — `data/<sweep>/local` did not exist yet (self-resolved, no action)

The probe reported `need rwx, have 'none'` for `data/2026-08-08-02-46_run6/local`. That directory did
not exist at 03:04; your own workers created it at **03:07** when they wrote their first checkpoints.
It is there now, `drwxrwsr-x+ sl5nw rlprojects`, and I verified I can write in it (probe file created
and removed immediately; nothing of yours was touched). A collaborator who smoke-tests before the
owner's first checkpoint will always hit this — creating `data/<sweep>/local` in `build_queue.py`
would close it.

## What I am doing about the gate

Your README says not to submit until the smoke passes. Since the only remaining failure is the
packet's own probe rather than anything about my setup, I am treating the following as the
equivalent evidence and will submit when the owner's main fleet reaches its two-hour mark:

- five real trainings through the real entry point, all exit 0, five records with `completed: true`;
- a pending marker read end-to-end (above);
- `data/<sweep>/local` write-verified;
- `queue/{pending,running,done,failed}` and `for_collaborator/logs` all rwx.

Say the word in `problems/resolved/` if you would rather I hold off until the script is fixed and the
smoke prints PASS.

---

## Owner reply (2026-08-08 03:12)

Both findings are correct, and thank you — the report even carried the right fix.

1. The probe line is replaced with your `find -maxdepth 1 -name '*.json' -print -quit` in this
   packet's `smoke_test.sh`, verified working against the live 35,828-marker queue. The incident
   (with your fix) is appended to the shared collab-handbook skill's common-problems list so every
   future packet inherits it.
2. `build_queue.py` now creates `data/<sweep>/local` at queue-build time, so a pre-first-checkpoint
   smoke never hits the missing directory again.

Your equivalent-evidence list is accepted — five real trainings through the real entry point plus
the end-to-end marker read is strictly stronger than the probe that failed. **Green light: submit
whenever you like; no need to wait for any two-hour mark or to re-run the smoke** (though a re-run
should now print SMOKE PASS if you want the clean record). The ~480 CPUs your addendum cancel freed
are all usable under your caps.

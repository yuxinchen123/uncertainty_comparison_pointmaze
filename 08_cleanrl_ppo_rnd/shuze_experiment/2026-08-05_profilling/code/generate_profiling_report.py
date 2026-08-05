"""Render the throughput and resource profiling report from the measured JSON files.

Reads everything under ../data/ and writes ../2026-08-05_throughput_and_resource_profiling.md.
The prose is here; the numbers all come from the JSON, so re-running this after more jobs land
updates the document without hand-editing it.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python generate_profiling_report.py
"""

import glob
import json
import os
import re
import statistics

import per_node_section
from options_detail import OPTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "2026-08-05_throughput_and_resource_profiling.md")

# The ladder's variants in order, with the plain-language description used in the tables.
VARIANT_LABEL = {
    "00_upstream": "upstream, unchanged",
    "01_env_threads": "envpool worker threads set to the job's cpu count",
    "02_opt_gpu_obs_rms": "observation-normalizer moments computed on the GPU",
    "03_opt_fused_policy_pass": "one policy trunk pass per rollout step instead of two",
    "04_opt_rnd_no_grad": "rollout bonus computed under no_grad",
    "05_opt_uint8_obs": "observation buffer held as uint8 instead of float32",
    "06_opt_gpu_norm_stats": "normalizer statistics kept on the GPU in float32",
    "07_opt_no_sync_update": "forced host-device synchronizations removed from the update",
    "08_opt_cudnn_benchmark": "cuDNN convolution autotuning on",
    "10_tier_a_plus_bugfix": "all of the above, plus the auto-reset fix",
    "11_amp_fp16_channels_last": "plus float16 autocast and channels-last (CHANGES THE NUMBERS)",
    "12_matmul_tf32": "plus TF32 matrix multiplication (CHANGES THE NUMBERS)",
    "13_torch_compile": "plus torch.compile on the three networks (CHANGES THE NUMBERS)",
}


def load_ladder():
    """Load every per-node ladder report, newest file per node."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "profile_*.json"))):
        d = json.load(open(f))
        ok = {v["variant"]: v for v in d["variants"] if "steps_per_second" in v}
        if ok:
            out[d["node"]] = {"report": d, "variants": ok}
    return out


def load_cpu_sweep():
    """Load the cores-versus-throughput sweep, sorted by cpu count."""
    rows = []
    for f in glob.glob(os.path.join(DATA, "cpu_sweep", "*.json")):
        d = json.load(open(f))
        ok = [v for v in d["variants"] if "steps_per_second" in v]
        if ok:
            rows.append((d["cpus_per_task"], d, ok[0]))
    return sorted(rows)


def load_packing():
    """Load the multi-run-per-GPU packing results, sorted by node then concurrency."""
    rows = []
    for f in glob.glob(os.path.join(DATA, "packing", "*.json")):
        d = json.load(open(f))
        if "aggregate_steps_per_second" in d:
            rows.append(d)
    return sorted(rows, key=lambda d: (d["node"], d["n_concurrent"]))


def load_shape_test():
    """Load the minibatch-shape experiment from its per-node log lines."""
    # The shape test prints one line per configuration rather than writing JSON, because it runs
    # four whole trainings and the line is the entire result.
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "shape_test", "*.log"))):
        pass
    for f in sorted(glob.glob(os.path.join(HERE, "..", "logs", "shape_*.log"))):
        node, rows = None, []
        for line in open(f):
            m = re.match(r"node=(\S+) cpus=(\d+)", line)
            if m:
                node = m.group(1)
            m = re.match(r"(\S+)\s+steps/s\s+(\d+) \| rollout\s+([\d.]+)s \| update\s+([\d.]+)s", line)
            if m:
                rows.append({"config": m.group(1), "steps_per_second": int(m.group(2)),
                             "rollout_seconds": float(m.group(3)), "update_seconds": float(m.group(4))})
        if node and rows:
            out[node] = rows
    return out


def load_canaries():
    """Load the per-architecture canary verdicts."""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "canary", "*.json"))):
        out.append(json.load(open(f)))
    return out


def option_gain(ladder, ladder_key):
    """Per node, this option's throughput divided by the previous ladder row's.

    The ladder is cumulative, so a variant's step over its predecessor is exactly the contribution of
    the single option that variant switched on.
    """
    order = list(VARIANT_LABEL)
    idx = order.index(ladder_key)
    prev_key = order[idx - 1]
    rows = []
    for node in sorted(ladder):
        vs = ladder[node]["variants"]
        if ladder_key in vs and prev_key in vs:
            before, after = vs[prev_key]["steps_per_second"], vs[ladder_key]["steps_per_second"]
            rows.append([f"`{node}`", vs[ladder_key].get("gpu_name", "?"),
                         f"{before:,.0f}", f"{after:,.0f}", f"{after / before:.2f}x"])
    return rows


def render_options(ladder, startup, shape):
    """Render one subsection per change the 30-seed run uses, with its measured gain."""
    L = [
        "## Every change the run uses, one at a time",
        "",
        "Each subsection below is one change that the 30-seed run actually switches on: what it is,",
        "the upstream code, the replacement, what it bought, and what it cost. The gain column is the",
        "ladder step attributable to that one change — the ladder is cumulative, so a row's throughput",
        "divided by the row above it is the contribution of the single option that row added.",
        "",
        "Changes that alter the arithmetic — float16 autocast, channels-last, TF32 matrix",
        "multiplication, `torch.compile` — are measured in the ladder table above but are **not** used,",
        "because the run is a faithful reproduction. They are not repeated here.",
        "",
    ]
    for n, opt in enumerate(OPTIONS, start=1):
        L += [f"### {n}. {opt['title']}", "", f"Flag: `{opt['flag']}`", "",
              "**What it is.**", opt["what"].strip(), "",
              "**Before** — upstream `cleanrl/ppo_rnd_envpool.py`:", "",
              "```python", opt["before"], "```", "",
              "**After** — `src/ppo_rnd_envpool_shuze.py`:", "",
              "```python", opt["after"], "```", "",
              "**What it gives.**", ""]
        if opt["ladder_key"]:
            rows = option_gain(ladder, opt["ladder_key"])
            if rows:
                L += [table(["node", "GPU", "before (steps/s)", "after (steps/s)", "gain"], rows), ""]
            else:
                L += ["No ladder rows reached this option yet.", ""]
        elif opt["flag"] == "--opt_fast_obs_norm_init" and startup:
            rows = [[f"`{n}`", f"{s['startup_original']:.0f} s", f"{s['startup_fast']:.0f} s",
                     f"{s['startup_original'] / s['startup_fast']:.0f}x"]
                    for n, s in startup if s.get("startup_original") and s.get("startup_fast")]
            L += [table(["node", "start-up before", "start-up after", "ratio"], rows), "",
                  "This is start-up time, not throughput, so it does not appear in the ladder.", ""]
        elif opt["flag"] == "--opt_fixed_minibatch_shape" and shape:
            for node, srows in shape.items():
                L += [f"On `{node}`:", "",
                      table(["configuration", "steps/s", "update phase (s)"],
                            [[r["config"], f"{r['steps_per_second']:,}", f"{r['update_seconds']:.2f}"]
                             for r in srows]), ""]
        L += ["**What it costs.**", opt["costs"].strip(), "", "---", ""]
    return L


def table(headers, rows):
    """Render a markdown table."""
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    """Assemble the report and write it."""
    ladder, sweep, packing, shape, canaries = (load_ladder(), load_cpu_sweep(), load_packing(),
                                               load_shape_test(), load_canaries())
    L = []
    L += [
        "<!-- Generated by code/generate_profiling_report.py from the JSON under data/.",
        "     Do not edit by hand; rerun the generator. -->",
        "",
        "# Profiling CleanRL PPO + RND: throughput, and how many runs fit on a GPU",
        "",
        "Measured 2026-08-05 on this cluster. The workload is CleanRL's `ppo_rnd_envpool.py`",
        "(upstream commit `fe8d8a0`) on `MontezumaRevenge-v5`: 128 envpool copies, a 128-step rollout,",
        "a 16,384-row batch, 4 minibatches, 4 update epochs.",
        "",
        "Two numbers frame everything below. CleanRL reports one seed of this workload at",
        "2 billion steps in about 250 hours, which is **2,222 steps per second**. And the whole",
        "workload holds **under 5 GB of GPU memory**. So the published run was neither GPU-memory",
        "bound nor, as the measurements below show, GPU-compute bound.",
        "",
        "## How the measurements were taken",
        "",
        "- **Every configuration runs in a fresh process.** cuDNN's autotune cache and the CUDA",
        "  allocator carry state within a process, so toggling options in one process manufactures",
        "  speedups that are not there.",
        "- **Throughput is `batch_size / iteration_seconds`**, averaged over the updates after a",
        "  warm-up of 4, not the script's own cumulative counter, which is dragged down by start-up.",
        "- **The ladder is cumulative**: each row adds one option to the row above, so a row's gain is",
        "  attributable to the option it added.",
        "- **The options are split by whether they change the arithmetic.** A faithful reproduction may",
        "  take the first group; the second group is measured but not used for the 30-seed run.",
        "",
    ]

    # ---- which GPUs can run this at all ----
    if canaries:
        L += ["## Which GPUs can run this stack", "",
              "The environment uses `torch 2.6.0+cu124`, compiled for `sm_50` through `sm_90`. That",
              "choice is what makes the Pascal cards usable: the `torch 2.10.0+cu128` build that",
              "shadows the shared environment from `/u/sl5nw/.local` compiles for `sm_70` and up with",
              "no PTX fallback, so it cannot run any card below Turing. Verified card by card:", ""]
        rows = [[f"`{c['node']}`", c.get("gpu_name", "?"), c.get("compute_capability", "?"),
                 "**runs**" if c["verdict"] == "OK" else "**fails**",
                 "yes" if c.get("triton_usable_for_torch_compile") else "no",
                 "yes" if c.get("bf16_supported_native") else "no"] for c in canaries]
        L += [table(["node", "GPU", "compute capability", "verdict", "torch.compile usable",
                     "native bf16"], rows), "",
              "`torch.compile` needs Triton, which needs compute capability 7.0 or higher, so it is",
              "unavailable on the Pascal cards. Note that `torch.cuda.is_bf16_supported()` returns",
              "`True` on Pascal because it counts emulation; emulated bf16 is far slower than float32,",
              "so the native column is the one to read.", ""]

    # ---- results per node: the table to read when deciding where to put work ----
    L += per_node_section.render(table)

    # ---- the ladder ----
    if ladder:
        L += ["## The optimization ladder, per GPU", "",
              "Each row adds one option to the row above. All at 16 cpus per run.", ""]
        variants = [v for v in VARIANT_LABEL if any(v in d["variants"] for d in ladder.values())]
        headers = ["option added"] + [n for n in sorted(ladder)]
        rows = []
        for v in variants:
            r = [VARIANT_LABEL[v]]
            for n in sorted(ladder):
                x = ladder[n]["variants"].get(v)
                r.append(f"{x['steps_per_second']:,.0f}" if x else "not reached")
            rows.append(r)
        L += [table(headers, rows), ""]
        gains = []
        for n in sorted(ladder):
            vs = ladder[n]["variants"]
            if "00_upstream" in vs and "08_opt_cudnn_benchmark" in vs:
                base, tier_a = vs["00_upstream"], vs["08_opt_cudnn_benchmark"]
                gains.append([f"`{n}`", base.get("gpu_name", "?"),
                              f"{base['steps_per_second']:,.0f}", f"{tier_a['steps_per_second']:,.0f}",
                              f"{tier_a['steps_per_second']/base['steps_per_second']:.2f}x",
                              f"{tier_a['gpu_memory_peak_mb']:,.0f}"])
        if gains:
            L += ["### Before and after, same numbers either way", "",
                  table(["node", "GPU", "upstream steps/s", "after the options", "gain",
                         "peak GPU memory (MB)"], gains), "",
                  "The single largest step is holding the observation buffer as `uint8` rather than",
                  "`float32`. The buffer is 128 x 128 x 4 x 84 x 84, which is 1.85 GB as float32 and",
                  "0.46 GB as uint8, and the original converted uint8 to float32 **on the host** before",
                  "copying, so it moved four times the bytes it needed to. The networks divide by 255.0",
                  "either way, and every uint8 value is exactly representable in float32, so the",
                  "arithmetic is unchanged.", ""]
        startups = [(n, d["report"]["startup"]) for n, d in ladder.items() if d["report"].get("startup")]
        if startups:
            L += ["### Start-up", ""]
            rows = [[f"`{n}`", f"{s['startup_original']:.0f} s", f"{s['startup_fast']:.0f} s",
                     f"{s['startup_original']/s['startup_fast']:.0f}x"]
                    for n, s in startups if s.get("startup_original") and s.get("startup_fast")]
            L += [table(["node", "as written", "rewritten", "ratio"], rows), "",
                  "The observation normalizer is primed by a random agent before training. The original",
                  "builds that batch by calling `.tolist()` on a uint8 array 6,400 times, creating on the",
                  "order of a hundred million Python objects, then `np.stack`s a list of nested lists.",
                  "Filling a preallocated uint8 buffer instead gives bit-identical statistics. This is",
                  "paid once per job, so it is paid again on every resume.", ""]

    # ---- the cpu sweep ----
    if sweep:
        L += ["## How many cpus a run should get", "",
              "One GPU held constant (jaguar03, RTX A4500), the run configuration, only the cpu",
              "allocation varying.", ""]
        rows = [[c, f"{v['steps_per_second']:,.0f}", f"{v['rollout_seconds_mean']:.2f}",
                 f"{v['update_seconds_mean']:.2f}", f"{v['rollout_seconds_mean']/128*1000:.1f}",
                 f"{v['steps_per_second']/c:,.0f}"] for c, d, v in sweep]
        L += [table(["cpus", "steps/s", "rollout (s)", "update (s)", "per rollout step (ms)",
                     "steps/s per cpu"], rows), "",
              "**The curve flattens at 8 to 16 cpus.** Past that, more cores buy nothing: the rollout",
              "stops falling at about 17 ms per step and the update is a flat GPU cost that cpus do not",
              "touch. This corrects a natural assumption. envpool measured *on its own* keeps scaling",
              "with cores, but in this script envpool and the GPU are strictly serialized, so once",
              "envpool is fast enough the floor becomes the per-step policy forward, bonus forward and",
              "host transfers.",
              "",
              "For a campaign, the number that matters is the last column: **8 cpus per run gets most of",
              "the peak rate at half the cores**, and cpus are the capped resource on this cluster",
              "(400 in `gpu`, 80 in `gnolim`).", ""]

    # ---- the shape finding ----
    if shape:
        L += ["## A trap: the bug fix and cuDNN autotuning are destructive together", "",
              "Dropping the auto-reset rows makes the surviving row count differ every iteration, so a",
              "minibatch sized as `valid // num_minibatches` changes shape on nearly every update.",
              "`cudnn.benchmark` autotunes convolution algorithms **per shape**, so it re-benchmarks",
              "constantly. Each change is sound on its own and they are destructive together.", ""]
        for node, rows in shape.items():
            L += [f"On `{node}`:", "",
                  table(["configuration", "steps/s", "rollout (s)", "update (s)"],
                        [[r["config"], f"{r['steps_per_second']:,}", f"{r['rollout_seconds']:.2f}",
                          f"{r['update_seconds']:.2f}"] for r in rows]), ""]
        L += ["The fix is to hold back a fixed row allowance so the minibatch shape is constant for the",
              "whole run. It drops a few extra genuine rows per update, chosen at random by the shuffle",
              "that was already there, instead of training on rows that describe transitions which never",
              "happened.", ""]

    # ---- packing ----
    if packing:
        L += ["## How many runs fit on one GPU", "",
              "A single run leaves the GPU idle for more than half of every iteration, because the",
              "rollout alternates a small GPU forward pass with a cpu-side environment step. Packing",
              "several runs on one GPU should fill those gaps. Measured directly:", ""]
        rows = [[f"`{d['node']}`", d["n_concurrent"], d["cpus_per_run"],
                 f"{d['mean_steps_per_second_per_run']:,.0f}",
                 f"{d['aggregate_steps_per_second']:,.0f}",
                 f"{d['steps_per_second_per_cpu']:,.0f}",
                 f"{d.get('gpu_memory_used_mb_while_running', 0):,.0f}"] for d in packing]
        L += [table(["node", "runs on the GPU", "cpus per run", "steps/s per run",
                     "aggregate steps/s", "steps/s per cpu", "GPU memory in use (MB)"], rows), ""]
    else:
        L += ["## How many runs fit on one GPU", "",
              "The packing jobs had not finished when this document was last generated. Rerun the",
              "generator once `data/packing/` is populated.", ""]

    # The per-change detail: what each option is, its before/after code, its gain, and its cost.
    startups = [(n, d["report"]["startup"]) for n, d in ladder.items() if d["report"].get("startup")]
    L += render_options(ladder, startups, shape)

    L += ["## What this means for the 30-seed run", "",
          "The run configuration is: every option that leaves the arithmetic unchanged, the auto-reset",
          "fix, a constant minibatch shape, and 8 cpus per run. The options that change the arithmetic",
          "are measured above and switched off, because the point of the run is a faithful reproduction.",
          ""]
    with open(OUT, "w") as f:
        f.write("\n".join(L))
    os.chmod(OUT, 0o664)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()

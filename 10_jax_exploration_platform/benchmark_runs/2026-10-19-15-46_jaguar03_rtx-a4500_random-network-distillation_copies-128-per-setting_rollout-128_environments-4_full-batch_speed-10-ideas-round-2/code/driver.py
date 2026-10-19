"""Run the ten jaguar03 speed ideas inside one Slurm job holding several RTX A4500 cards.

Every round puts the baseline on one card and ideas on the others, all at the same time, so each
idea is timed beside a baseline under the same node load. The card that carries the baseline turns
over from round to round, and each idea runs twice on two different cards, so a difference between
cards cannot pass for an idea's effect. Each process gets its own block of the job's processors.

Usage (inside the job): python3 driver.py <benchmark folder>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/p/rlprojects/RND/.venvs/platform_jax/bin/python"
BENCH = "/p/rlprojects/RND/10_jax_exploration_platform/benchmarks/harness/bench_jaguar03_ideas.py"
CMDBUF_ALL = ("--xla_gpu_enable_command_buffer="
              "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE,CONDITIONAL")

# each idea: extra script arguments, processes sharing the card, XLA flags, a compilation cache,
# or a smaller processor count; everything not named is the baseline's
IDEAS = {
    "01_four_settings_one_program": {"args": ["--settings", "4"]},
    "02_eight_settings_one_program": {"args": ["--settings", "8"]},
    "03_two_programs_per_card": {"processes": 2},
    "04_four_programs_per_card": {"processes": 4},
    "05_command_buffers_off": {"xla": "--xla_gpu_enable_command_buffer="},
    "06_command_buffers_with_loops": {"xla": CMDBUF_ALL},
    "07_latency_hiding_scheduler": {"xla": "--xla_gpu_enable_latency_hiding_scheduler=true"},
    "08_compilation_cache": {"cache": True},
    "09_two_processors": {"cpus": 2},
    "10_diagnostics_on": {"args": ["--diagnostics"]},
}
REPLICATES = 2


def plan_rounds(n_cards: int):
    """Rounds of (card, idea, replicate) triples: one baseline per round, ideas on the rest.

    before: 4 cards, ideas A..J twice each
    after:  round 0 = baseline card 0, A1 card 1, B1 card 2, C1 card 3; round 1 = baseline card 1,
            D1 card 0, ...; a replicate never lands on the card its first replicate used
    """
    queue = [(name, rep) for rep in range(REPLICATES) for name in IDEAS]
    used = {}
    rounds = []
    r = 0
    while queue:
        base_card = r % n_cards
        units = [(base_card, "baseline", r)]
        for card in range(n_cards):
            if card == base_card:
                continue
            pick = next((u for u in queue if used.get(u[0]) != card), None)
            if pick is None:
                continue
            queue.remove(pick)
            used[pick[0]] = card
            units.append((card, pick[0], pick[1]))
        rounds.append(units)
        r += 1
    return rounds


def launch(folder: Path, card_token: str, cpu_block, name: str, rep: int, proc: int, spec: dict,
           cache_dir: Path = None):
    """Start one measuring process pinned to one card and some of its processors."""
    tag = f"{name}__rep{rep}__card{card_token}__proc{proc}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card_token, PYTHONNOUSERSITE="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(round(0.8 / spec.get("processes", 1), 3)))
    if spec.get("xla"):
        env["XLA_FLAGS"] = spec["xla"]
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
        env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    cpus = cpu_block[:spec.get("cpus", len(cpu_block))]
    env["OMP_NUM_THREADS"] = str(len(cpus))
    cmd = ["taskset", "-c", ",".join(str(c) for c in cpus), PYTHON, "-u", BENCH,
           "--output", str(folder / "data" / f"{tag}.json"), *spec.get("args", [])]
    log = open(folder / "logs" / f"{tag}.log", "w")
    return tag, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def run_unit(folder: Path, card_token: str, cpu_block, name: str, rep: int):
    """One unit's processes on one card; the cache idea runs a cold then a warm process."""
    spec = {} if name == "baseline" else IDEAS[name]
    if spec.get("cache"):
        cache = folder / "data" / f"cache_{name}_rep{rep}"
        cache.mkdir(parents=True, exist_ok=True)
        tag, p = launch(folder, card_token, cpu_block, name + "_cold", rep, 0, spec, cache)
        p.wait()
        return [launch(folder, card_token, cpu_block, name + "_warm", rep, 0, spec, cache)]
    return [launch(folder, card_token, cpu_block, name, rep, k, spec)
            for k in range(spec.get("processes", 1))]


def main():
    """Plan the rounds over the job's cards, run them one after another, log each round."""
    folder = Path(sys.argv[1]).resolve()
    (folder / "logs").mkdir(exist_ok=True)
    cards = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    cpus = sorted(os.sched_getaffinity(0))
    per_card = len(cpus) // len(cards)
    blocks = [cpus[i * per_card:(i + 1) * per_card] for i in range(len(cards))]
    rounds = plan_rounds(len(cards))
    (folder / "data" / "plan.json").write_text(json.dumps(
        {"cards": cards, "cpu_blocks": blocks, "rounds": rounds}, indent=1))
    for r, units in enumerate(rounds):
        start = time.time()
        # the cache idea's cold process runs before its warm one, so its unit is started in a
        # thread of its own; every other unit's processes start together
        import threading
        procs = []
        lock = threading.Lock()

        def start_unit(card, name, rep):
            """Start one unit and keep its processes for the wait below."""
            started = run_unit(folder, cards[card], blocks[card], name, rep)
            with lock:
                procs.extend(started)

        threads = [threading.Thread(target=start_unit, args=u) for u in units]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        codes = {tag: p.wait() for tag, p in procs}
        print(json.dumps({"round": r, "units": units, "exit_codes": codes,
                          "seconds": round(time.time() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()

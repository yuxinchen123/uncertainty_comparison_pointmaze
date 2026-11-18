"""Seconds per update of the pixel Montezuma trainer under named settings, and of the fast last layer's parts.

    python profile_fast_last_layer.py --output-dir DIR --card h100 --copies 4 --seeds 1,2,3,4 \
        --variant default --variant game-events [--parts 128,16,4] [--budget-seconds 540]

One process holds everything: the imports, the game, and one initialized and warmed state that
every variant starts from. Per variant: one compile of the whole update (ahead of time, so its
memory analysis is read from the same executable that runs), two warm updates, then the timed
updates. A timed update is what train_final.py does for one update: the compiled update, reading
its metrics to the host, the non-finite check, the JSON line written and flushed, the fold into
the game records and the first-times file when it changed. Device time and host time are
recorded separately; their sum is the seconds per update.

--parts times the fast last layer's post-rollout pass on its own, for each listed sharing group
and each listed bonus form (the reference first), on stored features of the rollout's shape, and
the two bonus-profile functions; every form's bonus is compared with the reference form's.

--rollout NAME (repeatable) times the rollout's time loop alone (PixelPPORND.collect_steps) of
each named variant, from the shared warmed state: the game, the networks of the rollout, the bonus
or the stored features, and the per-step event reductions.

--compare-updates N runs N real updates of --compare-variant (a fast-last-layer variant in the
reference form) and, on the stored features of every rollout, computes the bonus of each listed
bonus form beside the reference form's: the largest absolute and relative difference per update
and over all N updates. Training continues with the reference bonus, so the N rollouts are those
of the reference.

Each measurement is one JSON line in DIR/profile.jsonl, flushed at once. A measurement that would
start after --budget-seconds is recorded as skipped. Other processes on the card are read before
and after each timed block; a block with another process on the card is marked for repeating.
Run with the deterministic runtime (XLA_FLAGS containing --xla_gpu_exclude_nondeterministic_ops=true)
on a compute node or a profiling card, never on a login node.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
PLATFORM_ROOT = SCRIPT_DIRECTORY.parents[1]
sys.path.insert(0, str(PLATFORM_ROOT / "src"))

import game_records  # noqa: E402  (standard library)
from train_final import ACCEPTED_ENVIRONMENT, REQUIRED_XLA_FLAG, seeds_with_non_finite_loss  # noqa: E402

ARCHIVE = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/"
               "2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_"
               "cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_"
               "h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/data")
# The archived run whose learner and environment settings each card profile copies exactly.
CARD_RUNS = {"h100": ARCHIVE / "h100-serval02-card-1_final-training_seeds-1-to-4/run/run.json",
             "a4500": ARCHIVE / "a4500-jaguar03_final-training_seed-5/run/run.json"}
# Learner fields of the archived record that the profile sets itself.
PROFILE_FIELDS = ("n_copies", "observation_warmup_iterations")
GAME_EVENTS = ({"record_game_events": True}, {"report_game_events": True})
# The learner settings of the fast last layer's reference form (the defaults give the accepted form).
REFERENCE_FORM_SETTINGS = {"fast_last_layer_chunk_rows": 0, "fast_last_layer_plain_errors": False,
                           "fast_last_layer_float32_products": "none", "fast_last_layer_small_solve": "cholesky"}


def card_settings(card):
    """The archived learner and environment settings of one card profile (h100: four copies, a4500: one copy)."""
    record = json.loads(CARD_RUNS[card].read_text())
    learner = {name: value for name, value in record["learner"].items() if name not in PROFILE_FIELDS}
    return learner, dict(record["environment"])


def fast_layer_settings(copies, group):
    """Learner settings of the fast last layer: the plan's four step sizes over four copies, else 0.1."""
    if copies == 4:
        return {"fast_last_layer_sharing_group": group,
                "per_copy_scalar_settings": [["fast_last_layer_step", [0.01, 0.03, 0.1, 0.3]]]}
    return {"fast_last_layer_step": 0.1, "fast_last_layer_sharing_group": group}


def named_variant(name, copies):
    """Learner and environment changes of a named variant.

    default; game-events; fast-last-layer_sharing-group-<g>_game-events_bonus-profile (the
    setting the grid runs use); fast-last-layer_sharing-group-<g> (no logging); a final part
    _reference-form gives the reference form of the bonus instead of the default (accepted) form.
    A JSON object {"name": ..., "learner": {...}, "environment": {...}} is taken as it is.
    """
    if name.startswith("{"):
        value = json.loads(name)
        return value["name"], value.get("learner", {}), value.get("environment", {})
    if name == "default":
        return name, {}, {}
    if name == "game-events":
        return name, dict(GAME_EVENTS[0]), dict(GAME_EVENTS[1])
    # before: "fast-last-layer_sharing-group-16_game-events_bonus-profile"
    # after: group 16, events on, bonus profile on
    parts = name.split("_")
    if parts[0] != "fast-last-layer" or not parts[1].startswith("sharing-group-"):
        raise ValueError(f"unknown variant name: {name}")
    learner = fast_layer_settings(copies, int(parts[1][len("sharing-group-"):]))
    environment = {}
    for part in parts[2:]:
        if part == "game-events":
            learner.update(GAME_EVENTS[0])
            environment.update(GAME_EVENTS[1])
        elif part == "bonus-profile":
            learner["record_bonus_profile"] = True
        elif part == "reference-form":
            learner.update(REFERENCE_FORM_SETTINGS)
        else:
            raise ValueError(f"unknown part {part} of variant name {name}")
    return name, learner, environment


def merged_learner(base, change):
    """The archived learner with a variant's changes; per-copy settings are joined by name."""
    # before: base per_copy_scalar_settings [], change [["fast_last_layer_step", [...]]]
    # after: [["fast_last_layer_step", [...]]]
    merged = {**base, **{name: value for name, value in change.items() if name != "per_copy_scalar_settings"}}
    if "per_copy_scalar_settings" in change:
        settings = dict(base.get("per_copy_scalar_settings") or [])
        settings.update(dict(change["per_copy_scalar_settings"]))
        merged["per_copy_scalar_settings"] = sorted(settings.items())
    return merged


def other_processes():
    """(card UUID, other processes on it) read from nvidia-smi; the card is the one this process uses."""
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    # before: "GPU-fed6..., 12345, 30000" per line ; after: [("GPU-fed6...", 12345, 30000)]
    entries = [tuple(item.strip() for item in line.split(",")) for line in rows.splitlines() if line.strip()]
    mine = [uuid for uuid, pid, _ in entries if int(pid) == os.getpid()]
    card = mine[0] if mine else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    others = [{"pid": int(pid), "used_mib": int(used)} for uuid, pid, used in entries
              if uuid == card and int(pid) != os.getpid()]
    return card, others


def spread_summary(values):
    """Median, smallest, largest and their difference of repeated seconds."""
    return {"median": statistics.median(values), "smallest": min(values), "largest": max(values),
            "spread": max(values) - min(values), "count": len(values)}


def throughput(seconds, copies, steps_per_update):
    """The five throughput columns of one setting (rule rl-throughput-tables.md)."""
    per_copy = steps_per_update / seconds
    return {"copies": copies, "seconds_per_update": seconds,
            "total_steps_per_second_millions": copies * per_copy / 1e6,
            "steps_per_second_per_copy_thousands": per_copy / 1e3,
            "hours_per_million_steps_per_copy": 1e6 / (3600 * per_copy)}


class Profile:
    """The shared state, the output file and the time budget of one profiling process."""

    def __init__(self, args):
        """Configure the runtime, build the game and one warmed state for every variant."""
        self.args, self.started = args, time.monotonic()
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output = (self.output_dir / "profile.jsonl").open("a")
        self.write({"record": "process start", "host": os.uname().nodename, "pid": os.getpid(),
                    "arguments": {name: str(value) for name, value in vars(args).items()}})
        # Precision and determinism exactly as train_final.py sets them.
        if REQUIRED_XLA_FLAG not in os.environ.get("XLA_FLAGS", "").split():
            raise SystemExit(f"launch with XLA_FLAGS containing {REQUIRED_XLA_FLAG}")
        os.environ["JAX_ENABLE_X64"] = "true"
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        if args.compilation_cache:
            # A persistent cache only shortens compilation; the compiled code is the same.
            jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache))
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        from exploration_platform.atari_pixels.config import PixelPPOConfig
        from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
        from exploration_platform.atari_pixels.training import PixelPPORND
        self.jax, self.PixelPPOConfig, self.PixelPPORND = jax, PixelPPOConfig, PixelPPORND
        self.AtariPixelEnvironment = AtariPixelEnvironment
        self.seeds = tuple(int(value) for value in args.seeds.split(","))
        if len(self.seeds) != args.copies:
            raise SystemExit(f"--seeds needs one seed per copy: {args.copies} copies, {len(self.seeds)} seeds")
        self.learner, self.environment = card_settings(args.card)
        self.environments = {}
        self.write({"record": "imports done", "seconds": time.monotonic() - self.started,
                    "device_kind": jax.devices()[0].device_kind, "jax_version": jax.__version__})
        if args.parts_only:
            # The post-rollout pass needs only the predictor's output layer: no game, no warmup.
            from exploration_platform.atari_pixels.networks import initialize_networks
            parameters, _ = initialize_networks(jax.numpy.asarray(self.seeds, jax.numpy.uint32), 18)
            self.state, self.layer = None, parameters["predictor"]["dense"][-1]
            return
        # One initialized and warmed state, shared by every variant (the state layout does not
        # depend on the logging or the bonus options).
        trainer = self.trainer({}, {})
        self.write({"record": "game built", "seconds": time.monotonic() - self.started})
        if args.initial_checkpoint:
            # Start from a saved run's complete state (for example the archive's end of training).
            import pickle
            with args.initial_checkpoint.open("rb") as stream:
                payload = pickle.load(stream)
            if tuple(payload["copy_seeds"]) != self.seeds:
                raise SystemExit(f"the checkpoint holds seeds {payload['copy_seeds']}, not {self.seeds}")
            self.state = jax.tree.map(jax.device_put, payload["state"])
            self.layer = self.state.parameters["predictor"]["dense"][-1]
            self.write({"record": "checkpoint loaded", "path": str(args.initial_checkpoint),
                        "iteration": int(self.state.iteration), "seconds": time.monotonic() - self.started})
            return
        state = trainer.initialize(self.seeds)
        jax.block_until_ready(state)
        self.write({"record": "state initialized", "seconds": time.monotonic() - self.started})
        state = trainer.warmup(state)
        jax.block_until_ready(state)
        self.state, self.layer = state, state.parameters["predictor"]["dense"][-1]
        self.write({"record": "initialized and warmed", "seconds": time.monotonic() - self.started,
                    "observation_warmup_iterations": args.observation_warmup_iterations})

    def write(self, record):
        """Append one record with the elapsed seconds of the process, flushed at once."""
        record = {**record, "elapsed_seconds": time.monotonic() - self.started if hasattr(self, "started") else 0.0,
                  "unix": time.time()}
        self.output.write(json.dumps(record, default=str) + "\n")
        self.output.flush()
        print(json.dumps(record, default=str), flush=True)

    def remaining(self):
        """Seconds left of the time budget."""
        return self.args.budget_seconds - (time.monotonic() - self.started)

    def config(self, learner_change):
        """The learner configuration of one variant."""
        return self.PixelPPOConfig(n_copies=self.args.copies,
                                   observation_warmup_iterations=self.args.observation_warmup_iterations,
                                   **merged_learner(self.learner, learner_change))

    def trainer(self, learner_change, environment_change):
        """A trainer for one variant; games with the same settings are built once."""
        environment = {**ACCEPTED_ENVIRONMENT, **self.environment, **environment_change}
        key = json.dumps(environment, sort_keys=True)
        if key not in self.environments:
            self.environments[key] = self.AtariPixelEnvironment(num_envs=128, **environment)
        return self.PixelPPORND(self.config(learner_change), self.environments[key])

    def variant(self, name):
        """Compile, warm and time one variant's whole update as train_final.py runs it."""
        jax = self.jax
        import numpy as np
        name, learner_change, environment_change = named_variant(name, self.args.copies)
        expected = self.args.expected_variant_seconds
        if self.remaining() < expected:
            self.write({"record": "variant skipped", "name": name, "reason": "time budget",
                        "remaining_seconds": self.remaining(), "expected_seconds": expected})
            return
        trainer = self.trainer(learner_change, environment_change)
        config = trainer.config
        # Compile ahead of time: the executable that runs is the one whose memory is analyzed.
        started = time.monotonic()
        lowered = trainer._compiled_iteration.lower(self.state)
        lowered_seconds = time.monotonic() - started
        compiled = lowered.compile()
        compile_seconds = time.monotonic() - started - lowered_seconds
        analysis = compiled.memory_analysis()
        memory = {field: getattr(analysis, field, None) for field in (
            "temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes",
            "alias_size_in_bytes", "generated_code_size_in_bytes")}

        def one_update(state):
            """The compiled update, then train_iteration's separate observation buffer."""
            state, metrics = compiled(state)
            return state._replace(observations=jax.numpy.array(state.observations, copy=True)), metrics

        # Two warm updates, then the timed ones, each continuing from the previous state.
        state, warm = self.state, []
        for _ in range(2):
            began = time.perf_counter()
            state, metrics = one_update(state)
            jax.block_until_ready((state, metrics))
            warm.append(time.perf_counter() - began)
        card, before = other_processes()
        directory = self.output_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        metrics_path = directory / "metrics.jsonl"
        records = game_records.GameRecords(environments_per_policy=config.n_envs, rollout_steps=config.num_steps)
        first_times = directory / game_records.FIRST_TIMES_FILE
        windows = directory / game_records.WINDOWS_FILE
        device, host = [], []
        with metrics_path.open("w") as stream:
            for index in range(self.args.timed_updates):
                began = time.perf_counter()
                state, metrics = one_update(state)
                values = jax.device_get(metrics)
                jax.block_until_ready(state)
                middle = time.perf_counter()
                # The host work of one update in train_final.py (the update number restarts at 1
                # so that the fold accepts the lines).
                line = {"update": index + 1, "unix": time.time(),
                        **{key: np.asarray(value).tolist() for key, value in values.items()}}
                broken = seeds_with_non_finite_loss(line["loss"], self.seeds)
                if broken:
                    raise FloatingPointError(f"variant {name}: the loss of the copies with seeds {broken} is not finite")
                stream.write(json.dumps(line) + "\n")
                stream.flush()
                window = records.fold(line)
                if window is not None:
                    game_records.append_window(windows, window)
                if records.changed:
                    game_records.write_first_times(first_times, records)
                device.append(middle - began)
                host.append(time.perf_counter() - middle)
        _, after = other_processes()
        totals = [left + right for left, right in zip(device, host)]
        peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
        self.write({"record": "variant", "name": name, "learner_change": learner_change,
                    "environment_change": environment_change, "learner": asdict(config),
                    "lower_seconds": lowered_seconds, "compile_seconds": compile_seconds,
                    "warm_update_seconds": warm, "update_seconds": totals, "device_seconds": device,
                    "host_seconds": host, "update_seconds_summary": spread_summary(totals),
                    "device_seconds_per_update": spread_summary(device),
                    "host_seconds_per_update": spread_summary(host),
                    **throughput(statistics.median(totals), config.n_copies, config.batch_size),
                    "memory_analysis": memory, "process_peak_bytes_in_use": peak,
                    "card": card, "other_processes_before": before, "other_processes_after": after,
                    "repeat_needed": bool(before or after)})
        del state, metrics, compiled, lowered

    def parts(self, groups, forms):
        """Time the post-rollout pass of each group and bonus form, and the bonus-profile functions."""
        jax = self.jax
        jnp = jax.numpy
        import numpy as np
        from exploration_platform.atari_pixels.networks import dense
        from exploration_platform.atari_pixels.training import (fast_last_layer_bonus,
                                                                predictor_bonus_from_features, row_alignment)
        config = self.config(fast_layer_settings(self.args.copies, groups[0]))
        steps = config.fast_last_layer_steps
        options = {name: {"true": True, "false": False}.get(value, value)
                   for name, value in config.iteration_compiler_options}
        # Stored features of one rollout's shape: [time, copies, environments, 512] after a ReLU,
        # and targets of the same shape; the layer is the state's own predictor output layer.
        shape = (config.num_steps, config.n_copies, config.n_envs, 512)
        keys = jax.random.split(jax.random.PRNGKey(self.args.parts_seed), 2)
        hidden = jax.nn.relu(jax.random.normal(keys[0], shape, jnp.float32))
        targets = jax.random.normal(keys[1], shape, jnp.float32)
        layer = self.layer
        # The plain errors a rollout with that form would store: the predictor's output minus the target's.
        errors = jax.jit(lambda h, y, w: dense(w, h, config.dense_batching) - y)(hidden, targets, layer)

        def timed(label, function, *arguments, reference=None, **context):
            """Compile, warm twice and time one function; compare its result with a reference."""
            if self.remaining() < self.args.expected_part_seconds:
                self.write({"record": "part skipped", "part": label, "reason": "time budget", **context})
                return None
            compiled = jax.jit(function, **({"compiler_options": options} if options else {}))
            began = time.perf_counter()
            compiled = compiled.lower(*arguments).compile()
            compile_seconds = time.perf_counter() - began
            for _ in range(2):
                result = jax.block_until_ready(compiled(*arguments))
            card, before = other_processes()
            seconds = []
            for _ in range(self.args.part_repeats):
                began = time.perf_counter()
                result = jax.block_until_ready(compiled(*arguments))
                seconds.append(time.perf_counter() - began)
            _, after = other_processes()
            record = {"record": "part", "part": label, **context, "compile_seconds": compile_seconds,
                      "seconds": seconds, "summary": spread_summary(seconds),
                      "temp_size_in_bytes": compiled.memory_analysis().temp_size_in_bytes,
                      "card": card, "other_processes_before": before, "other_processes_after": after,
                      "repeat_needed": bool(before or after)}
            if reference is not None:
                # Largest differences against the reference form, over the whole rollout.
                expected = np.asarray(reference, np.float64)
                actual = np.asarray(result, np.float64)
                difference = np.abs(actual - expected)
                record.update(largest_absolute_difference=float(difference.max()),
                              largest_relative_difference=float((difference / np.maximum(np.abs(expected), 1e-300)).max()),
                              bit_equal=bool(np.array_equal(actual, expected)),
                              bonus_mean=float(expected.mean()))
            self.write(record)
            return result

        for group in groups:
            reference = None
            for name, form_options in forms:
                form = bonus_form(form_options, config.num_steps, group)
                result = timed(f"post-rollout pass, sharing group {group}, form {name}",
                               lambda h, second, w, g=group, f=form: bonus_of_form(
                                   fast_last_layer_bonus, h, second, w, steps, g, f),
                               hidden, errors if form.plain_errors else targets, layer, reference=reference,
                               group=group, form=name, form_options=form_options, chunk_steps=form.chunk_steps)
                if reference is None:
                    reference = result
        from_features = timed("bonus profile: the predictor's own bonus from the stored features",
                              lambda h, y, w: predictor_bonus_from_features(h, y, w, config), hidden, targets, layer)
        # With plain errors stored, the same bonus is half their squared sum (finish_rollout).
        timed("bonus profile: the predictor's own bonus from the stored plain errors",
              lambda e: jnp.sum(e ** 2, axis=-1) * 0.5, errors, reference=from_features)
        timed("bonus profile: row alignment", row_alignment, hidden)

    def rollout(self, name):
        """Compile, warm twice and time the rollout's time loop of one variant."""
        jax = self.jax
        name, learner_change, environment_change = named_variant(name, self.args.copies)
        if self.remaining() < self.args.expected_rollout_seconds:
            self.write({"record": "rollout skipped", "name": name, "reason": "time budget"})
            return
        trainer = self.trainer(learner_change, environment_change)
        options = {key: {"true": True, "false": False}.get(value, value)
                   for key, value in trainer.config.iteration_compiler_options}
        began = time.perf_counter()
        compiled = jax.jit(trainer.collect_steps, **({"compiler_options": options} if options else {})).lower(
            self.state).compile()
        compile_seconds = time.perf_counter() - began
        for _ in range(2):
            jax.block_until_ready(compiled(self.state))
        card, before = other_processes()
        seconds = []
        for _ in range(self.args.part_repeats):
            began = time.perf_counter()
            jax.block_until_ready(compiled(self.state))
            seconds.append(time.perf_counter() - began)
        _, after = other_processes()
        self.write({"record": "rollout", "name": name, "learner_change": learner_change,
                    "environment_change": environment_change, "compile_seconds": compile_seconds,
                    "seconds": seconds, "summary": spread_summary(seconds),
                    "temp_size_in_bytes": compiled.memory_analysis().temp_size_in_bytes, "card": card,
                    "other_processes_before": before, "other_processes_after": after,
                    "repeat_needed": bool(before or after)})

    def compare(self, variant, forms, updates):
        """Each bonus form beside the reference form on the stored features of real rollouts, update after update."""
        jax = self.jax
        import numpy as np
        from exploration_platform.atari_pixels.networks import dense
        from exploration_platform.atari_pixels.training import fast_last_layer_bonus
        name, learner_change, environment_change = named_variant(variant, self.args.copies)
        trainer = self.trainer(learner_change, environment_change)
        config = trainer.config
        if not trainer.fast_last_layer or trainer.fast_layer_form.chunk_steps:
            raise SystemExit("--compare-variant must be a fast-last-layer variant in the reference form "
                             "(a name ending in _reference-form)")
        steps, group = config.fast_last_layer_steps, config.fast_last_layer_sharing_group
        chosen = [(label, bonus_form(options, config.num_steps, group)) for label, options in forms[1:]]

        def compared_update(state):
            """One reference update, returning the reference bonus and every form's bonus of its rollout."""
            next_state, rollout = trainer.collect_steps(state)
            layer = state.parameters["predictor"]["dense"][-1]
            hidden, targets = rollout.bonus_hidden, rollout.bonus_targets
            # The plain errors the rollout of a plain-error form stores, from the same features.
            errors = dense(layer, hidden, config.dense_batching) - targets
            reference = fast_last_layer_bonus(hidden, targets, layer, steps, group)
            others = {label: bonus_of_form(fast_last_layer_bonus, hidden, errors if form.plain_errors else targets,
                                           layer, steps, group, form) for label, form in chosen}
            rollout = trainer.finish_rollout(state, rollout)
            next_state, batch = trainer.prepare_batch(next_state, rollout)
            next_state, _ = trainer.update(next_state, batch)
            return next_state._replace(iteration=state.iteration + 1), reference, others

        options = {key: {"true": True, "false": False}.get(value, value) for key, value in config.iteration_compiler_options}
        began = time.perf_counter()
        compiled = jax.jit(compared_update, **({"compiler_options": options} if options else {})).lower(self.state).compile()
        self.write({"record": "comparison compiled", "variant": name, "seconds": time.perf_counter() - began,
                    "forms": {label: form._asdict() for label, form in chosen}})
        state, overall = self.state, {label: {"absolute": 0.0, "relative": 0.0, "unequal": 0, "count": 0}
                                      for label, _ in chosen}
        for update in range(1, updates + 1):
            if self.remaining() < 10:
                self.write({"record": "comparison stopped", "reason": "time budget", "updates_done": update - 1})
                break
            state, reference, others = compiled(state)
            expected = np.asarray(reference, np.float64)
            per_update = {}
            for label, value in jax.device_get(others).items():
                difference = np.abs(np.asarray(value, np.float64) - expected)
                relative = difference / np.maximum(np.abs(expected), 1e-300)
                per_update[label] = {"largest_absolute_difference": float(difference.max()),
                                     "largest_relative_difference": float(relative.max()),
                                     "transitions_not_bit_equal": int((difference > 0).sum())}
                total = overall[label]
                total["absolute"] = max(total["absolute"], float(difference.max()))
                total["relative"] = max(total["relative"], float(relative.max()))
                total["unequal"] += int((difference > 0).sum())
                total["count"] += difference.size
            self.write({"record": "compared update", "update": update, "bonus_mean": float(expected.mean()),
                        "bonus_smallest": float(expected.min()), "bonus_largest": float(expected.max()),
                        "forms": per_update})
        self.write({"record": "comparison over all updates", "variant": name,
                    "forms": {label: {"largest_absolute_difference": total["absolute"],
                                      "largest_relative_difference": total["relative"],
                                      "transitions_not_bit_equal": total["unequal"], "transitions": total["count"]}
                              for label, total in overall.items()}})


def bonus_forms(text):
    """Bonus forms to time: 'reference' first, then each JSON object {"name": ..., "options": {...}}."""
    # before: '[{"name": "whole rollout", "options": {"chunk_rows": 512}}]'
    # after: [("reference", {}), ("whole rollout", {"chunk_rows": 512})]
    forms = [("reference", {})]
    for value in json.loads(text):
        forms.append((value["name"], value["options"]))
    return forms


def bonus_form(options, steps, group):
    """The FastLayerForm of a form's learner-style options for one sharing group.

    before: {"chunk_rows": 512, "plain_errors": true}, 128 steps, group 16
    after: FastLayerForm(chunk_steps=32, plain_errors=True, ...); whole_rollout_rows defaults to 0
    """
    from exploration_platform.atari_pixels.fast_layer_forms import FastLayerForm, chunk_steps_for
    from exploration_platform.atari_pixels.fast_layer_forms import small_solve_for
    unknown = set(options) - {"chunk_rows", "whole_rollout_rows", "plain_errors", "float32_products", "small_solve"}
    if unknown:
        raise ValueError(f"unknown bonus form options: {sorted(unknown)}")
    rows = options.get("chunk_rows", 0)
    whole = options.get("whole_rollout_rows", 0)
    return FastLayerForm(chunk_steps=chunk_steps_for(rows, whole, steps, group) if rows else 0,
                         plain_errors=options.get("plain_errors", False),
                         float32_products=options.get("float32_products", "none"),
                         small_solve=small_solve_for(options.get("small_solve", "cholesky"), group))


def bonus_of_form(function, hidden, second, layer, steps, group, form):
    """fast_last_layer_bonus of one form; second is the plain errors for a plain-error form, else the targets."""
    if form.plain_errors:
        return function(hidden, None, layer, steps, group, form, second)
    return function(hidden, second, layer, steps, group, form)


def main():
    """Run every requested variant and part within the time budget."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--card", choices=sorted(CARD_RUNS), required=True,
                        help="h100: the archived four-copy settings; a4500: the archived one-copy settings")
    parser.add_argument("--copies", type=int, required=True)
    parser.add_argument("--seeds", required=True, help="one seed per copy, comma separated")
    parser.add_argument("--variant", action="append", default=[], help="a named variant or a JSON object")
    parser.add_argument("--parts", default="", help="sharing groups whose post-rollout pass is timed alone, e.g. 128,16,4")
    parser.add_argument("--bonus-forms", default="[]", help="JSON list of bonus forms timed against the reference")
    parser.add_argument("--timed-updates", type=int, default=6)
    parser.add_argument("--part-repeats", type=int, default=7)
    parser.add_argument("--parts-seed", type=int, default=0)
    parser.add_argument("--observation-warmup-iterations", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=540)
    parser.add_argument("--expected-variant-seconds", type=float, default=170,
                        help="a variant starts only with at least this many seconds of budget left")
    parser.add_argument("--expected-part-seconds", type=float, default=30)
    parser.add_argument("--compilation-cache", type=Path, default=None)
    parser.add_argument("--rollout", action="append", default=[], help="a named variant whose rollout loop is timed alone")
    parser.add_argument("--expected-rollout-seconds", type=float, default=120)
    parser.add_argument("--compare-variant", default="", help="a fast-last-layer variant in the reference form")
    parser.add_argument("--compare-updates", type=int, default=0)
    parser.add_argument("--initial-checkpoint", type=Path, default=None,
                        help="start every variant from this checkpoint.pkl's state instead of a new warmed state")
    parser.add_argument("--parts-only", action="store_true",
                        help="build no game: only --parts can run (the layer is the seeds' initial output layer)")
    args = parser.parse_args()
    profile = Profile(args)
    for name in args.variant:
        profile.variant(name)
    for name in args.rollout:
        profile.rollout(name)
    if args.parts:
        profile.parts([int(value) for value in args.parts.split(",")], bonus_forms(args.bonus_forms))
    if args.compare_updates:
        profile.compare(args.compare_variant, bonus_forms(args.bonus_forms), args.compare_updates)
    profile.write({"record": "process end"})


if __name__ == "__main__":
    main()

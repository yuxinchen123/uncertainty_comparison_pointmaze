"""Produce common-wall A4500 worker records without changing the numerical learner.

Only explicit future staged plans are accepted. This module supplies metadata
control and bound-worker measurement APIs; it never submits jobs, selects a
configuration, rewrites CUDA selectors, or admits final/portable execution.
"""
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time

import uva_group_binding as group
import configured_scaling_evidence as evidence

selected = group.adjacent("measure_selected")
configured = group.adjacent("configured_measurement")
MAX_MESSAGE_BYTES = 64 * 2 ** 20


def require(value, message):
    """Reject incomplete worker control or measurement state."""
    group.require(value, message)



def helper_names():
    """Retain all existing shared helpers and every explicit new group dependency."""
    return tuple(dict.fromkeys((*selected.HELPERS, *configured.helper_names(), *group.NEW_HELPERS)))

def step_command(plan, job_id, role, argv):
    """Build an explicit future step request without launching or inferring devices."""
    options = group.validate_plan(plan)
    require(role in {"coordinator", "worker"} and type(job_id) is str and job_id.isdecimal()
        and int(job_id) > 0 and isinstance(argv, list) and argv and all(type(x) is str for x in argv),
        "step commands need an actual owned job, role and literal argv")
    owned = group.owned_inputs(plan)
    require(job_id == owned["job_id"], "step request differs from the durably owned submitted job")
    shape = options["resources"][role]
    # Installed srun documents --gres=none; actual step TRES must still prove zero.
    resource = "--gres=none" if role == "coordinator" else "--gres=gpu:nvidia_rtx_a4500:1"
    return ["srun", "--jobid=" + job_id, "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
        "--cpu-bind=cores", "--kill-on-bad-exit=1",
        "--cpus-per-task=" + str(shape["cpus"]), "--mem=" + str(shape["host_memory_mib"]) + "M",
        resource, *argv]


def begin_group(plan, coordinator):
    """Publish one actual owned coordinator attempt before any worker starts."""
    group.validate_plan(plan)
    group.validate_metadata(plan, coordinator)
    require(coordinator["role"] == "coordinator"
        and not ({"jax", "jaxlib", "numpy", "torch"} & {name.split(".")[0] for name in sys.modules}),
        "coordinator must remain a zero-GPU standard-library process")
    group.current_process(plan, coordinator)
    directory = group.unit_directory(plan)
    # This token uses operating-system entropy for message identity, not learner randomness.
    value = {"schema_version": 1, "kind": "owned_a4500_group_control_attempt",
        "plan": group.receipt(directory / "scaling_plan.json"),
        "coordinator": group.receipt(directory / "group_step_metadata.json"),
        "launch_token": secrets.token_hex(32), "clock_epoch_origin": time.time() - time.monotonic(),
        "started_at_seconds": time.time(), "physical_identities_resolved": False,
        "coordinator_has_gpu": False, "automatic_retry_permitted": False}
    group.publish(directory / "group_control_attempt.json", value)
    (directory / "workers").mkdir(mode=0o700)
    for index in range(plan["gpu_count"]):
        worker = group.worker_directory(plan, index)
        worker.mkdir(mode=0o700)
        (worker / "from_coordinator").mkdir(mode=0o700)
        (worker / "from_worker").mkdir(mode=0o700)
    return value


def message_path(plan, index, direction, phase, round_index):
    """Name one immutable phase message without an inherited descriptor."""
    require(direction in {"from_coordinator", "from_worker"}
        and phase in {"initialize", "bound", "group_resolved", "compiled", "ready", "release",
            "gpu_complete", "collect", "round_record", "finished", "work_permitted"}
        and type(round_index) is int and 0 <= round_index <= 7, "invalid control message name")
    return group.worker_directory(plan, index) / direction / (str(round_index) + "_" + phase + ".json")


def publish_message(plan, attempt, index, direction, phase, round_index, payload, process):
    """Write exactly one bounded source-bound message from the actual own process."""
    require(attempt["plan"] == group.receipt(group.unit_directory(plan) / "scaling_plan.json")
        and type(attempt["launch_token"]) is str and len(attempt["launch_token"]) == 64,
        "group attempt or plan changed")
    actual = group.adjacent("device_memory_limit").process_identity()
    require(group.exact(actual, process), "message writer process identity changed")
    metadata_path = (group.unit_directory(plan) if direction == "from_coordinator"
        else group.worker_directory(plan, index)) / "group_step_metadata.json"
    metadata = group.read_receipt(group.receipt(metadata_path))
    group.current_process(plan, metadata)
    require(group.exact(metadata["process"], process) and metadata["role"] ==
        ("coordinator" if direction == "from_coordinator" else "worker"), "message role is not this actual owned step")
    value = {"schema_version": 1, "kind": "owned_a4500_group_control_message",
        "attempt": group.receipt(group.unit_directory(plan) / "group_control_attempt.json"),
        "launch_token": attempt["launch_token"], "worker_index": index, "direction": direction,
        "phase": phase, "round": round_index, "process": process, "metadata": group.receipt(metadata_path),
        "written_at_monotonic": time.monotonic(), "payload": payload}
    require(len(group.encoded(value)) <= MAX_MESSAGE_BYTES, "group control message exceeds its bound")
    path = message_path(plan, index, direction, phase, round_index)
    temporary = path.with_name(path.name + ".prepared")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(group.encoded(value)); stream.flush(); os.fsync(stream.fileno())
    # A no-replace hard link exposes only complete fsynced bytes to the reader.
    os.link(temporary, path, follow_symlinks=False)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
        temporary.unlink()
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return group.receipt(path)


def read_message(plan, attempt, index, direction, phase, round_index, timeout):
    """Wait for one exact record and reject substitution, outsiders or replay."""
    require(type(timeout) in {int, float} and math.isfinite(timeout) and timeout > 0,
        "message wait requires a finite positive timeout")
    path = message_path(plan, index, direction, phase, round_index)
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("group control message was not published before its deadline")
        time.sleep(min(.01, max(0, deadline - time.monotonic())))
    require(path.stat().st_size <= MAX_MESSAGE_BYTES and path.stat().st_uid == 2618919,
        "control message is not a bounded ordinary owned file")
    value = group.read_receipt(group.receipt(path))
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "owned_a4500_group_control_message"
        and value.get("attempt") == group.receipt(group.unit_directory(plan) / "group_control_attempt.json")
        and value.get("launch_token") == attempt["launch_token"] and value.get("worker_index") == index
        and value.get("direction") == direction and value.get("phase") == phase and value.get("round") == round_index
        and type(value.get("written_at_monotonic")) in {int, float}
        and math.isfinite(value["written_at_monotonic"]), "wrong attempt, sender, sequence or malformed control frame")
    metadata = group.read_receipt(value["metadata"])
    if direction == "from_coordinator":
        require(value["metadata"] == attempt["coordinator"], "coordinator metadata changed after attempt began")
    elif phase == "bound":
        group.validate_metadata(plan, metadata)
    else:
        resolved = group.read_receipt(group.receipt(group.unit_directory(plan) / "resolved_group_binding.json"))
        require(value["metadata"] == resolved["members"][index]["metadata"],
            "worker metadata differs from its independently resolved group member")
    require(group.exact(value["process"], metadata["process"]) and metadata["role"] ==
        ("coordinator" if direction == "from_coordinator" else "worker"), "control message process or actual step changed")
    expected = (group.unit_directory(plan) if direction == "from_coordinator"
        else group.worker_directory(plan, index)) / "group_step_metadata.json"
    require(value["metadata"] == group.receipt(expected), "control message used another worker's metadata")
    return value["payload"]



def collect_phase(plan, attempt, phase, number, timeout):
    """Bound the entire multi-worker barrier by one shared monotonic deadline."""
    require(type(timeout) in {int, float} and math.isfinite(timeout) and timeout > 0,
        "group barrier needs a finite positive timeout")
    deadline = time.monotonic() + timeout
    rows = []
    for index in range(plan["gpu_count"]):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the complete multi-worker barrier exceeded its common deadline")
        rows.append(read_message(plan, attempt, index, "from_worker", phase, number, remaining))
    return rows

def group_clock(attempt):
    """Project this same host's monotonic clock onto one recorded origin."""
    return attempt["clock_epoch_origin"] + time.monotonic()


class GroupMemoryObservation(configured.MemoryObservation):
    """Reuse the unchanged assigned-only sampler after explicit group binding."""
    def __init__(self, plan, metadata, binding, device, directory):
        """Require actual saved group binding before the first assigned-device query."""
        self.group_plan, self.group_metadata = plan, metadata
        worker = Path(directory).parent
        require(Path(directory).name == "group_memory", "group memory uses its reserved worker subdirectory")
        group.validate_binding(plan, metadata, binding, worker)
        require(group.read_receipt(group.receipt(worker / "resolved_device_binding.json")) == binding,
            "memory sampler cannot use a remembered or different binding")
        group.current_process(plan, metadata)
        self.policy, self.device, self.binding, self.directory = "uva_a4500", device, binding, Path(directory)
        self.uuid, self.pid = binding["gpu_uuid"], os.getpid()
        self.stop, self.thread, self.counter = threading.Event(), None, 0

    def begin(self):
        """Recheck own process and source-bound device before each measured window."""
        group.current_process(self.group_plan, self.group_metadata)
        group.validate_binding(self.group_plan, self.group_metadata, self.binding, self.directory.parent)
        return super().begin()

    def finish(self):
        """Retain the unchanged complete sample proof and actual final own placement."""
        value = super().finish()
        group.current_process(self.group_plan, self.group_metadata)
        return value


def measure_worker_windows(plan, index, metadata, binding, resolved, attempt, executor, host,
        paired, jax, device, trainer):
    """Run the unchanged selected Executor for seven complete update windows."""
    configuration = evidence.validate_protocol(plan)
    directory = group.worker_directory(plan, index)
    group.validate_binding(plan, metadata, binding, directory)
    require(group.exact(group.read_receipt(group.receipt(group.unit_directory(plan) /
        "resolved_group_binding.json")), resolved), "worker cannot start before immutable full group resolution")
    evidence.verify_target_qualifications(plan, resolved)
    evidence.verify_completed_phase(plan, group_clock(attempt), resolved)
    require(configuration == plan["configuration"] and trainer.config.n_copies ==
        configuration["learner"]["n_copies"] and type(int(host.iteration)) is int
        and int(host.iteration) > 0 and int(host.optimizer.step) == int(host.iteration) * 16,
        "worker must retain the actual completed compilation-warmup origin and optimizer counters")
    source = Path(plan["source_root"]) / "10_jax_exploration_platform"
    selected.verify_imported_sources(source, paired)
    require(Path(sys.modules[type(executor).__module__].__file__).absolute() ==
        source / "scripts/atari_pixel_campaign/paired_benchmark.py",
        "worker must use the actual selected source Executor")
    memory = GroupMemoryObservation(plan, metadata, binding, device, directory / "group_memory")
    rows = []
    before_target = paired.host_copy(host.target)
    training_origin = int(host.iteration)
    for number in range(1, 8):
        # Compilation and normalization warmup are complete before these shared timed intervals.
        origin = int(host.iteration)
        state = paired.device_copy(host)
        jax.block_until_ready(state)
        memory.begin()
        try:
            publish_message(plan, attempt, index, "from_worker", "ready", number,
                {"iteration": origin}, metadata["process"])
            require(read_message(plan, attempt, index, "from_coordinator", "release", number,
                plan["barrier_timeout_seconds"]) == {"release": True}, "different group round released")
            start = group_clock(attempt)
            state, metrics = executor.run(state, plan["iterations"], initial_iteration=origin)
            jax.block_until_ready((state, metrics))
            end = group_clock(attempt)
            timing = {"start_time_seconds": start, "end_time_seconds": end, "seconds": end - start}
            publish_message(plan, attempt, index, "from_worker", "gpu_complete", number, timing, metadata["process"])
            require(read_message(plan, attempt, index, "from_coordinator", "collect", number,
                plan["barrier_timeout_seconds"]) == {"collect": True}, "different group collection request")
        finally:
            memory_record = memory.finish()
        host = paired.host_copy(state)
        paired.require_finite((host, jax.device_get(metrics)), "configured A4500 group full round")
        # Frozen target comparison reads actual state, without altering policy or random streams.
        target = host.target
        require(jax.tree.structure(before_target) == jax.tree.structure(target)
            and all(a.shape == b.shape and a.dtype == b.dtype and (a == b).all()
                for a, b in zip(jax.tree.leaves(before_target), jax.tree.leaves(target))),
            "the original frozen target changed during full training updates")
        require(int(host.iteration) == origin + plan["iterations"]
            and int(host.optimizer.step) == (origin + plan["iterations"]) * 16,
            "actual full-update or sixteen-minibatch optimizer counters changed")
        groups = {"parameters": host.parameters, "target": host.target, "optimizer_first": host.optimizer.first,
            "optimizer_second": host.optimizer.second, "observation_statistics": host.observation_statistics,
            "reward_statistics": host.reward_statistics}
        dtypes = {name: sorted({str(leaf.dtype) for leaf in jax.tree.leaves(tree)}) for name, tree in groups.items()}
        require(dtypes == configured.EXPECTED_DTYPES, "actual network/Adam/statistics precision changed")
        transitions = plan["iterations"] * trainer.config.batch_size * trainer.config.n_copies
        row = {"round": number, **timing, "gpu_uuid": binding["gpu_uuid"],
            "pid": metadata["process"]["pid"], "process_start_time": metadata["process"]["process_start_time"],
            "step_id": metadata["environment"]["SLURM_STEP_ID"], "copy_seeds": plan["copy_seeds"][index],
            "copies": trainer.config.n_copies, "iterations": plan["iterations"], "environments_per_copy": 128,
            "rollout_steps": 128, "training_transitions": transitions, "report_origin_iteration": origin,
            "actual_end_iteration": int(host.iteration), "actual_optimizer_step": int(host.optimizer.step),
            "training_start_iteration": training_origin,
            "dtypes": dtypes,
            "transitions_per_second_total": transitions / timing["seconds"], "memory": memory_record,
            "numerical_state_finite": True, "frozen_target_unchanged": True,
            "published_reports": jax.tree.map(lambda value: value.tolist() if hasattr(value, "tolist") else value,
                executor.last_reports)}
        rows.append(row)
        group.publish(directory / ("round_" + str(number) + ".json"), row)
        publish_message(plan, attempt, index, "from_worker", "round_record", number, row, metadata["process"])
        del state, metrics
    return host, rows


def collect_group_windows(plan, coordinator, resolved, attempt):
    """Release all bound workers together and close the shared interval at GPU completion."""
    evidence.validate_protocol(plan)
    group.validate_metadata(plan, coordinator)
    group.current_process(plan, coordinator)
    require(coordinator["role"] == "coordinator", "group timing belongs to the actual zero-GPU coordinator")
    timeout, rounds = plan["barrier_timeout_seconds"], []
    for number in range(1, 8):
        ready = collect_phase(plan, attempt, "ready", number, timeout)
        require(len({row["iteration"] for row in ready}) == 1, "workers have different schedule origins")
        start = group_clock(attempt)
        for index in range(plan["gpu_count"]):
            publish_message(plan, attempt, index, "from_coordinator", "release", number,
                {"release": True}, coordinator["process"])
        completed = collect_phase(plan, attempt, "gpu_complete", number, timeout)
        end = group_clock(attempt)
        for index in range(plan["gpu_count"]):
            publish_message(plan, attempt, index, "from_coordinator", "collect", number,
                {"collect": True}, coordinator["process"])
        rows = collect_phase(plan, attempt, "round_record", number, timeout)
        require(all(actual == {name: row[name] for name in
            ("start_time_seconds", "end_time_seconds", "seconds")} for actual, row in zip(completed, rows)),
            "worker GPU-complete timing changed after common interval closed")
        value = evidence.aggregate_round(plan, resolved, rows, start, end, number)
        group.publish(group.unit_directory(plan) / ("group_round_" + str(number) + ".json"), value)
        rounds.append(value)
    return rounds


def refuse_unqualified_entrypoint():
    """Keep this new library out of existing workers until separately reviewed and qualified."""
    raise ValueError("configured group producer APIs require a separately reviewed staged launcher and actual group qualification; default, final and portable execution are not admitted")


if __name__ == "__main__":
    refuse_unqualified_entrypoint()

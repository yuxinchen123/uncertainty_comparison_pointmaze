"""Bind a direct capture's actual release bytes to its child's pre-work acknowledgment."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

PROTOCOL = "direct-h100-owned-nsight-and-retained-descriptor-version-1"
DIAGNOSTIC = "direct_nsight_process_descriptor_diagnostic"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
GPU = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
MAX_METADATA_BYTES = 8 * 1024 * 1024
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
LOCK_FIELDS = ("path", "fd", "fd_stat", "path_stat", "path_symlink", "fd_target", "fdinfo", "fdinfo_sha256")
RELEASE_KEYS = {"schema_version", "kind", "protocol", "challenge", "plan", "source", "intent", "claim", "observation",
    "producer_lock", "producer", "capture_process", "released_at_seconds", "prework_only"}
ACK_KEYS = {"schema_version", "kind", "protocol", "challenge", "plan", "source", "intent", "claim", "observation",
    "producer_lock", "producer", "capture_process", "release", "release_bytes", "release_size_bytes", "observed_at_seconds",
    "numerical_modules", "environment_work_started"}


def require(value, message):
    """Reject missing or changed release evidence without creating substitute records."""
    if not value:
        raise ValueError(message)


def digest(raw):
    """Hash one exact retained byte sequence."""
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    """Encode finite metadata with unambiguous key order."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def timestamp(value, name):
    """Require an actual finite nonnegative time rather than a Boolean or unknown value."""
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, name + " must be a finite actual timestamp")
    return value


def read_bytes(path):
    """Read one bounded ordinary file through a pinned descriptor and reject replacement."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
        "release evidence must use an exact ordinary absolute path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_METADATA_BYTES,
            "release evidence must be a nonempty bounded regular file")
        raw = os.read(descriptor, MAX_METADATA_BYTES + 1)
        after, named = os.fstat(descriptor), path.lstat()
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key) == getattr(named, key)
            for key in fields), "release evidence changed during the pinned read")
    finally:
        os.close(descriptor)
    return raw


def finite_float(token):
    """Reject overflowed JSON floating-point values while retaining finite signed metadata."""
    value = float(token)
    require(math.isfinite(value), "nonfinite release metadata")
    return value


def document(raw):
    """Decode strict unique-key finite JSON from an already pinned byte sequence."""
    def unique(pairs):
        """Keep exactly one value for each recorded field."""
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate release metadata key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique, parse_float=finite_float,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite release metadata")))


def file_record(path):
    """Return the actual reference and bytes from one descriptor read."""
    raw = read_bytes(path)
    return {"path": str(path), "sha256": digest(raw)}, raw


def reserved(value, directory, name):
    """Check a reserved metadata path before any caller opens a supplied reference."""
    expected = Path(directory) / name
    require(type(value) is dict and set(value) == {"path", "sha256"}
        and value.get("path") == str(expected) and type(value.get("sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "release reference differs from its reserved metadata path")
    return expected


def checked(value, directory, name):
    """Read only one exact reserved metadata reference and return its decoded content."""
    path = reserved(value, directory, name)
    actual, raw = file_record(path)
    require(actual == value, "reserved release metadata bytes changed")
    return document(raw)


def unit(directory, plan):
    """Limit release records to the declared direct capture or separate diagnostic unit."""
    directory = Path(directory)
    require(directory.is_absolute() and directory.resolve(strict=True) == directory
        and directory == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and plan.get("capture_process_protocol") == PROTOCOL and plan.get("gpu_uuid") == GPU
        and plan.get("hardware_class") == "h100" and plan.get("kind") in {DIAGNOSTIC, "selected_configuration_profile"}
        and "configured_isolation" not in plan
        and (plan["kind"] == DIAGNOSTIC or (plan.get("direct_profile_protocol") == "direct-h100-reviewed-common-profile-version-1"
            and "direct_binding" in plan)),
        "release evidence is only for the declared direct capture unit")
    return directory


def actors(intent, claim, observer):
    """Validate the two exact own process snapshots without inspecting historical PIDs."""
    producer, capture = intent["producer"], claim["process"]
    uids = producer["uids"]
    require(type(uids) is list and len(uids) == 4 and all(type(value) is int and value == uids[0] for value in uids),
        "the release producer lacks four exact owner identifiers")
    observer.validate_snapshot(producer, uids[0], producer["cgroup"])
    observer.validate_snapshot(capture, uids[0], producer["cgroup"])
    require(type(claim.get("pid")) is int and claim["pid"] > 0 and claim["pid"] == capture["pid"]
        and claim["process_start_time"] == capture["process_start_time"] and claim["pid"] != producer["pid"]
        and capture["command"] == intent["capture_command"], "the actual release child changed identity or source command")
    return observer.stable(producer), observer.stable(capture)


def basis(directory, plan, intent, claim, observation_reference, producer_lock_reference, observer):
    """Rebuild the exact retained plan, source, actors and pre-release producer custody."""
    directory = unit(directory, plan)
    plan_ref, plan_raw = file_record(directory / "measurement_plan.json")
    require(encoded(document(plan_raw)) == encoded(plan) and intent["plan"] == claim["plan"] == plan_ref,
        "the release plan differs from the actual immutable unit")
    entry = "diagnose_direct_nsight_process.py" if plan["kind"] == DIAGNOSTIC else "profile_selected.py"
    source = reserved(intent["source"], directory, "code/" + entry)
    source_ref, unused = file_record(source)
    require(source_ref == intent["source"] == claim["source"] and source_ref["sha256"] == plan["helpers"]["files"].get(PREFIX + entry),
        "the released source differs from its actual staged bytes")
    intent_ref, intent_raw = file_record(directory / "direct_profile_process_intent.json")
    claim_ref, claim_raw = file_record(directory / "capture_identity.json")
    require(encoded(document(intent_raw)) == encoded(intent) and encoded(document(claim_raw)) == encoded(claim)
        and claim["intent"] == intent_ref and type(intent.get("challenge")) is str
        and re.fullmatch(r"[0-9a-f]{64}", intent["challenge"]) and claim["challenge"] == intent["challenge"],
        "the retained source claim belongs to another release attempt")
    observation = checked(observation_reference, directory, "capture_process_observation.json")
    custody = checked(producer_lock_reference, directory, "producer_lock_before_release.json")
    producer, capture = actors(intent, claim, observer)
    observer.validate_observation(observation, claim, intent["capture_command"], intent["producer"])
    require(type(observation.get("capture_pid")) is int and observation["capture_pid"] == claim["pid"]
        and observer.stable(observation["parent"]) == producer,
        "the released process observation belongs to another producer or child")
    initial = intent["producer_lock_before"]
    require(custody.get("local_to_producer") is True and type(custody.get("read_by_pid")) is int
        and custody["read_by_pid"] == intent["producer"]["pid"]
        and observer.stable(custody["producer"]) == producer
        and all(encoded(custody[key]) == encoded(initial[key]) for key in LOCK_FIELDS),
        "the exact sole producer's descriptor changed before release")
    times = [timestamp(intent.get("created_at_seconds"), "intent creation"),
        timestamp(claim.get("claimed_at_seconds"), "child claim"),
        timestamp(observation.get("observed_at_seconds"), "completed process observation"),
        timestamp(custody.get("observed_at_seconds"), "pre-release producer custody")]
    require(times == sorted(times), "the pre-release process or custody chronology changed")
    # Before: exact files and raw process snapshots; after: immutable references plus stable actor identities.
    return {"plan": plan_ref, "source": source_ref, "intent": intent_ref, "claim": claim_ref,
        "observation": observation_reference, "producer_lock": producer_lock_reference,
        "producer": producer, "capture_process": capture}, times


def validate_release(release, bindings, times, challenge):
    """Require one complete release joining the exact pre-work identities and references."""
    require(type(release) is dict and set(release) == RELEASE_KEYS and type(release.get("schema_version")) is int
        and release["schema_version"] == 1 and release.get("kind") == "direct_capture_prework_release"
        and release.get("protocol") == PROTOCOL and release.get("challenge") == challenge and release.get("prework_only") is True,
        "the actual capture release is missing or uses another protocol")
    require(all(encoded(release.get(key)) == encoded(value) for key, value in bindings.items()),
        "the actual capture release changed plan, source, claim, observation or producer custody")
    released = timestamp(release.get("released_at_seconds"), "actual capture release")
    require(times[-1] <= released, "the actual capture release predates completed process observation or custody")
    return released


def publish_bytes(path, raw):
    """Publish one fsynced new immutable artifact without replacing prior or partial bytes."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "release publication parent must be ordinary")
    pending = path.with_name("." + path.name + ".pending")
    require(not os.path.lexists(path) and not os.path.lexists(pending), "preserve prior complete or partial release evidence")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path, follow_symlinks=False)
    pending.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(path), "sha256": digest(raw)}


def publish_release(directory, plan, intent, claim, observation_reference, producer_lock_reference, observer):
    """Release one actual blocked child after its positive observation and local producer custody."""
    directory = unit(directory, plan)
    bindings, times = basis(directory, plan, intent, claim, observation_reference, producer_lock_reference, observer)
    require(os.getpid() == intent["producer"]["pid"], "only the actual sole producer may publish a capture release")
    release = {"schema_version": 1, "kind": "direct_capture_prework_release", "protocol": PROTOCOL,
        "challenge": intent["challenge"], **bindings, "released_at_seconds": time.time(), "prework_only": True}
    validate_release(release, bindings, times, intent["challenge"])
    return publish_bytes(directory / "capture_release.json", encoded(release))


def acknowledge_release(directory, plan, intent, claim, actual_process, observer):
    """Retain exact observed release bytes before this actual child imports or starts payload work."""
    directory = unit(directory, plan)
    require(os.getpid() == claim["pid"] and not any(name.split(".")[0] in NUMERICAL for name in sys.modules),
        "the actual child must acknowledge release before any numerical import")
    release_ref, release_raw = file_record(directory / "capture_release.json")
    release = document(release_raw)
    bindings, times = basis(directory, plan, intent, claim, release["observation"], release["producer_lock"], observer)
    released = validate_release(release, bindings, times, intent["challenge"])
    observer.validate_snapshot(actual_process, intent["producer"]["uids"][0], intent["producer"]["cgroup"])
    require(observer.stable(actual_process) == bindings["capture_process"], "the actual acknowledgment process changed child identity")
    observed = timestamp(time.time(), "child release observation")
    require(released <= observed, "the child acknowledgment predates its actual release")
    # Keep the exact observed bytes separately; parsing or re-encoding cannot substitute another release.
    raw_ref = publish_bytes(directory / "capture_release_observed.raw", release_raw)
    acknowledgment = {"schema_version": 1, "kind": "actual_direct_capture_prework_release_observation",
        "protocol": PROTOCOL, "challenge": intent["challenge"], **bindings, "release": release_ref,
        "release_bytes": raw_ref, "release_size_bytes": len(release_raw), "observed_at_seconds": observed,
        "numerical_modules": [], "environment_work_started": False}
    return publish_bytes(directory / "capture_release_observed.json", encoded(acknowledgment))


def validate_saved_release(directory, plan, intent, claim, observation, observer, work=None):
    """Require the real release and exact child acknowledgment before accepting saved process evidence."""
    directory = unit(directory, plan)
    release_ref, release_raw = file_record(directory / "capture_release.json")
    release = document(release_raw)
    bindings, times = basis(directory, plan, intent, claim, release["observation"], release["producer_lock"], observer)
    actual_observation = checked(bindings["observation"], directory, "capture_process_observation.json")
    require(encoded(actual_observation) == encoded(observation), "saved release received another process observation")
    released = validate_release(release, bindings, times, intent["challenge"])
    acknowledgment_ref, acknowledgment_raw = file_record(directory / "capture_release_observed.json")
    acknowledgment = document(acknowledgment_raw)
    require(type(acknowledgment) is dict and set(acknowledgment) == ACK_KEYS
        and type(acknowledgment.get("schema_version")) is int and acknowledgment["schema_version"] == 1
        and acknowledgment.get("kind") == "actual_direct_capture_prework_release_observation"
        and acknowledgment.get("protocol") == PROTOCOL and acknowledgment.get("challenge") == intent["challenge"]
        and acknowledgment.get("numerical_modules") == [] and acknowledgment.get("environment_work_started") is False,
        "the actual child did not retain a complete pre-work release acknowledgment")
    require(all(encoded(acknowledgment.get(key)) == encoded(value) for key, value in bindings.items())
        and acknowledgment.get("release") == release_ref, "the child acknowledgment changed its actual release or identity joins")
    raw_path = reserved(acknowledgment["release_bytes"], directory, "capture_release_observed.raw")
    observed_raw_ref, observed_raw = file_record(raw_path)
    require(observed_raw_ref == acknowledgment["release_bytes"] and observed_raw == release_raw
        and type(acknowledgment.get("release_size_bytes")) is int and acknowledgment["release_size_bytes"] == len(release_raw),
        "the child's retained release bytes differ from the actual immutable release")
    acknowledged = timestamp(acknowledgment.get("observed_at_seconds"), "saved child release observation")
    require(released <= acknowledged, "the saved child acknowledgment predates its actual release")
    if work is not None:
        require(type(work) is dict and type(work.get("pid")) is int and work["pid"] == claim["pid"]
            and work.get("process_start_time") == claim["process_start_time"], "the released actual work belongs to another process")
        started = timestamp(work.get("started_at_seconds"), "actual payload work start")
        finished = timestamp(work.get("finished_at_seconds"), "actual payload work finish")
        require(acknowledged <= started <= finished, "actual payload work preceded its retained pre-work acknowledgment")
    return {"capture_release": release_ref, "capture_release_observed": acknowledgment_ref,
        "capture_release_observed_bytes": observed_raw_ref, "capture_release_observed_at_seconds": acknowledged,
        "prework_chronology_verified": True, "scientific_credit": 0, "admission_granted": False}

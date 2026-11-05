"""Observe one owned direct Nsight payload while only its producer holds the GPU lock."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

PROTOCOL = "direct-h100-owned-nsight-and-retained-descriptor-version-1"
DIAGNOSTIC = "direct_nsight_process_descriptor_diagnostic"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
GPU = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
LOCK = "/tmp/sl5nw_serval02_card1.lock"
OWNER = 2618919
LOCK_IDENTITY = {"device": 64512, "inode": 23077017, "uid": OWNER, "mode": 420, "regular_file": True}
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
DIAGNOSTIC_BUDGET = {'schema_version': 1, 'work_seconds': 240, 'cleanup_seconds': 60, 'total_seconds': 300}


def require(value, message):
    """Refuse unknown ownership or descriptor evidence before releasing the payload."""
    if not value:
        raise ValueError(message)


def diagnostic_budget(plan):
    """Recognize only the explicit diagnostic time budget; leave absent/default and formal paths unchanged."""
    if "profile_target_process_control" in plan:
        target = adjacent_helper(plan, "profile_target_process")
        target.policy(plan)
        return dict(target.BUDGET)
    if 'diagnostic_time_budget' not in plan:return None
    value = plan['diagnostic_time_budget']
    require(plan.get('kind') == DIAGNOSTIC and type(value) is dict and set(value) == set(DIAGNOSTIC_BUDGET)
        and all(type(value[key]) is int and value[key] == expected for key, expected in DIAGNOSTIC_BUDGET.items())
        and type(plan.get('maximum_active_seconds')) is int and plan['maximum_active_seconds'] == 240,
        'the direct diagnostic requires exactly 240 work, 60 cleanup and 300 total seconds')
    return value


def monotonic_clock_domain():
    """Bind same-host monotonic observations to the actual public kernel boot identifier."""
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', boot), 'the actual clock boot identifier is invalid')
    clock = time.get_clock_info('monotonic')
    require(clock.monotonic is True and clock.adjustable is False, 'the diagnostic clock must be monotonic and nonadjustable')
    return {'hostname': socket.gethostname().split('.')[0], 'boot_id': boot,
        'clock': 'monotonic', 'implementation': clock.implementation}


def finite_monotonic(value):
    """Refuse Boolean, negative and nonfinite monotonic timestamps."""
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class DiagnosticDeadline:
    """Reuse one actual controller origin across all work, roles and cleanup attempts."""
    def __init__(self, origin, reference):
        """Keep immutable origin bytes separate from this process's retained first cleanup observation."""
        self.origin, self.reference = origin, reference
        self.started = origin['started_monotonic']
        self.work_end, self.total_end = self.started + 240, self.started + 300
        self.cleanup_started, self.cleanup_end = None, None

    def remaining(self, phase, *, allow_zero=False):
        """Return remaining time on the original absolute clock, never a restarted duration."""
        require(phase in ('work', 'total', 'cleanup'), 'an exact diagnostic deadline phase is required')
        now = time.monotonic()
        require(finite_monotonic(now) and now >= self.started, 'the diagnostic monotonic clock moved to another domain')
        end = self.work_end if phase == 'work' else self.total_end if phase == 'total' else self.cleanup_end
        require(end is not None, 'cleanup must retain its first actual start before a cleanup wait')
        remaining = max(0., end - now)
        if not allow_zero and remaining <= 0:raise TimeoutError('the shared diagnostic ' + phase + ' deadline expired')
        return remaining

    def begin_cleanup(self):
        """Reserve at most sixty seconds once, capped by the original total deadline."""
        if self.cleanup_started is None:
            self.cleanup_started = time.monotonic()
            require(finite_monotonic(self.cleanup_started) and self.cleanup_started >= self.started,
                'the first diagnostic cleanup clock is invalid')
            self.cleanup_end = min(self.total_end, self.cleanup_started + 60)
        return self.cleanup_end

    def timeout(self, requested=None, *, phase='work'):
        """Cap one requested wait by the common absolute deadline without extending it."""
        remaining = self.remaining(phase)
        require(requested is None or type(requested) in (int, float) and math.isfinite(requested) and requested > 0,
            'a requested diagnostic timeout must be a finite positive number')
        return remaining if requested is None else min(remaining, requested)

    def snapshot(self):
        """Retain actual clock observations even after expiry, without claiming successful completion."""
        return {'origin': self.reference, 'clock_domain': self.origin['clock_domain'],
            'started_monotonic': self.started, 'work_end_monotonic': self.work_end,
            'total_end_monotonic': self.total_end, 'observed_monotonic': time.monotonic(),
            'cleanup_started_monotonic': self.cleanup_started, 'cleanup_end_monotonic': self.cleanup_end}

    def verify_snapshot(self, value, *, phase, cleanup_required=False):
        """Check retained deadline arithmetic and expiry without sampling a live clock during saved review."""
        require(type(value) is dict and set(value) == {'origin', 'clock_domain', 'started_monotonic',
            'work_end_monotonic', 'total_end_monotonic', 'observed_monotonic', 'cleanup_started_monotonic',
            'cleanup_end_monotonic'} and value['origin'] == self.reference
            and value['clock_domain'] == self.origin['clock_domain']
            and all(finite_monotonic(value[key]) for key in ('started_monotonic', 'work_end_monotonic',
                'total_end_monotonic', 'observed_monotonic'))
            and value['started_monotonic'] == self.started and value['work_end_monotonic'] == self.work_end
            and value['total_end_monotonic'] == self.total_end
            and self.started <= value['observed_monotonic'] <= (self.work_end if phase == 'work' else self.total_end),
            'saved diagnostic deadline changed its actual origin, domain or successful time window')
        first, end = value['cleanup_started_monotonic'], value['cleanup_end_monotonic']
        if first is None:
            require(end is None and not cleanup_required, 'successful scope completion needs its actual cleanup window')
        else:
            require(finite_monotonic(first) and finite_monotonic(end) and self.started <= first <= value['observed_monotonic']
                and end == min(self.total_end, first + 60) and value['observed_monotonic'] <= end,
                'saved diagnostic cleanup restarted or exceeded its original reserved window')
        return value


def validate_diagnostic_origin(directory, plan, observer):
    """Read the frozen controller origin without querying a historical live process or native state."""
    if "profile_target_process_control" in plan:
        return adjacent_helper(plan, "profile_target_process").saved_deadline(directory, plan, observer)
    require(diagnostic_budget(plan) is not None, 'the declared diagnostic time budget is required')
    directory = Path(directory)
    reference = observer.receipt(directory / 'diagnostic_controller_time_origin.json')
    origin = observer.read_json(reference['path'])
    expected_command = [plan['python'], '-I', '-S', '-B', str(directory / 'code/diagnose_direct_nsight_process.py'),
        '--controller-unit', str(directory)]
    require(type(origin) is dict and set(origin) == {'kind', 'schema_version', 'plan', 'source', 'controller',
        'clock_domain', 'budget', 'started_monotonic', 'published_monotonic', 'published_at_seconds'}
        and origin['kind'] == 'actual_direct_diagnostic_controller_time_origin'
        and type(origin['schema_version']) is int and origin['schema_version'] == 1
        and origin['plan'] == observer.receipt(directory / 'measurement_plan.json')
        and origin['source'] == observer.receipt(directory / 'code/diagnose_direct_nsight_process.py')
        and origin['source']['sha256'] == plan['helpers']['files'][PREFIX + 'diagnose_direct_nsight_process.py']
        and type(origin['budget']) is dict and set(origin['budget']) == set(DIAGNOSTIC_BUDGET)
        and all(type(origin['budget'][key]) is int and origin['budget'][key] == expected
            for key, expected in DIAGNOSTIC_BUDGET.items()), 'the immutable diagnostic origin changed source, plan or budget')
    observer.validate_snapshot(origin['controller'], OWNER, origin['controller']['cgroup'])
    domain = origin['clock_domain']
    require(origin['controller']['command'] == expected_command and type(domain) is dict
        and set(domain) == {'hostname', 'boot_id', 'clock', 'implementation'} and domain['hostname'] == 'serval02'
        and domain['clock'] == 'monotonic' and type(domain['implementation']) is str
        and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', domain['boot_id'])
        and all(finite_monotonic(origin[key]) for key in ('started_monotonic', 'published_monotonic', 'published_at_seconds'))
        and origin['started_monotonic'] <= origin['published_monotonic'] < origin['started_monotonic'] + 240,
        'the diagnostic controller identity or original clock is invalid')
    return DiagnosticDeadline(origin, reference)


def read_diagnostic_deadline(directory, plan):
    """Require the same actual host and boot clock before a live role uses the shared deadline."""
    if diagnostic_budget(plan) is None:return None
    deadline = validate_diagnostic_origin(directory, plan, generic(plan))
    require(deadline.origin['clock_domain'] == monotonic_clock_domain(), 'the diagnostic role changed host or monotonic domain')
    deadline.remaining('work')
    return deadline


def publish_diagnostic_origin(directory, plan, controller, started):
    """Create and fsync actual parent/source/clock bytes before Popen, with no future child identity."""
    require(diagnostic_budget(plan) is not None, 'a diagnostic origin cannot be published for another protocol')
    now = time.monotonic()
    require(finite_monotonic(started) and started <= now < started + 240, 'actual controller startup exhausted its work deadline')
    observer = generic(plan)
    observer.publish(Path(directory) / 'diagnostic_controller_time_origin.json', {
        'kind': 'actual_direct_diagnostic_controller_time_origin', 'schema_version': 1,
        'controller': controller, 'plan': observer.receipt(Path(directory) / 'measurement_plan.json'),
        'source': observer.receipt(Path(directory) / 'code/diagnose_direct_nsight_process.py'),
        'budget': diagnostic_budget(plan), 'clock_domain': monotonic_clock_domain(), 'started_monotonic': started,
        'published_monotonic': now, 'published_at_seconds': time.time()})
    return read_diagnostic_deadline(directory, plan)


def generic(plan):
    """Use only generic positive-ancestry primitives from the exact staged source."""
    path = Path(__file__).with_name("nsight_process_identity.py")
    require(path.resolve(strict=True) == path and path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == plan["helpers"]["files"].get(PREFIX + path.name),
        "the staged generic process observer changed")
    name = "direct_nsight_generic_process_identity"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        sys.modules[name] = module
    module = sys.modules[name]
    require(Path(module.__file__).resolve(strict=True) == path, "an ambient process observer cannot replace staged source")
    return module



def runtime_observer(plan, observer=None):
    """Require an explicitly started observer for native live calls; keep pure saved/default reads unchanged."""
    if "profile_target_process_control" in plan:
        target = adjacent_helper(plan, "profile_target_process")
        target.policy(plan)
        require(observer is not None and getattr(observer, "native_process_runtime", None) is not None,
            "profile-target native calls require their actual prior process bootstrap")
        observer.native_process_runtime.require_owner(plan)
        return observer
    if 'native_process_control' not in plan:
        require(observer is None, 'the default observer cannot receive an undeclared native runtime')
        return generic(plan)
    evidence = adjacent_helper(plan, 'native_process_evidence')
    evidence.policy(plan)
    require(observer is not None and getattr(observer, 'native_process_runtime', None) is not None,
        'the explicit native diagnostic requires its previously bootstrapped process observer')
    observer.native_process_runtime.require_owner(plan)
    return observer


def adjacent_helper(plan, name):
    """Load one exact new process-control or release helper from this staged closure."""
    path = Path(__file__).with_name(name + ".py")
    require(path.resolve(strict=True) == path and path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == plan["helpers"]["files"].get(PREFIX + path.name),
        "the staged direct helper changed: " + name)
    key = "direct_owned_" + name
    if key not in sys.modules:
        specification = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        sys.modules[key] = module
    module = sys.modules[key]
    require(Path(module.__file__).resolve(strict=True) == path, "an ambient helper cannot replace direct owned-process source")
    return module


def tool_scope(directory, plan, descriptor, *, observer=None, deadline=None):
    """Establish closed child ownership before any version, capture or export tool is spawned."""
    owner = context(plan, directory)
    control = adjacent_helper(plan, "direct_tool_descendants")
    if 'native_process_control' in plan or 'profile_target_process_control' in plan:
        require(runtime_observer(plan, observer).native_process_runtime.role == 'producer',
            'only the native producer observer may own the closed tool scope')
    selected = runtime_observer(plan, observer)
    if diagnostic_budget(plan) is not None:
        deadline = deadline or (selected.native_process_runtime.deadline if ('native_process_control' in plan or 'profile_target_process_control' in plan)
            else read_diagnostic_deadline(directory, plan))
        require(deadline is not None, 'the diagnostic scope needs its original time budget')
        deadline.remaining('work')
    else:require(deadline is None, 'another protocol cannot receive a diagnostic deadline')
    arguments = {} if deadline is None else {'deadline': deadline}
    scope = control.OwnedToolScope(directory, plan, selected, owner,
        lambda: descriptor_record(plan, descriptor, owner, local=True), **arguments)
    scope.standing_descriptor = descriptor
    return scope


def contract(plan, directory):
    """Separate direct process observations from Slurm, configured, final and capacity protocols."""
    directory = Path(directory)
    require(plan.get("capture_process_protocol") == PROTOCOL and plan.get("gpu_uuid") == GPU
        and plan.get("hardware_class") == "h100" and directory == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and directory.resolve(strict=True) == directory and not any(key in plan for key in (
            "configured_isolation", "configured_group", "device_memory_limit_protocol", "final_protocol", "scaling_protocol")),
        "this observer is only for the separately declared direct card-one process protocol")
    require(plan.get("kind") == DIAGNOSTIC or (plan.get("kind") in {"selected_configuration_profile", "full_original_profile_source_target_qualification"}
        and plan.get("direct_profile_protocol") == "direct-h100-reviewed-common-profile-version-1"),
        "an unrelated producer cannot use direct process observations")
    for name in ("direct_profile_process.py", "nsight_process_identity.py", "direct_capture_release.py", "direct_tool_descendants.py"):
        path = Path(__file__).with_name(name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == plan["helpers"]["files"].get(PREFIX + name),
            "the exact direct process helper changed")
    diagnostic_budget(plan)
    if 'profile_target_process_control' in plan:
        adjacent_helper(plan, 'profile_target_process').policy(plan)
    elif 'native_process_control' in plan:
        adjacent_helper(plan, 'native_process_evidence').policy(plan)
    return plan


def context(plan, directory):
    """Require the actual own direct host and unchanged UUID selector without querying a device."""
    contract(plan, directory)
    require(socket.gethostname().split(".")[0] == "serval02" and os.getuid() == os.geteuid() == OWNER
        and os.getresuid() == (OWNER, OWNER, OWNER), "direct process work requires the exact owner on serval02")
    environment = {key: value for key, value in os.environ.items() if key == "CUDA_VISIBLE_DEVICES" or key.startswith("SLURM_")}
    require(environment == {"CUDA_VISIBLE_DEVICES": GPU}, "direct process work cannot rewrite selectors or use a scheduler fallback")
    current = generic(plan).ProcReader().snapshot(os.getpid(), OWNER)
    require(current["uids"] == [OWNER] * 4, "all four actual user identifiers are required")
    return current


def descriptor_record(plan, descriptor, holder, *, local):
    """Read only the known producer's approved lock descriptor after checking its owned identity."""
    observer = generic(plan)
    require(type(descriptor) is int and descriptor >= 0 and type(local) is bool, "one actual typed producer descriptor is required")
    reader = observer.ProcReader()
    before = reader.snapshot(holder["pid"], OWNER, holder["cgroup"])
    require(observer.stable(before) == observer.stable(holder) and (not local or holder["pid"] == os.getpid()),
        "the known producer changed before descriptor inspection")
    source = Path("/proc") / str(holder["pid"])
    target = source / "fd" / str(descriptor)
    link = os.readlink(target)
    information, named = target.stat(), Path(LOCK).lstat()
    with (source / "fdinfo" / str(descriptor)).open() as stream:
        raw = stream.read(8193)
    require(len(raw.encode()) <= 8192, "producer descriptor metadata exceeded its bound")
    fields = lambda value: {"device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
        "mode": stat.S_IMODE(value.st_mode), "regular_file": stat.S_ISREG(value.st_mode)}
    after = reader.snapshot(holder["pid"], OWNER, holder["cgroup"])
    require(observer.stable(before) == observer.stable(after), "the known producer changed during descriptor inspection")
    value = {"path": LOCK, "fd": descriptor, "fd_stat": fields(information), "path_stat": fields(named),
        "path_symlink": stat.S_ISLNK(named.st_mode), "fd_target": link, "fdinfo": raw,
        "fdinfo_sha256": observer.digest(raw.encode()), "producer": holder, "producer_before": before,
        "producer_after": after, "read_by_pid": os.getpid(), "local_to_producer": local}
    validate_descriptor(plan, value, holder)
    return value


def validate_descriptor(plan, value, holder):
    """Rebuild the sole producer's FLOCK from saved actual descriptor and process bytes."""
    observer = generic(plan)
    require(value["path"] == value["fd_target"] == LOCK and type(value["fd"]) is int and value["fd"] >= 0
        and value["fd_stat"] == value["path_stat"] == LOCK_IDENTITY and value["path_symlink"] is False
        and observer.digest(value["fdinfo"].encode()) == value["fdinfo_sha256"],
        "the exact approved standing lock descriptor changed")
    observer.validate_snapshot(holder, OWNER, holder["cgroup"])
    for key in ("producer", "producer_before", "producer_after"):
        observer.validate_snapshot(value[key], OWNER, holder["cgroup"])
        require(observer.stable(value[key]) == observer.stable(holder), "producer descriptor evidence changed PID or birth")
    rows = re.findall(r"^lock:\s*[0-9]+:\s+FLOCK\s+ADVISORY\s+WRITE\s+([1-9][0-9]*)\s+([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9]+)\s+0\s+EOF\s*$", value["fdinfo"], re.M)
    require(len(rows) == 1 and int(rows[0][0]) == holder["pid"]
        and int(rows[0][1], 16) == os.major(LOCK_IDENTITY["device"])
        and int(rows[0][2], 16) == os.minor(LOCK_IDENTITY["device"])
        and int(rows[0][3]) == LOCK_IDENTITY["inode"], "the exact producer is not the exclusive standing FLOCK holder")
    require(type(value["read_by_pid"]) is int and value["read_by_pid"] > 0
        and type(value["local_to_producer"]) is bool
        and (not value["local_to_producer"] or value["read_by_pid"] == holder["pid"]),
        "readonly producer evidence cannot claim ownership by another process")
    return value


def no_standing_descriptor():
    """Check only this process's descriptors without retaining unrelated filenames or contents."""
    descriptors = os.listdir("/proc/self/fd")
    require(len(descriptors) <= 1024, "the controlled metadata process has an unsupported descriptor count")
    present = 0
    for value in descriptors:
        require(value.isdecimal(), "an actual descriptor name is invalid")
        try:
            information = os.fstat(int(value))
        except OSError as error:
            if error.errno == 9:
                continue  # The directory enumeration's temporary descriptor has already closed.
            raise
        present += information.st_dev == LOCK_IDENTITY["device"] and information.st_ino == LOCK_IDENTITY["inode"]
    require(present == 0, "the metadata controller, waiter or payload inherited the standing GPU descriptor")
    return {"pid": os.getpid(), "standing_descriptor_count": 0, "own_descriptors_only": True,
        "unrelated_descriptor_names_or_contents_retained": False}


def commands(directory, plan):
    """Launch metadata waiter and payload with no inherited standing descriptor."""
    entry = "diagnose_direct_nsight_process.py" if plan["kind"] == DIAGNOSTIC else "profile_selected.py"
    source = str(Path(directory) / "code" / entry)
    waiter = [plan["python"], "-I", "-S", "-B", source, "--direct-owned-waiter-unit", str(directory)]
    payload = [plan["python"], source, "--direct-owned-capture-unit", str(directory)]
    if plan["kind"] == DIAGNOSTIC:
        payload[1:1] = ["-I", "-S", "-B"]
    return waiter, payload


def tool_command(directory, plan):
    """Keep the recorded actual profiler options while retaining one real metadata primary."""
    waiter, unused = commands(directory, plan)
    return [plan["profile_tool"]["path"], "profile", "--trace=cuda,cudnn", "--sample=none", "--cpuctxsw=none",
        "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--stop-on-exit=true", "--wait=primary", "--force-overwrite=false", "--output=" + str(Path(directory) / "trace/capture"), *waiter]


def validate_intent(directory, plan, intent):
    """Reconstruct one exact attempt without a Slurm policy or inherited-descriptor assumption."""
    contract(plan, directory)
    observer = generic(plan)
    waiter, payload = commands(directory, plan)
    entry = "diagnose_direct_nsight_process.py" if plan["kind"] == DIAGNOSTIC else "profile_selected.py"
    require(type(intent.get("schema_version")) is int and intent["schema_version"] == 1 and intent.get("protocol") == PROTOCOL
        and type(intent.get("challenge")) is str and re.fullmatch(r"[0-9a-f]{64}", intent["challenge"])
        and intent["plan"] == observer.receipt(Path(directory) / "measurement_plan.json")
        and intent["source"] == observer.receipt(Path(directory) / "code" / entry)
        and intent["source"]["sha256"] == plan["helpers"]["files"].get(PREFIX + entry)
        and intent["tool_scope_begin"] == observer.receipt(Path(directory) / "tool_descendant_scope_begin.json")
        and intent["waiter_command"] == waiter and intent["capture_command"] == payload
        and intent["nsight_command"] == tool_command(directory, plan)
        and intent["standing_descriptor_passed_to_children"] is False
        and intent["metadata_controller_owns_standing_descriptor"] is False
        and intent["root_transport_owns_standing_descriptor"] is False,
        "the direct process attempt changed source, command or lock ownership")
    require(intent.get("hostname") == "serval02" and intent.get("original_environment") == {"CUDA_VISIBLE_DEVICES": GPU},
        "the direct attempt changed host or original selectors")
    holder = intent["producer"]
    observer.validate_snapshot(holder, OWNER, holder["cgroup"])
    validate_descriptor(plan, intent["producer_lock_before"], holder)
    require(intent["producer_lock_before"]["local_to_producer"] is True, "only the actual producer may begin the held-lock observation")
    return intent


def make_intent(directory, plan, descriptor):
    """Publish the actual sole lock owner's source and process before launching its tool."""
    observer = generic(plan)
    parent = context(plan, directory)
    waiter, payload = commands(directory, plan)
    entry = "diagnose_direct_nsight_process.py" if plan["kind"] == DIAGNOSTIC else "profile_selected.py"
    value = {"schema_version": 1, "protocol": PROTOCOL, "challenge": os.urandom(32).hex(),
        "created_at_seconds": time.time(),
        "hostname": "serval02", "original_environment": {"CUDA_VISIBLE_DEVICES": GPU},
        "plan": observer.receipt(Path(directory) / "measurement_plan.json"), "source": observer.receipt(Path(directory) / "code" / entry),
        "tool_scope_begin": observer.receipt(Path(directory) / "tool_descendant_scope_begin.json"),
        "producer": parent, "producer_lock_before": descriptor_record(plan, descriptor, parent, local=True),
        "waiter_command": waiter, "capture_command": payload, "nsight_command": tool_command(directory, plan),
        "standing_descriptor_passed_to_children": False, "metadata_controller_owns_standing_descriptor": False,
        "root_transport_owns_standing_descriptor": False}
    validate_intent(directory, plan, value)
    observer.publish(Path(directory) / "direct_profile_process_intent.json", value)
    return value


def producer_handle(plan, intent, *, observer=None):
    """Pin the known sole descriptor owner before any payload waits for release."""
    observer = runtime_observer(plan, observer)
    holder = intent["producer"]
    descriptor = observer.open_process_handle(holder["pid"])
    try:
        current = observer.ProcReader().snapshot(holder["pid"], OWNER, holder["cgroup"])
        require(observer.stable(current) == observer.stable(holder) and not observer.handle_exited(descriptor),
            "the exact sole lock producer is no longer alive")
    except BaseException:
        observer.close_process_handle(descriptor)
        raise
    return descriptor


def validate_claim(directory, plan, intent, value):
    """Join the blocked actual child to the immutable direct source and sole lock owner."""
    observer = generic(plan)
    validate_intent(directory, plan, intent)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("protocol") == PROTOCOL and value.get("challenge") == intent["challenge"]
        and value["plan"] == intent["plan"] and value["source"] == intent["source"]
        and value["intent"] == observer.receipt(Path(directory) / "direct_profile_process_intent.json")
        and value["standing_descriptor_inherited"] is False, "the actual child claim changed attempt or descriptor ownership")
    current = value["process"]
    observer.validate_snapshot(current, OWNER, intent["producer"]["cgroup"])
    require(value["pid"] == current["pid"] and type(value["pid"]) is int
        and value["process_start_time"] == current["process_start_time"] and current["command"] == intent["capture_command"],
        "the actual capture child changed its exact source command or birth")
    validate_descriptor(plan, value["observed_producer_lock"], intent["producer"])
    require(value["observed_producer_lock"]["local_to_producer"] is False
        and value["observed_producer_lock"]["read_by_pid"] == current["pid"], "a readonly child cannot become the GPU lock owner")
    absent = value["no_standing_descriptor"]
    require(absent["pid"] == current["pid"] and type(absent["standing_descriptor_count"]) is int
        and absent["standing_descriptor_count"] == 0 and absent["own_descriptors_only"] is True
        and absent["unrelated_descriptor_names_or_contents_retained"] is False,
        "the actual child did not observe absence of the standing descriptor")
    return value


def claim_self(directory, plan):
    """Record the actual source child and readonly producer lock proof before numerical work."""
    observer = generic(plan)
    current = context(plan, directory)
    intent = validate_intent(directory, plan, observer.read_json(Path(directory) / "direct_profile_process_intent.json"))
    require(current["command"] == intent["capture_command"], "the current child is not the intended staged command")
    value = {"schema_version": 1, "protocol": PROTOCOL, "challenge": intent["challenge"], "plan": intent["plan"],
        "claimed_at_seconds": time.time(),
        "source": intent["source"], "intent": observer.receipt(Path(directory) / "direct_profile_process_intent.json"),
        "pid": current["pid"], "process_start_time": current["process_start_time"], "process": current,
        "standing_descriptor_inherited": False, "no_standing_descriptor": no_standing_descriptor(), "observed_producer_lock": descriptor_record(plan,
            intent["producer_lock_before"]["fd"], intent["producer"], local=False)}
    validate_claim(directory, plan, intent, value)
    observer.publish(Path(directory) / "capture_identity.json", value)
    return value


def wait_release(directory, plan, *, observer=None):
    """Keep the actual payload blocked until positive ancestry and producer custody are saved."""
    observer = runtime_observer(plan, observer)
    deadline = None if diagnostic_budget(plan) is None else (observer.native_process_runtime.deadline
        if ('native_process_control' in plan or 'profile_target_process_control' in plan) else read_diagnostic_deadline(directory, plan))
    if 'native_process_control' in plan or 'profile_target_process_control' in plan:
        require(observer.native_process_runtime.role == 'capture', 'native release waiting requires the capture observer')
    claim = claim_self(directory, plan)
    intent = observer.read_json(Path(directory) / "direct_profile_process_intent.json")
    parent = producer_handle(plan, intent, observer=observer if ('native_process_control' in plan or 'profile_target_process_control' in plan) else None)
    try:
        started = time.monotonic()
        while not os.path.lexists(Path(directory) / "capture_release.json"):
            require(not observer.handle_exited(parent) and (time.monotonic() - started < 180 if deadline is None else deadline.remaining('work') > 0),
                "the producer disappeared or did not release this exact blocked payload")
            if deadline is not None:deadline.remaining('work')
            time.sleep(.05)
        if deadline is not None:deadline.remaining('work')
        observation = observer.read_json(Path(directory) / "capture_process_observation.json")
        observer.validate_observation(observation, claim, intent["capture_command"], intent["producer"])
        descriptor_record(plan, intent["producer_lock_before"]["fd"], intent["producer"], local=False)
        current = context(plan, directory)
        if deadline is not None:deadline.remaining('work')
        adjacent_helper(plan, "direct_capture_release").acknowledge_release(directory, plan, intent, claim, current, observer)
        if deadline is not None:deadline.remaining('work')
        return claim
    finally:
        observer.close_process_handle(parent)


def run_waiter(directory, plan, *, observer=None):
    """Spawn and wait on exactly one payload without holding the standing GPU descriptor."""
    observer = runtime_observer(plan, observer)
    deadline = None if diagnostic_budget(plan) is None else (observer.native_process_runtime.deadline
        if ('native_process_control' in plan or 'profile_target_process_control' in plan) else read_diagnostic_deadline(directory, plan))
    if 'native_process_control' in plan or 'profile_target_process_control' in plan:
        require(observer.native_process_runtime.role == 'waiter', 'native payload launching requires the waiter observer')
    current = context(plan, directory)
    waiter_no_lock = no_standing_descriptor()
    intent = validate_intent(directory, plan, observer.read_json(Path(directory) / "direct_profile_process_intent.json"))
    require(current["command"] == intent["waiter_command"] and not NUMERICAL.intersection(sys.modules),
        "the actual direct metadata waiter changed source or imported a numerical runtime")
    parent = producer_handle(plan, intent, observer=observer if ('native_process_control' in plan or 'profile_target_process_control' in plan) else None)
    child, interventions = None, []
    try:
        if deadline is not None:deadline.remaining('work')
        child = subprocess.Popen(intent["capture_command"], close_fds=True, pass_fds=())
        started = time.monotonic()
        while not os.path.lexists(Path(directory) / "capture_identity.json"):
            require(child.poll() is None and not observer.handle_exited(parent) and (time.monotonic() - started < 180 if deadline is None else deadline.remaining('work') > 0),
                "the actual payload ended or failed to publish its own blocked identity")
            if deadline is not None:deadline.remaining('work')
            time.sleep(.05)
        claim = validate_claim(directory, plan, intent, observer.read_json(Path(directory) / "capture_identity.json"))
        actual = observer.ProcReader().snapshot(child.pid, OWNER, current["cgroup"])
        require(child.pid == claim["pid"] and actual["parent_pid"] == current["pid"]
            and observer.stable(actual) == observer.stable(claim["process"]), "the real waiter Popen child differs from its claim")
        observer.publish(Path(directory) / "capture_actual_launch.json", {"schema_version": 1, "protocol": PROTOCOL,
            "challenge": intent["challenge"], "waiter": current, "child": actual, "command": intent["capture_command"],
            "claim": observer.receipt(Path(directory) / "capture_identity.json"), "close_fds": True, "pass_fds": [],
            "waiter_no_standing_descriptor": waiter_no_lock,
            "producer_lock_before_wait": descriptor_record(plan, intent["producer_lock_before"]["fd"], intent["producer"], local=False)})
        while child.poll() is None:
            if observer.handle_exited(parent):
                if deadline is not None:deadline.begin_cleanup()
                interventions.append("the sole standing lock producer exited")
                if deadline is not None:deadline.remaining('cleanup')
                child.terminate()
                try:
                    child.wait(timeout=10 if deadline is None else deadline.timeout(10, phase='cleanup'))
                except subprocess.TimeoutExpired:
                    interventions.append("the exact owned child required SIGKILL")
                    if deadline is not None:deadline.remaining('cleanup')
                    child.kill(); child.wait() if deadline is None else child.wait(timeout=deadline.remaining('cleanup'))
                break
            if deadline is not None:deadline.remaining('work')
            time.sleep(.05)
        returned = child.wait() if deadline is None else child.wait(timeout=deadline.remaining('total'))
        after = descriptor_record(plan, intent["producer_lock_before"]["fd"], intent["producer"], local=False)
        observer.publish(Path(directory) / "capture_actual_wait.json", {"schema_version": 1, "protocol": PROTOCOL,
            "challenge": intent["challenge"], "command": intent["capture_command"], "waited_pid": child.pid,
            "returncode": returned, "exited_normally": returned >= 0, "interventions": interventions,
            "launch": observer.receipt(Path(directory) / "capture_actual_launch.json"),
            "producer_lock_after_wait": after, "standing_descriptor_inherited": False})
        return returned if returned >= 0 else 128 - returned
    except BaseException as error:
        if deadline is not None:deadline.begin_cleanup()
        actual_return, cleanup_error = None, None
        try:
            if child is not None:
                if child.poll() is None:
                    interventions.append("the exact owned payload received SIGTERM during waiter failure")
                    if deadline is not None:deadline.remaining('cleanup')
                    child.terminate()
                    try:
                        child.wait(timeout=10 if deadline is None else deadline.timeout(10, phase='cleanup'))
                    except subprocess.TimeoutExpired:
                        interventions.append("the exact owned payload received SIGKILL after cleanup timeout")
                        if deadline is not None:deadline.remaining('cleanup')
                        child.kill()
                actual_return = child.wait() if deadline is None else child.wait(timeout=deadline.remaining('cleanup'))
        except BaseException as failed_cleanup:
            cleanup_error = repr(failed_cleanup)
        observer.publish(Path(directory) / "capture_actual_wait_failure.json", {
            "schema_version": 1, "protocol": PROTOCOL, "error_type": type(error).__name__, "error": str(error),
            "command": intent["capture_command"], "known_actual_popen_pid": child.pid if child is not None else None,
            "returncode": actual_return, "interventions": interventions, "cleanup_error": cleanup_error,
            "successful_capture_claimed": False, "scientific_credit": 0, "admission_granted": False})
        raise
    finally:
        if child is not None and child.poll() is None:
            if deadline is not None:deadline.begin_cleanup()
            if deadline is not None and deadline.remaining('cleanup', allow_zero=True) <= 0:
                observer.publish(Path(directory) / 'waiter_cleanup_budget_inconclusive.json', {
                    'known_actual_popen_pid': child.pid, 'deadline': deadline.snapshot(),
                    'completed': False, 'admission_granted': False, 'scientific_credit': 0,
                    'producer_scope_cleanup_still_required': True})
            else:
                if deadline is not None:deadline.remaining('cleanup')
                child.terminate()
                try:
                    child.wait(timeout=10 if deadline is None else deadline.timeout(10, phase='cleanup'))
                except subprocess.TimeoutExpired:
                    if deadline is not None:deadline.remaining('cleanup')
                    child.kill(); child.wait() if deadline is None else child.wait(timeout=deadline.remaining('cleanup'))
        observer.close_process_handle(parent)


def run_tool(directory, plan, descriptor, *, scope=None, observer=None):
    """Return only after closed tool descendants have exited, including startup failures."""
    if plan.get("kind") != DIAGNOSTIC:
        from direct_profile_prerequisites import require_admission
        require_admission(plan)
    if scope is None:
        with (tool_scope(directory, plan, descriptor) if observer is None else
                tool_scope(directory, plan, descriptor, observer=observer)) as owned:
            claim = _run_tool_in_scope(directory, plan, descriptor, owned)
        validate_saved(directory, plan)
        return claim
    require(scope.active and scope.pid == os.getpid() and scope.standing_descriptor == descriptor,
        "the actual sole producer must own this active closed tool scope")
    if 'native_process_control' in plan or 'profile_target_process_control' in plan:
        require(runtime_observer(plan, scope.observer).native_process_runtime.role == 'producer',
            'the native tool call must reuse its actual producer provider')
    return _run_tool_in_scope(directory, plan, descriptor, scope)


def _run_tool_in_scope(directory, plan, descriptor, scope):
    """Release the exact blocked payload only after owned startup ancestry and local custody are saved."""
    directory = Path(directory)
    observer = scope.observer
    deadline = scope.deadline if diagnostic_budget(plan) is not None else None
    intent = make_intent(directory, plan, descriptor)
    command = intent["nsight_command"]
    observer.publish(directory / "executed_command.json", command)
    (directory / "trace").mkdir()
    processes, child, started = None, None, time.monotonic()
    with (directory / "capture.log").open("x") as log:
        try:
            child = scope.spawn(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            root = observer.ProcReader().snapshot(child.pid, OWNER, intent["producer"]["cgroup"])
            owner = {"pid": child.pid, "process_start_time": root["process_start_time"],
                "parent_pid": os.getpid(), "parent_start_time": intent["producer"]["process_start_time"], "command": command,
                "close_fds": True, "pass_fds": []}
            observer.publish(directory / "profiler_process.json", owner)
            processes = observer.OwnedProcesses(child.pid, root["process_start_time"], intent["producer"])
            while not all(os.path.lexists(directory / name) for name in ("capture_identity.json", "capture_actual_launch.json")):
                scope.scan()
                require(child.poll() is None and (time.monotonic() - started < min(180, plan['maximum_active_seconds']) if deadline is None else deadline.remaining('work') > 0),
                    "Nsight or its actual payload did not publish the blocked source identity")
                time.sleep(.05)
            claim = validate_claim(directory, plan, intent, observer.read_json(directory / "capture_identity.json"))
            launch = observer.read_json(directory / "capture_actual_launch.json")
            require(launch["claim"] == observer.receipt(directory / "capture_identity.json")
                and launch["command"] == intent["capture_command"] and launch["close_fds"] is True and launch["pass_fds"] == []
                and observer.stable(launch["child"]) == observer.stable(claim["process"])
                and launch["waiter"]["pid"] == claim["process"]["parent_pid"], "the real blocked payload launch changed")
            observation = processes.observe(claim["pid"], claim["process_start_time"], intent["capture_command"])
            observation["observed_at_seconds"] = time.time()
            observer.publish(directory / "capture_process_observation.json", observation)
            observer.validate_observation(observer.read_json(directory / "capture_process_observation.json"),
                claim, intent["capture_command"], intent["producer"])
            custody = descriptor_record(plan, descriptor, intent["producer"], local=True)
            custody["observed_at_seconds"] = time.time()
            observer.publish(directory / "producer_lock_before_release.json", custody)
            if deadline is not None:deadline.remaining('work')
            adjacent_helper(plan, "direct_capture_release").publish_release(directory, plan, intent, claim,
                observer.receipt(directory / "capture_process_observation.json"),
                observer.receipt(directory / "producer_lock_before_release.json"), observer)
            returned = scope.wait(child, timeout=max(.001, plan['maximum_active_seconds'] - (time.monotonic() - started))
                if deadline is None else deadline.remaining('work'))
            after = descriptor_record(plan, descriptor, intent["producer"], local=True)
            observer.publish(directory / "profiler_actual_wait.json", {"schema_version": 1, "protocol": PROTOCOL,
                "command": command, "waited_pid": child.pid, "process_start_time": root["process_start_time"],
                "returncode": returned, "exited_normally": returned >= 0, "interventions": [],
                "launch": observer.receipt(directory / "profiler_process.json"), "producer_lock_after_wait": after})
            require(returned == 0, "the actual Nsight Popen returned a failure")
        except BaseException as error:
            scope.failed = True
            observer.publish(directory / "direct_process_failure.json", {"error_type": type(error).__name__,
                "error": str(error), "measurement_valid": False, "scientific_credit": 0,
                "admission_granted": False, "closed_tool_scope_cleanup_required_before_descriptor_release": True})
            raise
        finally:
            if processes is not None:
                processes.close()
    return observer.read_json(directory / "capture_identity.json")


def validate_saved(directory, plan):
    """Rebuild exact actual child/root waits and sole-producer descriptor custody from raw records."""
    observer = generic(plan)
    directory = Path(directory)
    intent = validate_intent(directory, plan, observer.read_json(directory / "direct_profile_process_intent.json"))
    claim = validate_claim(directory, plan, intent, observer.read_json(directory / "capture_identity.json"))
    observed = observer.validate_observation(observer.read_json(directory / "capture_process_observation.json"),
        claim, intent["capture_command"], intent["producer"])
    observation = observer.read_json(directory / "capture_process_observation.json")
    launch = observer.read_json(directory / "capture_actual_launch.json")
    waited = observer.read_json(directory / "capture_actual_wait.json")
    root, owner = observer.read_json(directory / "profiler_actual_wait.json"), observer.read_json(directory / "profiler_process.json")
    require(launch["challenge"] == intent["challenge"] and launch["command"] == intent["capture_command"]
        and launch["claim"] == observer.receipt(directory / "capture_identity.json")
        and launch["close_fds"] is True and launch["pass_fds"] == []
        and observer.stable(launch["child"]) == observer.stable(claim["process"])
        and launch["waiter"]["pid"] == claim["process"]["parent_pid"]
        and observer.stable(launch["waiter"]) == observer.stable(observation["nodes"][str(launch["waiter"]["pid"])]),
        "saved actual direct payload launch changed")
    observer.validate_snapshot(launch["waiter"], OWNER, intent["producer"]["cgroup"])
    require(type(waited.get("schema_version")) is int and waited["schema_version"] == 1 and waited.get("protocol") == PROTOCOL
        and type(waited.get("returncode")) is int and waited["returncode"] == 0 and waited["exited_normally"] is True
        and waited["interventions"] == [] and waited["challenge"] == intent["challenge"]
        and waited["waited_pid"] == claim["pid"] and waited["command"] == intent["capture_command"]
        and waited["launch"] == observer.receipt(directory / "capture_actual_launch.json")
        and waited["standing_descriptor_inherited"] is False,
        "the exact payload has no successful actual waiter Popen return")
    require(type(root.get("schema_version")) is int and root["schema_version"] == 1 and root.get("protocol") == PROTOCOL
        and type(root.get("returncode")) is int and root["returncode"] == 0 and root["exited_normally"] is True
        and root["interventions"] == [] and root["waited_pid"] == owner["pid"]
        and root["process_start_time"] == owner["process_start_time"] and root["command"] == intent["nsight_command"]
        and root["launch"] == observer.receipt(directory / "profiler_process.json")
        and owner["parent_pid"] == intent["producer"]["pid"] and owner["parent_start_time"] == intent["producer"]["process_start_time"]
        and owner["close_fds"] is True and owner["pass_fds"] == []
        and owner["pid"] == observed["root"]["pid"] and owner["process_start_time"] == observed["root"]["process_start_time"],
        "the exact Nsight root has no successful producer Popen return")
    for value in (launch["producer_lock_before_wait"], waited["producer_lock_after_wait"], root["producer_lock_after_wait"]):
        validate_descriptor(plan, value, intent["producer"])
        require(value["fd"] == intent["producer_lock_before"]["fd"], "producer descriptor number changed across actual waits")
    require(root["producer_lock_after_wait"]["local_to_producer"] is True,
        "the final descriptor proof must come from the actual sole producer")
    custody = observer.read_json(directory / "producer_lock_before_release.json")
    validate_descriptor(plan, custody, intent["producer"])
    if plan["kind"] == DIAGNOSTIC:
        payload = observer.read_json(directory / "probe_result.json")
        work = {"pid": payload["pid"], "process_start_time": payload["process_start_time"],
            "started_at_seconds": payload["payload_started_at_seconds"], "finished_at_seconds": payload["payload_finished_at_seconds"]}
    else:
        work = observer.read_json(directory / "capture_result.json")["execution_identity"]
    release = adjacent_helper(plan, "direct_capture_release").validate_saved_release(
        directory, plan, intent, claim, observation, observer, work=work)
    scope = adjacent_helper(plan, "direct_tool_descendants").validate_saved_scope(directory, plan, observer,
        intent["producer"], owner["pid"], owner["process_start_time"],
        lambda record: validate_descriptor(plan, record, intent["producer"]),
        **({} if diagnostic_budget(plan) is None else {'diagnostic_deadline': validate_diagnostic_origin(directory, plan, observer)}))
    if "profile_target_process_control" in plan:
        native = {"profile_target_process_evidence": adjacent_helper(plan, "profile_target_process").validate_saved(
            directory, plan, observer, intent, observation, launch, waited,
            validate_diagnostic_origin(directory, plan, observer))}
    else:
        native = {} if 'native_process_control' not in plan else {
            'native_process_evidence': adjacent_helper(plan, 'native_process_evidence').validate_saved(
                directory, plan, observer, intent, observation, launch, waited,
                diagnostic_deadline=None if diagnostic_budget(plan) is None else validate_diagnostic_origin(directory, plan, observer))}
    return {**native, "kind": "actual_direct_nsight_payload_and_producer_waits", "passed": True,
        "capture_identity": observer.receipt(directory / "capture_identity.json"),
        "capture_wait": observer.receipt(directory / "capture_actual_wait.json"),
        "root_wait": observer.receipt(directory / "profiler_actual_wait.json"),
        "producer_lock_before": intent["producer_lock_before"], "producer_lock_after": root["producer_lock_after_wait"],
        "standing_descriptor_passed_to_children": False, **release, **scope,
        "scientific_credit": 0, "admission_granted": False}


def profile_mode(plan):
    """Recognize only the explicit current direct profile for the separate custody reader."""
    return (plan.get("kind") in {"selected_configuration_profile", "full_original_profile_source_target_qualification"} and plan.get("capture_process_protocol") == PROTOCOL
        and plan.get("direct_profile_protocol") == "direct-h100-reviewed-common-profile-version-1")


def lock_fields(value):
    """Keep the unchanged direct lock-byte contract separate from changing process counters."""
    return {key: value[key] for key in ("path", "fd", "fd_stat", "path_stat", "path_symlink", "fd_target", "fdinfo", "fdinfo_sha256")}


def collect_capture_metadata(plan, directory, identity):
    """Admit an actual source child guarded by the sole producer, without inheriting its descriptor."""
    directory = Path(directory)
    require(profile_mode(plan) and directory == identity.unit_directory(plan), "current direct child metadata requires its exact profile unit")
    context(plan, directory)
    observer = generic(plan)
    intent = validate_intent(directory, plan, observer.read_json(directory / "direct_profile_process_intent.json"))
    claim = validate_claim(directory, plan, intent, observer.read_json(directory / "capture_identity.json"))
    observer.validate_observation(observer.read_json(directory / "capture_process_observation.json"),
        claim, intent["capture_command"], intent["producer"])
    value = {"schema_version": 1, "kind": "serval02_card1_process_metadata", "profile": identity.PROFILE,
        "capability": identity.CAPABILITY, "role": "producer_guarded_child", "output_directory": str(directory),
        "process_protocol": PROTOCOL, "standing_descriptor_inherited": False, **identity.BOUNDARY}
    try:
        value["source"] = identity.source_identity(plan)
        value["history"] = identity.validate_history(plan)
        value["current_admission"] = identity.current_admission_claim(plan)
        identity.check_live_admission(value["current_admission"])
        value["before"] = identity.current_process()
        value["environment"] = {key: val for key, val in os.environ.items() if key == "CUDA_VISIBLE_DEVICES" or key.startswith("SLURM_")}
        value["current_launch"] = identity.launch_inputs(plan)
        value["parent_lock_receipt"] = observer.receipt(directory / "supervisor/direct_process_metadata.json")
        value["parent_metadata"] = observer.read_json(value["parent_lock_receipt"]["path"])
        value["process_intent"] = observer.receipt(directory / "direct_profile_process_intent.json")
        value["capture_identity"] = observer.receipt(directory / "capture_identity.json")
        value["process_observation"] = observer.receipt(directory / "capture_process_observation.json")
        value["release_acknowledgment"] = observer.receipt(directory / "capture_release_observed.json")
        value["no_standing_descriptor"] = no_standing_descriptor()
        value["producer_descriptor_before"] = descriptor_record(plan, intent["producer_lock_before"]["fd"], intent["producer"], local=False)
        value["lock_before"] = lock_fields(value["producer_descriptor_before"])
        value["after"] = identity.current_process()
        value["producer_descriptor_after"] = descriptor_record(plan, intent["producer_lock_before"]["fd"], intent["producer"], local=False)
        value["lock_after"] = lock_fields(value["producer_descriptor_after"])
        validate_capture_metadata(plan, value, identity)
    except BaseException as error:
        value["capture_error"] = {"type": type(error).__name__, "error": str(error)}
        raise
    finally:
        identity.publish(directory / "direct_process_metadata.json", value)
    return observer.receipt(directory / "direct_process_metadata.json")


def validate_capture_metadata(plan, value, identity):
    """Retain original source/history/admission gates and substitute only producer-owned custody."""
    directory = identity.unit_directory(plan)
    require(profile_mode(plan) and value.get("process_protocol") == PROTOCOL
        and value.get("role") == "producer_guarded_child" and value.get("standing_descriptor_inherited") is False
        and value.get("kind") == "serval02_card1_process_metadata" and value.get("capability") == identity.CAPABILITY
        and type(value.get("schema_version")) is int and value["schema_version"] == 1 and "capture_error" not in value
        and value["output_directory"] == str(directory), "the current profile child has incomplete source or custody metadata")
    identity.validate_plan(plan, "component_profile")
    identity.require_profile(value["profile"])
    require(all(identity.exact(value.get(key), expected) for key, expected in identity.BOUNDARY.items()),
        "the new process reader cannot grant admission or scientific credit")
    observer = generic(plan)
    require(value["process_intent"] == observer.receipt(directory / "direct_profile_process_intent.json")
        and value["capture_identity"] == observer.receipt(directory / "capture_identity.json")
        and value["process_observation"] == observer.receipt(directory / "capture_process_observation.json")
        and value["release_acknowledgment"] == observer.receipt(directory / "capture_release_observed.json")
        and value["parent_lock_receipt"] == observer.receipt(directory / "supervisor/direct_process_metadata.json"),
        "new source-child metadata changed its exact reserved input paths")
    intent = validate_intent(directory, plan, observer.read_json(directory / "direct_profile_process_intent.json"))
    claim = validate_claim(directory, plan, intent, observer.read_json(value["capture_identity"]["path"]))
    observer.validate_observation(observer.read_json(value["process_observation"]["path"]), claim,
        intent["capture_command"], intent["producer"])
    adjacent_helper(plan, "direct_capture_release").validate_saved_release(directory, plan, intent, claim,
        observer.read_json(directory / "capture_process_observation.json"), observer)
    before, after = identity.validate_process(value["before"]), identity.validate_process(value["after"])
    require(before == after and before["pid"] == claim["pid"] and before["process_start_time"] == claim["process_start_time"]
        and before["parent_pid"] == claim["process"]["parent_pid"] and before["command"] == intent["capture_command"]
        and value["environment"] == {"CUDA_VISIBLE_DEVICES": GPU}
        and value["source"] == identity.source_identity(plan) and value["history"] == identity.validate_history(plan)
        and value["current_admission"] == identity.current_admission_claim(plan)
        and value["current_launch"] == identity.launch_inputs(plan), "the actual child changed protected original source or runtime admission")
    parent = value["parent_metadata"]
    require(observer.read_json(value["parent_lock_receipt"]["path"]) == parent and parent["role"] == "owner",
        "the descriptor must be owned by the original tool producer")
    identity.validate_metadata(plan, parent)
    producer = parent["after"]["identity"]
    require(producer["pid"] == intent["producer"]["pid"] and producer["process_start_time"] == intent["producer"]["process_start_time"]
        and value["current_launch"]["process_row"]["pid"] == producer["pid"]
        and value["current_launch"]["process_row"]["start_time"] == producer["process_start_time"]
        and value["lock_before"] == value["lock_after"] == parent["lock_after"],
        "the new profile substituted another producer or descriptor")
    for name, target in (("producer_descriptor_before", "lock_before"), ("producer_descriptor_after", "lock_after")):
        validate_descriptor(plan, value[name], intent["producer"])
        require(value[name]["local_to_producer"] is False and value[name]["read_by_pid"] == before["pid"]
            and lock_fields(value[name]) == value[target], "the source child cannot become a standing lock holder")
    no_lock = value["no_standing_descriptor"]
    require(no_lock["pid"] == before["pid"] and type(no_lock["standing_descriptor_count"]) is int
        and no_lock["standing_descriptor_count"] == 0 and no_lock["own_descriptors_only"] is True
        and no_lock["unrelated_descriptor_names_or_contents_retained"] is False,
        "the source child did not prove absence of an inherited standing descriptor")
    return value


def live_capture_custody(metadata, descriptor, identity):
    """Check the actual source child and its still-live sole producer at each numerical boundary."""
    current = identity.current_process()
    process = identity.validate_process(current)
    require(process == metadata["after"]["identity"] and type(descriptor) is int and descriptor == metadata["lock_after"]["fd"]
        and {key: val for key, val in os.environ.items() if key == "CUDA_VISIBLE_DEVICES" or key.startswith("SLURM_")}
        == metadata["environment"], "the current source child or original UUID selector changed")
    no_standing_descriptor()
    plan = identity.read_receipt(identity.receipt(Path(metadata["output_directory"]) / "measurement_plan.json"))
    intent = generic(plan).read_json(Path(metadata["output_directory"]) / "direct_profile_process_intent.json")
    actual = descriptor_record(plan, descriptor, intent["producer"], local=False)
    require(lock_fields(actual) == metadata["lock_after"], "the sole producer's standing descriptor changed after child admission")
    return {"process": process, "lock": lock_fields(actual)}


def validate_profile_terminal(directory, metadata, result, support):
    """Join a completed current profile to actual source-child waits and unchanged numerical capture."""
    directory = Path(directory)
    identity = support.core()
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    require(profile_mode(plan), "the producer-owned terminal reader requires an explicit current direct profile")
    validate_capture_metadata(plan, metadata, identity)
    proof = validate_saved(directory, plan)
    value = identity.read_receipt(identity.receipt(directory / "direct_profile_terminal.json"))
    captured = identity.read_receipt(identity.receipt(directory / "capture_result.json"))
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "direct_producer_guarded_profile_terminal" and value["process_proof"] == proof
        and value["capture_result"] == identity.receipt(directory / "capture_result.json")
        and value["parent_custody"] == identity.receipt(directory / "supervisor/direct_process_metadata.json")
        and value["standing_descriptor_inherited_by_child"] is False
        and all(identity.exact(value.get(key), expected) for key, expected in identity.BOUNDARY.items())
        and result.get("capture_process_protocol") == PROTOCOL and result.get("direct_process_wait_proof") == proof
        and captured.get("capture_completed") is True and result.get("completed") is True
        and all(identity.exact(result.get(key), expected) for key, expected in captured.items()),
        "the completed direct profile changed source-child waits or immutable numerical capture")
    require(result.get("trace_analysis_record") == str(directory / "trace/analysis.json")
        and result.get("trace_report_sha256") == identity.receipt(directory / "trace/capture.nsys-rep")["sha256"]
        and result.get("direct_trace_process_proof") == support.trace_process_proof(directory, metadata, publish_result=False),
        "the positive native trace was not joined to this actual waited source child")
    baseline = support.adjacent("direct_profile_prerequisites").completed_capture_binding(directory, plan, captured)
    require(all(result.get(key) == expected for key, expected in baseline.items()), "the current profile omitted exact pre-numerical baseline or post-core backend proof")
    return value

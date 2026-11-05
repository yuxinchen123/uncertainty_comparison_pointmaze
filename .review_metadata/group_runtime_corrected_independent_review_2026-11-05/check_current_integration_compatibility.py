"""Compare exact current integration dependencies and cold-import the group union on owned CPU."""
import ast
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

WORKTREE = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
INTEGRATION = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
COMMIT = "b181e171bb1bdb792f3a7082b00f7186ebc9939a"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
FREEZE = WORKTREE / ".review_metadata/group_runtime_four_predicate_corrections/corrected_source_freeze_for_independent_review.json"
CLOSURE = DIRECTORY / "independent_source_closure.json"


def digest(raw):
    """Hash exact source bytes without importing them."""
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    """Write one reviewer-owned immutable receipt while preserving old attempts."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def audit(event, args):
    """Allow exact local Git reads and an isolated system-Python metadata child only."""
    if event == "subprocess.Popen":
        command = args[1]
        if not isinstance(command, (list, tuple)) or not command:
            raise RuntimeError("literal metadata subprocess required")
        if command[0] == "git":
            if args[0] != "git" or command[:3] != ["git", "-C", str(INTEGRATION)] or command[3] not in {"show", "cat-file"}:
                raise RuntimeError("only exact integration Git source reads allowed")
        elif command[:4] != ["/usr/bin/python3", "-I", "-S", "-B"] or args[0] != "/usr/bin/python3":
            raise RuntimeError("only isolated metadata Python child allowed")
    if event in {"os.system", "os.posix_spawn", "socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym"}:
        raise RuntimeError("no shell, network, GPU, tool or later native library operation")
    if event == "import" and str(args[0]).split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "jaxtari", "cupy", "pynvml", "flax", "optax", "scipy"}:
        raise RuntimeError("numerical imports forbidden")


PROBE = r'''import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
root = Path(__file__).resolve().parent
executed = set()
def audit(event, args):
    if event == "exec" and hasattr(args[0], "co_filename"):
        filename = args[0].co_filename
        if filename.startswith(str(root) + "/"):
            executed.add(Path(filename).name)
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym"}:
        raise RuntimeError("no subprocess, network, tool, GPU or later native library operation")
    if event == "import" and str(args[0]).split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "jaxtari", "cupy", "pynvml", "flax", "optax", "scipy"}:
        raise RuntimeError("numerical import forbidden")
sys.addaudithook(audit)
sys.path.insert(0, str(root))
import group_runtime as runtime
names = list(runtime.helper_names())
result = {"helper_names": names, "helper_count": len(names), "executed_adjacent_source_files": sorted(executed),
    "metadata_modules": {name: Path(value.__file__).name for name,value in list(sys.modules.items()) if getattr(value,"__file__",None) and str(Path(value.__file__).absolute()).startswith(str(root)+"/")},
    "numerical_imports": [], "subprocess_gpu_device_tool_actions": 0}
print(json.dumps(result, sort_keys=True))
'''


def main():
    """Use current shared Git bytes, preserving all eight approved new group source files."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit() or os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("use only admitted owned CPU numeric allocation")
    expected = json.loads(FREEZE.read_text())["files"]
    if digest(FREEZE.read_bytes()) != "fb795e40e663b1638ea0536ddf3727c6d4b4e0e36efdcf97b62f757449bb626a" or any(digest(Path(path).read_bytes()) != sha for path,sha in expected.items()):
        raise ValueError("approved frozen source changed")
    helper_files = json.loads(CLOSURE.read_text())["helper_files"]
    sys.addaudithook(audit)
    rows, inputs = [], {}
    for old_path, approved_sha in helper_files.items():
        old = Path(old_path); relative = old.relative_to(WORKTREE).as_posix()
        presence = subprocess.run(["git", "-C", str(INTEGRATION), "cat-file", "-e", COMMIT + ":" + relative], capture_output=True)
        if presence.returncode == 0:
            raw = subprocess.check_output(["git", "-C", str(INTEGRATION), "show", COMMIT + ":" + relative])
            if (INTEGRATION / relative).read_bytes() != raw:
                raise ValueError("current integration source differs from fixed compatibility commit")
            state = "identical_shared" if digest(raw) == approved_sha else "retain_current_shared"
        elif presence.returncode == 128:
            raw = old.read_bytes(); state = "approved_new_group_source"
        else:
            raise ValueError("unrecognized local Git source presence failure")
        inputs[old.name] = raw
        old_tree, new_tree = ast.parse(old.read_bytes()), ast.parse(raw)
        def declarations(tree):
            return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        old_decl, new_decl = declarations(old_tree), declarations(new_tree)
        rows.append({"name":old.name,"relative_path":relative,"approved_sha256":approved_sha,"overlay_sha256":digest(raw),"state":state,
            "changed_declarations":[name for name in sorted(set(old_decl)|set(new_decl)) if old_decl.get(name)!=new_decl.get(name)]})
    with tempfile.TemporaryDirectory(prefix="group_current_integration_metadata_") as temporary:
        stage=Path(temporary)
        for name,raw in inputs.items(): (stage/name).write_bytes(raw)
        (stage/"probe.py").write_text(PROBE)
        command=["/usr/bin/python3","-I","-S","-B",str(stage/"probe.py")]
        child=subprocess.run(command,capture_output=True,text=True,timeout=60)
        save(DIRECTORY/"current_integration_cold_import_child_return.json",{"argv":command,"returncode":child.returncode,"stdout":child.stdout,"stderr":child.stderr})
        if child.returncode != 0:
            raise ValueError("current integration cold metadata import failed")
        proof=json.loads(child.stdout)
        if set(proof["helper_names"])!=set(inputs) or proof["helper_count"]!=29:
            raise ValueError("current integration changed the required group helper union")
    unchanged=all(digest(Path(path).read_bytes())==sha for path,sha in expected.items())
    record={"kind":"group_runtime_current_integration_metadata_compatibility","schema_version":2,"passed":unchanged,
        "current_integration_commit":COMMIT,"approved_source_freeze":{"path":str(FREEZE),"sha256":digest(FREEZE.read_bytes())},
        "source_rows":rows,"summary":{state:sum(row["state"]==state for row in rows) for state in ("identical_shared","retain_current_shared","approved_new_group_source")},
        "cold_metadata_import":proof,"frozen35_files_unchanged":unchanged,"actual_owned_job":os.environ["SLURM_JOB_ID"],
        "actual_numeric_step":os.environ["SLURM_STEP_ID"],"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stdlib_bootstrap_before_audit":"ctypes initializes its own-process Python API before audit; later library operations refused",
        "actual_numerical_gpu_device_tool_scheduler_operations":0,"scientific_credit":0,"operational_admission_granted":False}
    save(DIRECTORY/"current_integration_metadata_compatibility.json",record)
    print(json.dumps({"passed":record["passed"],"counts":record["summary"],"helper_count":proof["helper_count"],"step":record["actual_numeric_step"]}))


if __name__ == "__main__":
    main()

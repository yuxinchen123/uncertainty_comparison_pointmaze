"""Freeze the small monitor display correction and unchanged source-function evidence."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE="6b7d492217140bba3d2f36b93754ecde2d020ccc"
PREFIX="10_jax_exploration_platform/scripts/atari_pixel_campaign/"
STEP=os.environ["SLURM_STEP_ID"]
assert os.environ.get("SLURM_JOB_ID")=="6783003" and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path("/proc/self/cgroup").read_text().splitlines()==["0::/system.slice/slurmstepd.scope/job_6783003/step_"+STEP+"/user/task_0"]

def git(*args):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(ROOT),*args],env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})

def doc(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True)+"\n").encode()
    with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()}

assert git("rev-parse","HEAD").decode().strip()==BASE
monitor=ROOT/PREFIX/"monitor.py"
old=ast.parse(git("show",BASE+":"+PREFIX+"monitor.py"));new=ast.parse(monitor.read_bytes())
functions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
before,after=functions(old),functions(new)
assert set(after)-set(before)=={"direct_unit_display"} and not set(before)-set(after)
assert all(before[name]==after[name] for name in before if name!="unit_rows")
old_imports=[ast.dump(node,include_attributes=False) for node in old.body if isinstance(node,(ast.Import,ast.ImportFrom))]
new_imports=[ast.dump(node,include_attributes=False) for node in new.body if isinstance(node,(ast.Import,ast.ImportFrom))]
assert [node for node in new_imports if node not in old_imports]==[ast.dump(ast.Import(names=[ast.alias(name="stat")]),include_attributes=False)]
source_names=git("ls-tree","-r","--name-only",BASE,"10_jax_exploration_platform/src","10_jax_exploration_platform/scripts").decode().splitlines()
assert len(source_names)==286
assert PREFIX+"tests/test_monitor_direct_unit_display.py" not in source_names
metadata=ROOT/".review_metadata/direct_unit_monitor_display"
protected={name:hashlib.sha256(before[name].encode()).hexdigest() for name in before if name!="unit_rows"}
receipt=doc(metadata/("source_function_and_closure_check_"+STEP+".json"),{
 "schema_version":1,"kind":"monitor_display_source_scope_and_source_closure_metadata_check","passed":True,
 "job_id":"6783003","step_id":STEP,"base_commit":BASE,"all_existing_other_function_asts_unchanged":protected,
 "only_existing_function_changed":"unit_rows","only_function_added":"direct_unit_display","only_runtime_import_added":"stat",
 "source_prefix_file_count_before":len(source_names),"source_prefix_file_count_after_exact_new_test_addition":len(source_names)+1,
 "old_packet_still_immutable":True,"new_packet_required":True,"scientific_credit":0,"admission_authority":False,
 "numerical_imports":[],"gpu_operations":0,"remote_operations":0,"jobs_submitted":False,"monitor_stopped":False})
files={}
for path in [monitor,ROOT/PREFIX/"tests/test_monitor_direct_unit_display.py",*sorted(metadata.rglob("*"))]:
 if path.is_file():
  assert not path.is_symlink()
  raw=path.read_bytes();raw.decode("utf-8")
  files[str(path.relative_to(ROOT))]={"sha256":hashlib.sha256(raw).hexdigest(),"size":len(raw),"mode":stat.S_IMODE(path.stat().st_mode)}
freeze=doc(metadata/"frozen_source_for_independent_review.json",{
 "schema_version":1,"kind":"monitor_direct_preparation_and_owned_execution_display_source_freeze",
 "created_at_utc":datetime.now(timezone.utc).isoformat(),"base_commit":BASE,"files":files,
 "source_scope_check":receipt,"no_actual_monitor_deployment":True,"scientific_credit":0,"admission_authority":False})
handoff=doc(metadata/"author_handoff.json",{
 "schema_version":1,"kind":"monitor_direct_preparation_and_owned_execution_display_author_handoff",
 "source_freeze":freeze,"worktree":str(ROOT),"base_commit":BASE,
 "before":"A direct prepared measurement without progress was omitted; prepared progress phases could be displayed as running.",
 "after":"Display the prepared direct measurement once as planned; metadata binding and owned registration alone stay planned. Running requires exact owned registration, a recognized numerical phase and matching hardware process birth. Unknown metadata remains unknown.",
 "scientific_scope":"Display and metadata only. Exact staged numerical consumers and independent actual reviews remain the scientific authority.",
 "checks":{"first_focused_451":"Six methods plus existing six reviewed reporting joins passed; the seventh fixture omitted variant and its failure is preserved.","affected_fixture_452":"The one corrected fixture method passed without a monitor source change.","new_controller_binding_454":"Only the new exact controller metadata binding and wrong-plan negative passed."},
 "existing_reporting_index":"Six actual approved entries retained; H100 common timing copies one and two remain due. No future result or review hash was invented.",
 "fresh_packet_changes_required":["Pin a reviewed new monitor source commit and archive; unchanged old packet is not reusable.","Source prefix count is exactly 287 with this one added test, compared with the old 286; derive new full artifact count from the actual fresh packet.","Retarget and independently review the once-only monitor submission/release utility exact packet and counts.","After real timing packets exist, root may freeze a separate eight-unit display index with actual prepared references and no future result hashes.","Verify new actual runtime/source/archive/allocation/PID/cgroup/log descriptor and first full report independently before checking old auxiliary-step drain and stopping old monitor."],
 "jobs_submitted":False,"monitor_stopped":False,"remote_operations":0,"numerical_imports":[],"gpu_operations":0,"scientific_credit":0,"admission_authority":False})
print(json.dumps({"freeze":freeze,"handoff":handoff,"files":len(files),"source_scope_check":receipt},sort_keys=True))

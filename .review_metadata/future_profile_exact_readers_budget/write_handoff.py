"""Record exact immutable source/test evidence without replaying raw science or large payloads."""
import hashlib
import json
import os
from pathlib import Path
import time
M=Path(__file__).resolve().parent;W=M.parents[1]
def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
freeze=M/'source_freeze.json';frozen=json.loads(freeze.read_bytes())
assert ref(freeze)['sha256']=='b23cf4c912d8207f55b1671ce89b620f30800c62782b2334cbcffb1a040a69f5'
for row in frozen['source_and_tests'].values():assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
evidence={p.stem:ref(p) for p in M.glob('*_6821437_*.json')}
handoff={'schema_version':1,'kind':'closed_future_profile_exact_reader_and_metadata_budget_author_handoff',
    'worktree':str(W),'base_commit':frozen['base_commit'],'source_freeze':ref(freeze),
    'runtime_paths':list(frozen['source_and_tests'])[:3],'exact_changed_source_and_tests':frozen['source_and_tests'],
    'design':[
        'All3 metadata roles retain one actual owner/host/boot/monotonic origin:240swork plus60scleanup,300stotal.',
        'Metadata process scopes use positive pinned own-UID/birth/cgroup parent-child chains, kernel child-subreaper closure, exact waits and ECHILD. No process-group equivalence or global scans.',
        'Unknown closure cannot become success. After the deadline there is no new tool or signal; custody remains pending and any eventual late closure is failed/inconclusive.',
        'The outer coordinator never calls new29 common contract in an ambient helper namespace. A fresh child installs its exact per-input source closure.',
        'New29 original qualification/correctness use only exact independently reviewed accepted snapshot facts, with explicit old_saved_consumer_reexecuted=false. Original28/55 raw routes remain unchanged.',
        'Actual new confirmation still needs its genuine exact staged consumer, independent accepted timing/raw review, matching source/configuration/copy seeds and physical target metadata.',
        'Every new completed baseline requires schema2, actual parent+worker+three-reader waits, one origin, exact output hashes and complete scope closures; an old registered unit/proof cannot bypass it.',
        'Helper union adds the saved consumer and shared metadata runtime explicitly (34helpers); protected profile launcher, process/backend modules and numerical core remain unchanged.'
    ],
    'evidence':evidence,
    'focused_results':[
        {'receipt':'runtime_checks_6821437_28','passed':True,'methods':9,'scope':'ownCPU normal/nonzero/timeout/double-orphan closure, budget types, saved mutations, missing APIs and admission refusal'},
        {'receipt':'namespace_checks_6821437_20','passed':True,'scope':'genuine prepared02-24 common29contract and original26Q/C namespaces in3freshchildren; no scientific result opened'},
        {'receipt':'policy_check_6821437_31','passed':True,'scope':'genuine acceptedQ/C fact/binding joins and3negative boundaries; all1938captured payload paths and driver.soopens blocked'},
        {'receipt':'adapter_checks_6821437_24','passed':False,'methods':31,'errors':3,'scope':'old synthetic28fixture accidentally included29th helper; failure/source preserved'},
        {'receipt':'adapter_checks_6821437_25','passed':True,'methods':3,'scope':'only affected old28fixture methods after explicit fixture declaration correction'},
        {'receipt':'adapter_checks_6821437_29','passed':True,'methods':5,'scope':'oldschema/refwrongpath/actualfailed-parentoutput/post-core metadata wrappers'},
        {'receipt':'adapter_checks_6821437_34','passed':True,'methods':4,'scope':'affected parent/savedbaseline/post-core wrappers after exactstage/factproof additions'},
        {'job_id':'6821437','step_id':'35','passed':True,'scope':'frozen6source/test parses,3002protected tracked source files,13frozenNUMfiles,unchangedcapturecore/math/source gates/old28historytail'}
    ],
    'preserved_failures':[
        'Step16 in-memory integer PID keys versus serialized proof string keys; runtime_before_key_canonicalization.py retained.',
        'Step22 transient positive-owned Git child had actual empty cmdline while stat wasR; final record retains exact rawhex/hash and explicitly records command_observed=false instead of inventing a command. Actual root reader command still must be observed.',
        'Step24 source-only old28syntheticfixture declared29helpers; test_prerequisites_before_old28_fixture_fix.py retained. Only3affectedmethods rerun.',
        'Earlier unpersisted exploratory parser/import attempts are not counted as passing evidence; no guessed step or hash was assigned.'
    ],
    'unperformed_or_remaining':[
        'Independent full-source review is required. No integration commit/push or target launch was performed.',
        'The whole actual before_compute3-reader baseline has not run: genuine accepted new H100timing/result/rawreview is not yet available. No future result/review was fabricated.',
        'Selected scientific Python3.12 still lacks stdlib pidfd exports. This metadata runtime refuses before Popen/native bootstrap; no diagnostic-only native provider is silently enabled for formal profiles.',
        'A separately reviewed explicit process-provider/source/helper declaration and actual selected-Python formal target qualification remain necessary.',
        'Current same-process CUDA UUID/PCI/minor, backend/driver/package/asset/source and actual Nsight owned-process/descriptor-survival qualification remain separate prerequisites.',
        'Ordinary profile preparation/execution/saved acceptance remain unconditionally disabled by require_admission; no worker/currentNUM/fifty-idea/final2B admission or credit is granted.'
    ],
    'old22_42_frozen_component_unchanged':True,'protected_original_numerical_files':13,'protected_tracked_source_files':3002,
    'actual_NUM_or_GPU_operations':0,'new_operational_unit_prepared':False,'source_implementation_ready_for_independent_review':True,
    'full_operational_adapter_qualified':False,'scientific_credit':0,'operational_admission':False,
    'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'created_at_seconds':time.time()}
target=M/'author_handoff.json'
with target.open('x') as handle:
    json.dump(handoff,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(target));print(ref(target)['sha256'])

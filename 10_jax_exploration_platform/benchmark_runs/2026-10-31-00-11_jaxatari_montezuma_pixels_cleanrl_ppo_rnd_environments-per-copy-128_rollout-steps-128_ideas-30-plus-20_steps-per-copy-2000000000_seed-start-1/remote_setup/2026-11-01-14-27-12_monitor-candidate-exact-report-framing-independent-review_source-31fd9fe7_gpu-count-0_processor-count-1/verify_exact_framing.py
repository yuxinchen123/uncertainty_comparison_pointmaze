"""Verify existing candidate evidence without another monitor tick or scheduler action."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

AUDIT=Path(__file__).absolute().parent
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-14-02_pixel-study-monitor_source-commit-31fd9fe7cd96343741e8b48a18019efee9b4ac5c_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
RUN=PACKET.parent.parent
COMMIT='31fd9fe7cd96343741e8b48a18019efee9b4ac5c'
CHECKS=[]
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()

def image(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def doc(path):return json.loads(Path(path).read_bytes())
def ref(path):return {'path':str(path),'sha256':image(path)}
def check(value,message):
 assert value,message
 CHECKS.append(message)
def save(name,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with (AUDIT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(AUDIT/name),'sha256':hashlib.sha256(raw).hexdigest()}

try:
 names=['candidate_full_tick.py','candidate_full_tick_failure.json','candidate_allocation.json','candidate_full_tick_stdout.raw','candidate_full_tick_stderr.raw','candidate_full_tick_report.md','candidate_full_tick_status.json','owned_compute_source_visibility_stdout.raw','owned_compute_source_visibility_stderr.raw']
 before={n:ref(PACKET/n) for n in names}
 snapshot=doc(PACKET/'source_snapshot.json')
 root=Path(snapshot['source_root'])
 source=root/'10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py'
 check(image(source)==snapshot['files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py']['sha256'],'actual monitor source is exact archived31fd source')
 check(image(PACKET/'candidate_full_tick.py')=='6dd5a2feee5cd47d205c7547f6425861d89ed8cb658ca7db0c8d5810071ee999','candidate wrapper is unchanged reviewed source')
 syntax=ast.parse(source.read_text());tick=next(n for n in syntax.body if isinstance(n,ast.FunctionDef) and n.name=='tick')
 expected_text=ast.parse('text = "\\n".join(rows) + "\\n"').body[0]
 expected_print=ast.parse('print(text, flush=True)').body[0]
 expected_report=ast.parse('report = "# Pixel Montezuma study status\\n\\n" + now + "\\n\\n```text\\n" + text + "```\\n"').body[0]
 statements=[ast.dump(n,include_attributes=False) for n in tick.body]
 for expected in (expected_text,expected_print,expected_report):check(statements.count(ast.dump(expected,include_attributes=False))==1,'exact declared renderer statement '+ast.unparse(expected))
 check(statements.index(ast.dump(expected_text,include_attributes=False))<statements.index(ast.dump(expected_print,include_attributes=False))<statements.index(ast.dump(expected_report,include_attributes=False)),'renderer defines table then prints and frames that exact table')
 stdout=(PACKET/'candidate_full_tick_stdout.raw').read_bytes();stderr=(PACKET/'candidate_full_tick_stderr.raw').read_bytes();report=(PACKET/'candidate_full_tick_report.md').read_bytes();status=doc(PACKET/'candidate_full_tick_status.json');failure=doc(PACKET/'candidate_full_tick_failure.json');allocation=doc(PACKET/'candidate_allocation.json')
 check(stdout.endswith(b'\n\n') and not stdout.endswith(b'\n\n\n'),'stdout has exactly one table newline and one print newline')
 table=stdout[:-1]
 framed=b'# Pixel Montezuma study status\n\n'+status['time'].encode()+b'\n\n```text\n'+table+b'```\n'
 check(report==framed,'report equals exact declared prefix/time/fence and unchanged table bytes; remove only print-added final newline')
 check(stdout not in report and table in report,'original containment assertion failed only because print adds one newline outside report table')
 check(stderr==b'' and b'Traceback' not in stdout and b'Traceback' not in report,'actual monitor stdout/report and stderr contain no traceback')
 for heading in (b'TRAINING',b'WAITING',b'HELD',b'FINISHED',b'PLANNED'):
  check(table.splitlines().count(heading)==1,'exactly one '+heading.decode()+' status section')
 check(status['campaign']['queue_counts']=={'completed':120,'running':0,'pending':0,'failed':0},'actual captured campaign has120completed and zero running,pending,failed cases')
 check(status['campaign']['phase']=='adaptive_research_pending','next phase remains adaptive research pending')
 check(isinstance(status['jobs'],list) and all(isinstance(row,list) and len(row)==5 for row in status['jobs']),'actual job schema is five-field lists')
 check([row for row in status['jobs'] if row[0]=='6781231'][0][1]=='RUNNING','old ownedCPU monitor was running during candidate')
 check([row for row in status['jobs'] if row[0]=='6782778'][0][1]=='RUNNING','new qualification was running during candidate')
 check(status['final_training']['state']=='disabled' and status['final_training']['units']==[],'final training remains unadmitted and disabled')
 check(all(k in status for k in ['site_validation','allocation_observations','aces_allocation_observations','aces_full_site_checks']),'all existing UVA/ACES operational status fields retained')
 observer=[x['row'] for x in status['aces_allocation_observations'] if x['row']['slurm_job_id']=='2274380']
 check(len(observer)==1 and observer[0]['state']=='completed' and observer[0]['qualification_verified'] is False and observer[0]['workers_permitted'] is False,'ACES observer remains metadata-only completed with no admission')
 check(any('gather-minibatches_allocator-cuda-async' in x['run'] for x in status['units']),'long auxiliary gather-minibatches record remains present')
 check(all(x.get('scientific_hypothesis_credit',0)==0 for x in status['aces_allocation_observations']),'operational observers grant zero scientific credit')
 check(failure['type']=='AssertionError' and failure['job_id']=='6781231' and failure['step_id']=='119' and failure['blocked_action_attempts']==[] and failure['job_actions']==0,'actual failed wrapper receipt retains exact step119 and no blocked or job action')
 wrapper=(PACKET/'candidate_full_tick.py').read_text();lines=wrapper.splitlines()
 check(lines[146].strip()=="assert raw_stdout in report and status['campaign']['queue_counts'].get('completed', 0) >= before_counts.get('completed', 0)",'failure line147 is exact report containment assertion')
 check('line 147, in main' in failure['traceback'] and lines[144].strip()=='assert before == after and not BLOCKED and complete is False','prior protected equality/no-block/complete-false assertion completed before recorded line147 failure')
 check(all(k in allocation for k in ['job_id','step_id','uid','hostname','pid','birth','cgroup','job','node']) and allocation['job_id']=='6781231' and allocation['step_id']=='119' and allocation['old_monitor_guard_passed'] is True,'actual owned numeric step119 resource guard evidence retained')
 visible=json.loads((PACKET/'owned_compute_source_visibility_stdout.raw').read_bytes())
 check((PACKET/'owned_compute_source_visibility_stderr.raw').read_bytes()==b'','actual source visibility check stderr empty')
 check(str(COMMIT) in json.dumps(visible),'actual source verifier identifies exact31fd source')
 protected={str(RUN/n):image(RUN/n) for n in ['next_actions.json','campaign_provenance.json','SOURCE_AND_GATES_READY.json']}
 static=doc(PACKET/'metadata_verification.json');protected_static=static.get('protected_before',{})
 if protected_static:
  check(all(image(path)==digest for path,digest in protected_static.items()),'all frozen preparation protected187 bytes remain unchanged')
 after={n:ref(PACKET/n) for n in names};check(before==after,'all original failed-candidate evidence bytes remain unchanged by independent verification')
 result={'schema_version':1,'kind':'existing_candidate_monitor_exact_framing_independent_verification','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'source_commit':COMMIT,'audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'candidate_job_id':'6781231','candidate_step_id':'119','original_wrapper_verification_failed':True,'original_failure_preserved':ref(PACKET/'candidate_full_tick_failure.json'),'monitor_tick_returned_before_failure':True,'monitor_complete_return':False,'exact_difference':'stdout includes print-added final newline after the already newline-terminated table; report frames table without that extra print newline','arbitrary_whitespace_normalization':False,'full_tick_rerun':False,'checks':CHECKS,'check_count':len(CHECKS),'source_visibility':visible,'actual_counts':status['campaign']['queue_counts'],'five_sections_present':True,'actual_candidate_protected_equality_assertion_passed_before_failure':True,'candidate_protected_before_after_hash_maps_saved':False,'protected_hash_scope_note':'The exact candidate reached line147 after its line145 equality/no-block/completion assertion; its in-memory before/after maps were not serialized before failure. These current hashes are a later independent snapshot, not invented pre-candidate maps.','current_protected_hashes':protected,'candidate_inputs_before':before,'candidate_inputs_after':after,'numerical_imports':0,'network_calls':0,'scheduler_queries_by_this_verifier':0,'new_jobs':0,'release_calls':0,'cancel_calls':0,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'worker_admitted':False}
 print(json.dumps({'passed':True,'receipt':save('review.json',result),'check_count':len(CHECKS),'audit_step':os.environ['SLURM_STEP_ID']}))
except BaseException as error:
 save('failure.json',{'type':type(error).__name__,'error':str(error),'checks_before_failure':CHECKS,'audit_step':os.environ['SLURM_STEP_ID'],'full_tick_rerun':False,'job_actions':0});raise

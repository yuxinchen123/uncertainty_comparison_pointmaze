import pathlib,json,hashlib,datetime,subprocess,os
R=pathlib.Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1");M=pathlib.Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-14-02_pixel-study-monitor_source-commit-31fd9fe7cd96343741e8b48a18019efee9b4ac5c_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0");N=pathlib.Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-08-35_pixel-study-monitor_source-commit-fb3071f3772ed836144f5876dbdda1113a5339d5_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0")
def image(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,a):
 raw=a if isinstance(a,bytes) else (json.dumps(a,indent=2,allow_nan=False)+'\n').encode()
 with (M/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(M/name),'sha256':hashlib.sha256(raw).hexdigest()}
review=R/'remote_setup/2026-11-01-14-47-43_new-persistent-monitor-runtime-and-first-report-independent-review_job-6782816_source-31fd9fe7_gpu-count-0_processor-count-1/review.json'
assert image(review)=='e069a695ee0c2eb78189e8a806bf4cca2f80c4cb423700d05d63a7bee0293a78'
assert json.loads(review.read_text())['passed']
assert not (M/'old_monitor_cancel_intent.json').exists()
ids=(R/'slurm/submitted_jobids_uva.txt').read_text().splitlines()
assert ids.count('6781231')==ids.count('6782816')==1
rows=[x.split('\t') for x in (R/'slurm/submissions.tsv').read_text().splitlines() if x.split('\t')[0]=='6781231']
assert rows==[['6781231','monitor','fb3071f3772ed836144f5876dbdda1113a5339d5',str(N/'monitor.sbatch')]]
stepcommand=['squeue','-h','-s','-j','6781231','-o','%i'];q=subprocess.run(stepcommand,capture_output=True,timeout=20)
assert q.returncode==0 and q.stderr==b''
active=q.stdout.decode().splitlines();assert active and all(x in ['6781231.batch','6781231.extern'] for x in active),active
accountcommand=['sacct','-n','-P','-j','6781231','--format=JobIDRaw%32,State%32,ExitCode']
account=subprocess.run(accountcommand,capture_output=True,timeout=20);assert account.returncode==0
accountrows=[x.split('|') for x in account.stdout.decode().splitlines() if x.strip()]
numeric=[x for x in accountrows if x[0].startswith('6781231.') and x[0].split('.',1)[1].isdecimal()]
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
assert numeric and all(x[1].split()[0].rstrip('+') in terminal for x in numeric),[x for x in numeric if x[1] not in terminal]
proof=save('old_monitor_final_supported_step_drain.json',{'kind':'actual_old_monitor_all_auxiliary_steps_terminal','passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6781231','step_query':{'command':stepcommand,'returncode':q.returncode,'stdout':q.stdout.decode(),'stderr':q.stderr.decode()},'accounting_query':{'command':accountcommand,'returncode':account.returncode,'stdout':account.stdout.decode(),'stderr':account.stderr.decode()},'active_step_ids':active,'numeric_steps_checked':len(numeric),'all_numeric_steps_terminal':True,'previous_unsupported_step_state_format_preserved':True,'new_monitor_review':{'path':str(review),'sha256':image(review)}})
p=R/'next_actions.json';a=json.loads(p.read_text());old=a['persistent_monitor'];assert old['job_id']=='6781231'
a.setdefault('persistent_monitor_history',[]).append({'previous_canonical':old,'replaced_after_actual_review':str(review),'old_step_drain':proof})
new=dict(old);new.update(job_id='6782816',packet=str(M),source_commit='31fd9fe7cd96343741e8b48a18019efee9b4ac5c',state='running_independently_verified',runtime_verified=True,runtime_independent_review=str(review),runtime_independent_review_sha256=image(review),requested_cpus=1,allocated_logical_cpus=2,threads_per_core=2,host_memory_gib=2,gpus=0,interval_seconds=1200,first_startup_log_table_verified=True,first_startup_status_snapshot_claimed=False,first_attributed_full_tick=str(M/'first_independent_full_tick_verification.json'),first_attributed_full_tick_sha256=image(M/'first_independent_full_tick_verification.json'),supports_corrected_a4500_readers=True,future_direct_adapter_supported=False)
a['persistent_monitor']=new;a['replacement_monitor_preparation_current'].update(held=False,released=True,handover_completed=True,release_receipt=str(M/'release_receipt.json'),release_receipt_sha256=image(M/'release_receipt.json'))
t=p.with_name(p.name+'.root-handover.tmp');t.write_text(json.dumps(a,indent=2)+'\n');os.replace(t,p)
p=R/'campaign_provenance.json';a=json.loads(p.read_text());a['persistent_monitor_previous_job_id']='6781231';a['persistent_monitor_job_id']='6782816';a['replacement_monitor_released']=True;a['persistent_monitor_source_commit']='31fd9fe7cd96343741e8b48a18019efee9b4ac5c';a['persistent_monitor_runtime_review_sha256']=image(review);t=p.with_name(p.name+'.root-handover.tmp');t.write_text(json.dumps(a,indent=2)+'\n');os.replace(t,p)
intent=save('old_monitor_cancel_intent.json',{'kind':'exact_owned_old_monitor_cancel_after_verified_replacement','command':['scancel','6781231'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_numeric_steps_terminal':proof,'new_monitor_review':{'path':str(review),'sha256':image(review)},'canonical_new_monitor_job_id':'6782816','automatic_repeat_permitted':False})
result=subprocess.run(['scancel','6781231'],capture_output=True,timeout=30)
receipt=save('old_monitor_cancel_return.json',{'command':['scancel','6781231'],'returncode':result.returncode,'stdout':result.stdout.decode(),'stderr':result.stderr.decode(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'intent':intent,'automatic_repeat_permitted':False})
assert result.returncode==0
print(json.dumps({'canonical_new_monitor':'6782816','old_cancel_request_returncode':result.returncode,'old_numeric_steps_terminal':len(numeric),'receipt':receipt,'old_terminal_confirmation_still_required':True}))

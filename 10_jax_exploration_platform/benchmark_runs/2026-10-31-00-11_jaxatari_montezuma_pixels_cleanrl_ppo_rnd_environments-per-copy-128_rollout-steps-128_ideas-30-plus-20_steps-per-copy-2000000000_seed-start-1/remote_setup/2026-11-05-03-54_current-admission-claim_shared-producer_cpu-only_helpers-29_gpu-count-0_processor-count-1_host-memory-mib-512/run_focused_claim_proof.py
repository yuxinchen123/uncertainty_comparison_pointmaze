import ast, hashlib, importlib.util, json, os, subprocess, sys, time, unittest
from pathlib import Path
w=Path('/p/rlprojects/RND_worktrees/2026-11-05-03-38_direct-current-admission_reviewed-historical-copy_canonical-authority-preserved_helpers-29_gpu-count-0_processor-count-1_host-memory-mib-512'); p=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-05-03-54_current-admission-claim_shared-producer_cpu-only_helpers-29_gpu-count-0_processor-count-1_host-memory-mib-512'); scripts=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0,str(scripts))
import serval02_card1_binding as core
import probe_direct_current_admission as probe
spec=importlib.util.spec_from_file_location('focused_claim_tests',scripts/'tests/test_current_admission_snapshot.py'); tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests));assert result.wasSuccessful()
fixture=p/'helper_receipt_only_fixture_plan.json'; raw=fixture.read_bytes(); value=json.loads(raw)
proof=probe.probe(fixture,hashlib.sha256(raw).hexdigest(),fixture=True,canonical_source=True)
(p/'actual_fixture_claim_probe.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
q=list((Path(value['campaign_run_dir'])/'data').glob('2026-11-01-20-53_original-qualification_h100-nvl_serval02-card-1_*/measurement_plan.json')); assert len(q)==1
expected=core.current_admission_claim(json.loads(q[0].read_bytes()));assert proof['claim']==expected
oldraw=subprocess.check_output(['git','-C',str(w),'show','f7e8f5c8f0a2d1847fff4f93726fce62205c23aa:10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py'])
oldtree=ast.parse(oldraw);newtree=ast.parse((scripts/'serval02_card1_binding.py').read_bytes())
oldf={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,ast.FunctionDef)};newf={n.name:ast.dump(n,include_attributes=False) for n in newtree.body if isinstance(n,ast.FunctionDef)}
changed=[n for n in oldf if oldf[n]!=newf[n]];assert changed==['current_admission_claim'];assert oldf['check_live_admission']==newf['check_live_admission']
assert subprocess.check_output(['git','-C',str(w),'diff','f7e8f5c8f0a2d1847fff4f93726fce62205c23aa','--','10_jax_exploration_platform/src/exploration_platform/atari_pixels'])==b''
assert subprocess.check_output(['git','-C',str(w),'show','f7e8f5c8f0a2d1847fff4f93726fce62205c23aa:10_jax_exploration_platform/scripts/atari_pixel_campaign/accepted_evidence_snapshot.py'])==(scripts/'accepted_evidence_snapshot.py').read_bytes()
summary={'passed':True,'focused_methods':result.testsRun,'actual_new_claim_equals_genuine_legacy_qualification_claim':True,'legacy_qualification_plan':str(q[0]),'changed_binding_functions':changed,'unchanged_live_authority_function':True,'unchanged_snapshot_producer':True,'numerical_source_unchanged':True,'elapsed_seconds':time.monotonic()-started,'fixture_only':True,'actual_new_unit_qualification':False,'gpu_target_checked':False,'scientific_credit':0}
(p/'author_focused_result.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,sort_keys=True))

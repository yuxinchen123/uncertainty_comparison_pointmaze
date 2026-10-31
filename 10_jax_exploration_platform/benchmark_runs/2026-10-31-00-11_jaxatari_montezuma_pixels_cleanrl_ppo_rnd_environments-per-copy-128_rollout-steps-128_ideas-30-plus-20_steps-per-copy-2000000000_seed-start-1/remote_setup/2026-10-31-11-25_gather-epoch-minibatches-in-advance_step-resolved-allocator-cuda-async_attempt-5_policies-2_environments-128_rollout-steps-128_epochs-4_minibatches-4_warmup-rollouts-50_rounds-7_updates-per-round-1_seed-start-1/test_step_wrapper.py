"""Verify the immutable step wrapper with synthetic metadata commands and no GPU process."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class StepWrapperTests(unittest.TestCase):
    """Execute only fake resolver/recorder/worker programs; never start srun or numerical imports."""

    def test_batch_enters_srun_before_resolving_or_exporting_visibility(self):
        batch = (ROOT / "job.sbatch").read_text()
        inside = (ROOT / "inside_step.sh").read_text()
        self.assertIn("exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=threads /bin/bash", batch)
        self.assertNotIn("gpu_uuid=", batch)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", batch)
        self.assertLess(inside.index("--phase before"), inside.index("gpu_uuid=$("))
        self.assertLess(inside.index("gpu_uuid=$("), inside.index("export CUDA_VISIBLE_DEVICES"))
        for name in ("job.sbatch", "inside_step.sh"):
            subprocess.run(["/bin/bash", "-n", str(ROOT / name)], check=True, capture_output=True)

    def test_step_ordinal_is_recorded_then_resolved_before_unchanged_worker(self):
        plan = json.loads((ROOT / "launch_plan.json").read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorder = directory / "record.py"
            recorder.write_text("import json,os,sys\nfrom pathlib import Path\n"
                "phase=sys.argv[sys.argv.index('--phase')+1]\n"
                "p=Path(os.environ['TEST_EVENTS'])\n"
                "with p.open('a') as f:f.write(json.dumps({'phase':phase,'visible':os.environ['CUDA_VISIBLE_DEVICES'],'job_gpus':os.environ['SLURM_JOB_GPUS'],'step_gpus':os.environ['SLURM_STEP_GPUS']})+'\\n')\n")
            resolver = directory / "resolve.py"
            resolver.write_text("import os\nassert os.environ['CUDA_VISIBLE_DEVICES']=='0'\n"
                "assert os.environ['SLURM_JOB_GPUS']=='5'\nassert os.environ['SLURM_STEP_GPUS']=='5'\nprint('GPU-selected')\n")
            worker = directory / "worker.py"
            worker.write_text("import os,sys\nfrom pathlib import Path\n"
                "assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-selected'\n"
                "assert sys.argv[-2:]==['--gpu-uuid','GPU-selected']\n"
                "Path(os.environ['TEST_WORKER']).write_text('executed once')\n")
            contents = (ROOT / "inside_step.sh").read_text()
            contents = contents.replace(shlex.join(["/usr/bin/python3", str(ROOT / "record_step_metadata.py")]), shlex.join([sys.executable, str(recorder)]))
            contents = contents.replace(shlex.join(plan["resolver_command"]), shlex.join([sys.executable, str(resolver)]))
            contents = contents.replace(shlex.join(plan["worker_command_without_uuid"]), shlex.join([sys.executable, str(worker)]))
            script = directory / "step.sh"
            script.write_text(contents)
            events = directory / "events.jsonl"
            executed = directory / "worker-result.txt"
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "5", "SLURM_STEP_GPUS": "5",
                "TEST_EVENTS": str(events), "TEST_WORKER": str(executed)}
            subprocess.run(["/bin/bash", str(script)], env=environment, capture_output=True, check=True, text=True)
            records = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([(row['phase'], row['visible']) for row in records], [("before", "0"), ("resolved", "0"), ("after", "GPU-selected")])
            self.assertTrue(all(row['job_gpus']==row['step_gpus']=='5' for row in records))
            self.assertEqual(executed.read_text(), 'executed once')
            events.unlink()
            executed.unlink()
            resolver.write_text("raise SystemExit(2)\n")
            failed = subprocess.run(["/bin/bash", str(script)], env=environment, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertFalse(executed.exists())
            self.assertEqual([json.loads(line)['phase'] for line in events.read_text().splitlines()], ['before'])

    def test_metadata_snapshot_omits_unrelated_environment_and_credentials(self):
        specification = importlib.util.spec_from_file_location('step_metadata_test', ROOT / 'record_step_metadata.py')
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        snapshot = module.snapshot('before', environment={'CUDA_VISIBLE_DEVICES':'0','SLURM_JOB_GPUS':'5','SECRET_TOKEN':'never-record'})
        self.assertEqual(snapshot['environment'], {'SLURM_JOB_GPUS':'5','CUDA_VISIBLE_DEVICES':'0'})
        self.assertNotIn('never-record', json.dumps(snapshot))
        self.assertTrue(snapshot['process_start_time'].isdigit())


if __name__ == '__main__':
    unittest.main()

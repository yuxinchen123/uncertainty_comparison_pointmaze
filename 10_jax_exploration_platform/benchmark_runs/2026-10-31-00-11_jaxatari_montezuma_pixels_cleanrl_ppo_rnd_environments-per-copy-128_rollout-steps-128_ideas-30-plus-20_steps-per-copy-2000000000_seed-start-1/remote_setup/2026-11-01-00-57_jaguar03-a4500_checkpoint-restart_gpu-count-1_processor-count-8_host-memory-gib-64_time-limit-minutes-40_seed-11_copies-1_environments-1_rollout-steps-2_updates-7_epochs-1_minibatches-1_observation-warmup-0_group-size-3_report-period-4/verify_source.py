"""Verify the complete fixed source and prepared component configuration without numerical imports."""
import hashlib,json,os,stat,sys,time
from pathlib import Path
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-00-57_jaguar03-a4500_checkpoint-restart_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-40_seed-11_copies-1_environments-1_rollout-steps-2_updates-7_epochs-1_minibatches-1_observation-warmup-0_group-size-3_report-period-4')
EXPECTED={'source_snapshot.json': '719319c537039146cf4b81bf53c7c308fd2d5c513032be832710483d31613433', 'config_resolved.yaml': '7c476fa60b29c48acd029695a4f5d5b18ac8dc0d7ed800833eb58353730763ac', 'environment_registry.json': '92e7340d752951fab2432686adb2dd0a54aa49e6cf1177109d0a46bcc2ab2bb4', 'allocation_guard.py': '983a0fd0250d09f5d376d19b5ccb1cd4f6f20b9ab2fd9caaaaf4e520772e9960'}

def main():
    """Wait at most120seconds for fixed preparation files, then verify their exact bytes and source tree."""
    deadline=time.monotonic()+120
    while any(not (PACKET/name).is_file() for name in EXPECTED):
        if time.monotonic()>=deadline:raise RuntimeError('component preparation files are not visible on this compute host')
        time.sleep(1)
    for name,checksum in EXPECTED.items():
        path=PACKET/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('prepared input changed: '+name)
    manifest=json.loads((PACKET/'source_snapshot.json').read_text());source=Path(manifest['source_root'])
    actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    if actual!=set(manifest['files']):raise ValueError('component snapshot has missing or additional source files')
    for name,record in manifest['files'].items():
        path=source/name;contents=path.read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()
        if path.is_symlink() or path.resolve()!=path.absolute() or hashlib.sha256(contents).hexdigest()!=record['sha256'] or blob!=record['git_blob']:
            raise ValueError('component snapshot differs from its committed Git source: '+name)
        if stat.S_IMODE(path.stat().st_mode)&0o111!=(0o111 if record['git_mode']=='100755' else 0):raise ValueError('component source executable mode changed')
    for name,checksum in manifest['pixel_reference_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=checksum:raise ValueError('pixel numerical source differs from D28')
    registry=json.loads((PACKET/'environment_registry.json').read_text())
    if hashlib.sha256(Path(registry['path']).read_bytes()).hexdigest()!=registry['sha256'] or not os.access(registry['python'],os.X_OK):
        raise ValueError('canonical pixel environment registry or executable changed')
    if hashlib.sha256(Path(registry['python']).read_bytes()).hexdigest()!=registry['python_binary_sha256']:raise ValueError('canonical pixel Python binary changed')
    if any(name.split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch'} for name in sys.modules):raise RuntimeError('source verification imported a numerical runtime')
    print(json.dumps({'source_commit':manifest['source_commit'],'files_verified':len(actual),'pixel_numerical_files_verified':len(manifest['pixel_reference_sha256']),'numerical_imports':False}))
if __name__=='__main__':main()

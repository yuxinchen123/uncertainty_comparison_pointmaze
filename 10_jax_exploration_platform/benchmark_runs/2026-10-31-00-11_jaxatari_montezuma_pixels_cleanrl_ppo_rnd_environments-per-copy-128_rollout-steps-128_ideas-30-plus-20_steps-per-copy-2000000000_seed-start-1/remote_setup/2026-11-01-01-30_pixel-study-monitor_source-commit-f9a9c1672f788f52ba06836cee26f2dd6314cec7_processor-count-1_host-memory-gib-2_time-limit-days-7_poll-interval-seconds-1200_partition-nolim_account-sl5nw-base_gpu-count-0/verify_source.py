"""Verify static prepared source inputs without running a monitor tick or numerical code."""
import hashlib,json,os,stat,sys,time
from pathlib import Path
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-01-30_pixel-study-monitor_source-commit-f9a9c1672f788f52ba06836cee26f2dd6314cec7_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
EXPECTED={'source_snapshot.json': 'c333c4d2ae4af39512f839be442f08a18de1cf70af9ea3a210d8da90c1489163', 'config_resolved.yaml': '2101fdf5d7f211b60e45277f7c6f444e81ec9edc5027349c9de7b06e398e57ed', 'allocation_guard.py': 'ce4efb50b31358c9406e81a8bbdde83ac47521c5b2bc0797e10780e3b6cbee65', 'source_git_archive.tar': '9a278cd8ee8a55084e66fa3810c6456ba5a5f6d2f96d637e1ea8a60540054678'}

def main():
    """Wait for complete source visibility, then check Git blobs, SHA256 and executable bits."""
    deadline=time.monotonic()+120
    while any(not (PACKET/name).is_file() for name in EXPECTED):
        if time.monotonic()>=deadline:raise RuntimeError('prepared monitor inputs not visible on compute')
        time.sleep(1)
    for name,checksum in EXPECTED.items():
        if (PACKET/name).is_symlink() or hashlib.sha256((PACKET/name).read_bytes()).hexdigest()!=checksum:raise ValueError('prepared monitor input changed: '+name)
    manifest=json.loads((PACKET/'source_snapshot.json').read_text());source=Path(manifest['source_root'])
    while any(not (source/name).is_file() for name in manifest['files']):
        if time.monotonic()>=deadline:raise RuntimeError('prepared monitor source not fully visible on compute')
        time.sleep(1)
    paths=list(source.rglob('*'))
    if any(path.is_symlink() for path in paths):raise ValueError('monitor source snapshot cannot contain symlinks')
    if {path.relative_to(source).as_posix() for path in paths if path.is_file()}!=set(manifest['files']):raise ValueError('monitor source snapshot has missing or additional files')
    for name,record in manifest['files'].items():
        path=source/name;data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if hashlib.sha256(data).hexdigest()!=record['sha256'] or blob!=record['git_blob']:raise ValueError('monitor source differs from its committed Git blob: '+name)
        if stat.S_IMODE(path.stat().st_mode)&0o111!=(0o111 if record['git_mode']=='100755' else 0):raise ValueError('monitor executable mode differs: '+name)
    if any(name.split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch'} for name in sys.modules):raise RuntimeError('monitor source preparation imported numerical code')
    print(json.dumps({'source_commit':manifest['source_commit'],'files_verified':len(manifest['files']),'full_tick_executed':False,'numerical_imports':False}))
if __name__=='__main__':main()

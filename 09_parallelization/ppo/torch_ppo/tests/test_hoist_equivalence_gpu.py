"""GPU-only: hoisting work out of the rollout loop must not change what the trainer computes.

Round two moved the critic forward, the log-probability and the RND bonus out of the T-step
rollout loop into single wide passes in the post-rollout body. The claim is that this is exact:
each is a pure function of data the loop already stores and of parameters that do not change
during a rollout. This test checks that claim against the implementation as it was BEFORE the
change, taken from a git revision rather than a copy that could drift.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_hoist_equivalence_gpu.py [git-rev]
"""
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
BASE = HERE.parent.parent.parent
REPO = BASE.parent
REL = "09_parallelization/ppo/torch_ppo/torch_ppo_rnd.py"
OLD = Path("/localtmp/sl5nw/torch_ppo_rnd_prehoist.py")
FIELDS = ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int", "vext_old",
          "rnd_input", "rnd_tf"]
CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1,
           rollout_mode="compile-step", fused_adam=True)


def load_old(rev):
    """Import the trainer as of a git revision, with its path lookup made absolute."""
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{REL}"],
                         capture_output=True, text=True, check=True).stdout
    src = src.replace("BASE = Path(__file__).resolve().parent.parent.parent",
                      f'BASE = Path("{BASE}")')
    OLD.parent.mkdir(parents=True, exist_ok=True)
    OLD.write_text(src)
    import importlib.util
    sys.path.insert(0, str(HERE.parent))
    spec = importlib.util.spec_from_file_location("trainer_prehoist", OLD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    import torch_ppo_rnd as new_mod
    old_mod = load_old(rev)

    # the pre-hoist revision has no compiled-post knob; compare the hoist alone
    old = old_mod.PPORND(old_mod.PPOConfig(**CFG), device="cuda")
    new = new_mod.PPORND(new_mod.PPOConfig(compile_post=False, **CFG), device="cuda")
    torch.manual_seed(5); old.prime_obs_rms()
    torch.manual_seed(5); new.prime_obs_rms()

    def rel(a, b):
        """Deviation relative to the magnitude of the quantity itself."""
        scale = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
        return (a - b).abs().max().item() / scale

    # Iteration 0 is the isolated comparison: both trainers hold identical parameters and
    # identical environment state, so every difference here comes from this iteration alone.
    # Later iterations chain (a last-bit difference changes an action, which changes the next
    # observation), so they are reported as drift rather than gated.
    print(f"pre-hoist revision: {rev}")
    gate_worst, gate_name = 0.0, ""
    for it in range(3):
        torch.manual_seed(900 + it)
        bo = old.rollout()
        torch.manual_seed(900 + it)
        bn = new.rollout()
        worst = max((rel(bo[k], bn[k]), k) for k in FIELDS)
        print(f"  iteration {it}: worst relative field deviation {worst[0]:.3e} ({worst[1]})")
        if it == 0:
            gate_worst, gate_name = worst
        torch.manual_seed(950 + it)
        old.update_epoch_minibatch(bo)
        torch.manual_seed(950 + it)
        new.update_epoch_minibatch(bn)

    worst_param = max((a - b).abs().max().item()
                      for a, b in zip(old.trainable, new.trainable))
    print(f"drift after 3 iterations: worst parameter deviation {worst_param:.3e}")
    assert gate_worst <= 1e-5, f"the hoist changed the computation ({gate_name})"
    assert worst_param <= 1e-5, "parameters drifted more than float32 accumulation explains"
    print("hoist equivalence: PASS")


if __name__ == "__main__":
    main()

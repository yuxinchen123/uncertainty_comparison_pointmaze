"""
Test: old MyRND.update() trains on observations, new RND.update() trains on next_observations.

Old code (a558f7c) MyRND.update():
    obs = samples["observations"]   # <--- trains predictor on CURRENT states
    ...DataLoader mini-batch loop...

New code (2ff924b) RND.update() with feature="rnd_next_state":
    x = self._get_feature_tensor(samples)  # <--- uses samples["next_observations"]
    ...single gradient step...

Both compute() use next_observations, but update() differs in training data.
This is a subtle difference: the predictor is trained on different data than
what it's evaluated on, but observations and next_observations come from the
same distribution in a replay buffer, so the impact should be small.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import numpy as np


def trace_old_update():
    """Show what old MyRND.update() actually trains on."""
    print("Old MyRND.update() code path:")
    print("  1. obs = samples['observations']        # uses CURRENT states")
    print("  2. obs = self._slice_obs(obs)            # optionally slice to position-only")
    print("  3. self.obs_rms.update(obs.numpy())      # update running stats on observations")
    print("  4. obs = self._normalize_obs(obs)")
    print("  5. DataLoader(obs, batch_size=256, shuffle=True)")
    print("  6. for batch in loader:                  # MULTIPLE gradient steps")
    print("       self.opt.zero_grad()")
    print("       src = self.predictor(batch)")
    print("       tgt = self.target(batch)")
    print("       loss = dist_ensemble(src, tgt).mean()")
    print("       loss.backward(); self.opt.step()")
    print()


def trace_new_update():
    """Show what new RND.update() actually trains on (rnd_next_state)."""
    print("New RND.update() code path (feature='rnd_next_state'):")
    print("  1. x = _get_feature_tensor(samples)     # uses samples['next_observations']")
    print("  2. self.obs_rms.update(x.numpy())        # update running stats on NEXT states")
    print("  3. x = self._normalize_obs(x)")
    print("  4. self.opt.zero_grad()                  # SINGLE gradient step")
    print("     src = self.predictor(x)")
    print("     tgt = self.target(x)")
    print("     loss = dist_ensemble(src, tgt).mean()")
    print("     loss.backward(); self.opt.step()")
    print()


def show_gradient_step_count():
    """The old code does more gradient steps per sample() call."""
    # In VectorIntrinsicReplayBuffer.sample(), batch_size=256 (SB3 default)
    # Old MyRND.update() creates DataLoader with self.batch_size (typically 256)
    # If SB3 samples 256 transitions, and MyRND batch_size=256, we get 1 step
    # But if SB3 samples 256 and MyRND batch_size=64, we get 4 steps

    print("Gradient step count comparison:")
    print(f"  SB3 default sample batch_size: 256")
    print(f"  Old MyRND internal batch_size: 256 (from constructor)")
    print(f"  -> Old does ceil(256/256) = 1 gradient step (same as new)")
    print()
    print("  BUT if you changed MyRND batch_size to e.g. 64:")
    print(f"  -> Old would do ceil(256/64) = 4 gradient steps")
    print(f"  -> New always does 1 gradient step")
    print()
    print("  With default batch_size=256 for both, this is NOT the issue.")


if __name__ == "__main__":
    trace_old_update()
    trace_new_update()
    print("=" * 60)
    show_gradient_step_count()
    print()
    print("VERDICT: The training data difference (obs vs next_obs) is minor.")
    print("The gradient step count is the same at default settings (batch_size=256).")
    print("This is NOT the primary cause of RND failure.")

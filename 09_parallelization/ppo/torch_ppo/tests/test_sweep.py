"""Tests for the learning-rate sweep across copy groups (round 3).

A sweep trains several groups of copies at once, each group at its own learning rate. The
properties that make the result trustworthy:

  1. a sweep whose rates are all equal reproduces the ordinary uniform-rate run,
  2. a group whose rate is zero never moves, while the others do,
  3. changing one group's rate leaves every other group's parameters bitwise unchanged,
  4. paired seeding gives copy k of every group the same initial weights and the same
     environments, so a difference between groups is the rate's doing and nothing else,
  5. distinct seeding gives every copy its own stream, as before.

Run: PYTHONNOUSERSITE=1 <python> test_sweep.py   (CPU)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND, sweep_config  # noqa: E402

SMALL = dict(n_envs=2, num_steps=8, obs_norm_init_iters=1, rollout_mode="eager",
             capture_update=False, one_graph=False, fused_adam=False, tf32=False,
             compile_post=False)


def build(**kw):
    """A small CPU trainer with the given overrides."""
    return PPORND(PPOConfig(**{**SMALL, **kw}), device="cpu")


def run(trainer, iters=3, seed=17):
    """A few deterministic iterations."""
    torch.manual_seed(seed)
    trainer.prime_obs_rms()
    for it in range(iters):
        torch.manual_seed(400 + it)
        b = trainer.rollout()
        torch.manual_seed(450 + it)
        trainer.update_epoch_minibatch(b)


def test_uniform_sweep_matches_plain_run():
    """All rates equal: the sweep path must reproduce the ordinary path."""
    plain = build(n_copies=4, learning_rate=3e-4)
    swept = PPORND(sweep_config([3e-4, 3e-4], 2, sweep_seed_mode="distinct", **SMALL),
                   device="cpu")
    run(plain); run(swept)
    worst = max((a - b).abs().max().item() for a, b in zip(plain.trainable, swept.trainable))
    print(f"uniform sweep vs plain run: worst parameter deviation {worst:.3e}")
    assert worst <= 1e-6, "the sweep path changed the uniform-rate result"
    print("ok test_uniform_sweep_matches_plain_run")


def test_zero_rate_group_is_frozen():
    """A group at rate zero must not move; a group at a real rate must."""
    t = PPORND(sweep_config([0.0, 3e-3], 2, **SMALL), device="cpu")
    before = [p.detach().clone() for p in t.trainable]
    run(t)
    frozen = max((p[:2] - b[:2]).abs().max().item() for p, b in zip(t.trainable, before))
    moved = max((p[2:] - b[2:]).abs().max().item() for p, b in zip(t.trainable, before))
    print(f"zero-rate group movement {frozen:.3e}; trained group movement {moved:.3e}")
    assert frozen == 0.0, "the zero-rate group moved"
    assert moved > 0.0, "the trained group did not move"
    print("ok test_zero_rate_group_is_frozen")


def test_groups_do_not_influence_each_other():
    """Changing one group's rate must leave the other group's parameters bitwise identical."""
    a = PPORND(sweep_config([3e-4, 1e-3], 2, **SMALL), device="cpu")
    b = PPORND(sweep_config([3e-4, 5e-2], 2, **SMALL), device="cpu")   # second group differs
    run(a); run(b)
    same = all(torch.equal(pa[:2], pb[:2]) for pa, pb in zip(a.trainable, b.trainable))
    differ = any(not torch.equal(pa[2:], pb[2:]) for pa, pb in zip(a.trainable, b.trainable))
    print(f"unchanged group identical: {same}; changed group differs: {differ}")
    assert same, "changing one group's learning rate moved another group"
    assert differ, "changing the learning rate had no effect, so the test proves nothing"
    print("ok test_groups_do_not_influence_each_other")


def test_paired_seeding_gives_groups_the_same_start():
    """Paired seeding: copy k of every group starts identical and sees the same environments."""
    t = PPORND(sweep_config([3e-4, 1e-3], 3, sweep_seed_mode="paired", **SMALL), device="cpu")
    for p in t.trainable:
        assert torch.equal(p[:3], p[3:]), "paired groups did not start from the same weights"
    obs = t.env.reset()
    assert torch.equal(obs[:3], obs[3:]), "paired groups did not get the same environments"
    print("ok test_paired_seeding_gives_groups_the_same_start")


def test_distinct_seeding_separates_every_copy():
    """Distinct seeding: no two copies share weights or environments."""
    t = PPORND(sweep_config([3e-4, 1e-3], 3, sweep_seed_mode="distinct", **SMALL), device="cpu")
    w = t.actor["W0"]
    assert not torch.equal(w[:3], w[3:]), "distinct seeding still paired the groups"
    obs = t.env.reset()
    assert not torch.equal(obs[:3], obs[3:]), "distinct seeding still paired the environments"
    print("ok test_distinct_seeding_separates_every_copy")


def test_sweep_under_capture_gpu():
    """GPU only: the swept per-copy Adam must survive graph capture and still isolate groups.

    Runs a sweep whose first group has a zero learning rate as a captured iteration graph, and
    checks that the zero-rate group stays frozen through the replays while the other group
    trains. A captured optimizer that had baked in one rate would move the frozen group.
    """
    import torch
    if not torch.cuda.is_available():
        print("skip test_sweep_under_capture_gpu (no GPU)")
        return
    gpu = dict(n_envs=4, num_steps=32, obs_norm_init_iters=1)
    capt = PPORND(sweep_config([0.0, 1e-3], 2, **gpu), device="cuda")
    torch.manual_seed(31)
    capt.prime_obs_rms()
    capt._build_iteration_graph()
    before = [p.detach().clone() for p in capt.trainable]
    for _ in range(3):
        capt.iteration_captured()
    frozen = max((p[:2] - b0[:2]).abs().max().item()
                 for p, b0 in zip(capt.trainable, before))
    moved = max((p[2:] - b0[2:]).abs().max().item()
                for p, b0 in zip(capt.trainable, before))
    print(f"captured sweep: zero-rate group movement {frozen:.3e}, "
          f"trained group movement {moved:.3e}")
    assert frozen == 0.0, "the captured sweep moved the zero-rate group"
    assert moved > 0.0, "the captured sweep did not train the other group"
    print("ok test_sweep_under_capture_gpu")


if __name__ == "__main__":
    test_uniform_sweep_matches_plain_run()
    test_zero_rate_group_is_frozen()
    test_groups_do_not_influence_each_other()
    test_paired_seeding_gives_groups_the_same_start()
    test_distinct_seeding_separates_every_copy()
    test_sweep_under_capture_gpu()

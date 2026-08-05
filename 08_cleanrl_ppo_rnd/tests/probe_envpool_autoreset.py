"""Measure, rather than assume, what envpool returns on the step that ends an episode.

The CleanRL docs for ppo_rnd_envpool.py record a bug: under envpool<=0.6.4 the `obs` returned by
`env.step(a)` on a terminating step is the terminal state s_last, while gym's vectorised env returns
the first state of the next episode s_new, so s_last is "off by one". Everything we do to fix that
depends on the exact timeline, so this probe reads it off the installed envpool directly.

Two probes:

1. CartPole-v1 — the cheapest unambiguous test. A CartPole reset state is drawn uniformly from
   [-0.05, 0.05]^4, and a terminal state has |x| > 2.4 or |theta| > 0.2095. So "is this obs a reset
   state or a terminal state" is decidable by looking at the numbers.
2. MontezumaRevenge-v5 with the exact options ppo_rnd_envpool.py passes. Atari observations are
   84x84 frame stacks, so instead of reading the pixels we use envpool's own `elapsed_step` counter
   in `info`, which restarts at a small value on a fresh episode.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python probe_envpool_autoreset.py
"""

import numpy as np
import envpool


def probe_cartpole(max_steps=400):
    """Step one CartPole env until two episodes have ended, recording the obs around each ending."""
    # One env keeps the timeline unambiguous: with several envs the terminations interleave.
    env = envpool.make("CartPole-v1", env_type="gym", num_envs=1, seed=0)
    obs = env.reset()

    # CartPole's termination thresholds, from the classic-control definition. A state outside either
    # threshold is terminal; a freshly reset state is inside [-0.05, 0.05] on every coordinate.
    x_limit, theta_limit = 2.4, 0.2095

    def classify(o):
        """Name an observation a reset state, a terminal state, or an ordinary in-episode state."""
        x, _, theta, _ = o
        if abs(x) > x_limit or abs(theta) > theta_limit:
            return "TERMINAL"
        if abs(o).max() <= 0.05:
            return "RESET-LIKE"
        return "mid-episode"

    rng = np.random.RandomState(0)
    timeline, dones_seen = [], 0
    for t in range(max_steps):
        a = rng.randint(0, 2, size=(1,))
        obs, reward, done, info = env.step(a)
        timeline.append(
            {
                "t": t,
                "obs": np.round(obs[0], 4).tolist(),
                "class": classify(obs[0]),
                "reward": float(reward[0]),
                "done": bool(done[0]),
                "elapsed_step": int(info["elapsed_step"][0]) if "elapsed_step" in info else None,
            }
        )
        if done[0]:
            dones_seen += 1
            if dones_seen == 2:
                break
    env.close()
    return timeline


def probe_atari(steps_after_first_done=4):
    """Step MontezumaRevenge with ppo_rnd_envpool.py's options and read info around a termination."""
    # These are exactly the options cleanrl/ppo_rnd_envpool.py passes to envpool.make.
    env = envpool.make(
        "MontezumaRevenge-v5",
        env_type="gym",
        num_envs=1,
        episodic_life=True,
        reward_clip=True,
        seed=1,
        repeat_action_probability=0.25,
    )
    obs = env.reset()
    print(f"  atari reset obs: shape={obs.shape} dtype={obs.dtype}")

    rng = np.random.RandomState(1)
    n_actions = env.action_space.n
    timeline, after_done = [], -1
    # An Atari life is lost only after a few hundred steps of random play, so allow a long budget.
    for t in range(20000):
        a = rng.randint(0, n_actions, size=(1,))
        obs, reward, done, info = env.step(a)
        record = {
            "t": t,
            "done": bool(done[0]),
            "reward": float(reward[0]),
            # envpool's own per-env counters; these are what a fix would key off.
            "elapsed_step": int(info["elapsed_step"][0]) if "elapsed_step" in info else None,
            "terminated": int(info["terminated"][0]) if "terminated" in info else None,
            "lives": int(info["lives"][0]) if "lives" in info else None,
            # A cheap fingerprint of the frame stack, enough to see whether two steps show the
            # same picture or different ones.
            "obs_sum": int(obs[0, 3].astype(np.int64).sum()),
        }
        timeline.append(record)
        if done[0] and after_done < 0:
            after_done = 0
        elif after_done >= 0:
            after_done += 1
            if after_done >= steps_after_first_done:
                break
    env.close()
    # Keep only the window around the first termination: three steps before, and the steps after.
    first_done = next(i for i, r in enumerate(timeline) if r["done"])
    return timeline[max(0, first_done - 3) : first_done + steps_after_first_done + 1], info.keys()


def main():
    """Run both probes and print the timelines that decide what the fix must do."""
    print(f"envpool {envpool.__version__}")
    print()
    print("=== probe 1: CartPole-v1, one env, steps around the first two terminations ===")
    for r in probe_cartpole():
        if r["done"] or r["class"] != "mid-episode":
            print(
                f"  t={r['t']:<4} done={str(r['done']):<5} class={r['class']:<12} "
                f"elapsed_step={r['elapsed_step']} obs={r['obs']}"
            )
    print()
    print("=== probe 2: MontezumaRevenge-v5 with ppo_rnd_envpool.py's options ===")
    window, info_keys = probe_atari()
    print(f"  info keys: {sorted(info_keys)}")
    for r in window:
        print(
            f"  t={r['t']:<6} done={str(r['done']):<5} terminated={r['terminated']} "
            f"lives={r['lives']} elapsed_step={r['elapsed_step']} reward={r['reward']} "
            f"obs_sum={r['obs_sum']}"
        )
    print()
    print("Read the CartPole block: the step whose done=True shows whether obs is TERMINAL")
    print("(envpool's documented behaviour) or RESET-LIKE (gym's behaviour), and the step after it")
    print("shows which state the agent is actually asked to act on.")


if __name__ == "__main__":
    main()

"""CPU-only regression tests for envpool's next-step autoreset and cleanrl's ppo_rnd_envpool fix.

Run:  pytest -q test_envpool_autoreset.py
No GPU, no torch, no network. Every episode ends after `MAX_EPISODE_STEPS` env steps.
"""
import hashlib

import numpy as np
import envpool

MAX_EPISODE_STEPS = 4


def digest(a):
    """Short stable hash of an array, so 'the same observation' is a single comparable token."""
    return hashlib.sha1(np.ascontiguousarray(a)).hexdigest()[:12]


def make_deterministic_pool(seed=1):
    """A fully deterministic 1-env Atari pool: no sticky actions, no random no-ops, fixed short episodes.

    noop_max=1 + repeat_action_probability=0 make every reset return the SAME observation,
    which is what lets the tests assert an exact, repeating observation sequence.
    """
    return envpool.make(
        "Breakout-v5",
        env_type="gym",
        num_envs=1,
        seed=seed,
        max_episode_steps=MAX_EPISODE_STEPS,
        noop_max=1,
        repeat_action_probability=0.0,
        episodic_life=False,
    )


def reset(env):
    """Normalise reset() across envpool versions: 0.6.6 returns obs, 1.2.5 returns (obs, info)."""
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step(env, action):
    """Normalise step() across the 4-tuple (gym) and 5-tuple (gymnasium) APIs to (obs, rew, done, info)."""
    out = env.step(action)
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        return obs, rew, np.logical_or(term, trunc), info
    return out


NOOP = np.zeros(1, dtype=np.int32)


def test_autoreset_is_next_step_and_the_burned_step_is_inert():
    """After a done, envpool spends one extra step call resetting: reward 0, done False, fresh obs."""
    env = make_deterministic_pool()
    reset(env)
    rows = []
    for _ in range(3 * (MAX_EPISODE_STEPS + 1)):
        obs, rew, done, info = step(env, NOOP)
        rows.append((digest(obs[0, 3]), float(rew[0]), bool(done[0]), int(info["elapsed_step"][0])))

    # the cycle is MAX_EPISODE_STEPS real steps then one burned step
    elapsed = [r[3] for r in rows]
    assert elapsed == [1, 2, 3, 4, 0] * 3, elapsed
    dones = [r[2] for r in rows]
    assert dones == [False, False, False, True, False] * 3, dones

    # every burned row (index 4, 9, 14) carries reward exactly 0 and done False
    for i in (4, 9, 14):
        assert rows[i][1] == 0.0, f"burned row {i} reward {rows[i][1]!r}, expected 0.0"
        assert rows[i][2] is False


def test_exact_observation_sequence_repeats_with_the_burn_included():
    """The observation sequence is exactly periodic with period MAX_EPISODE_STEPS+1, not +0.

    This is the off-by-one made visible: the terminal observation s_last is returned ON the done
    step, and the new episode's first observation s_new only arrives on the NEXT call.
    """
    env = make_deterministic_pool()
    obs_after_reset = digest(reset(env)[0, 3])
    seq = []
    for _ in range(3 * (MAX_EPISODE_STEPS + 1)):
        obs, _, _, _ = step(env, NOOP)
        seq.append(digest(obs[0, 3]))

    cycle = seq[: MAX_EPISODE_STEPS + 1]
    assert seq == cycle * 3, f"sequence is not periodic with period {MAX_EPISODE_STEPS + 1}:\n{seq}"

    # the observation on the done step (index 3) is the terminal state s_last...
    s_last = seq[MAX_EPISODE_STEPS - 1]
    # ...and s_new, the new episode's first observation, arrives one call LATER (index 4)
    s_new = seq[MAX_EPISODE_STEPS]
    assert s_last != s_new, "s_last and s_new must differ for this test to mean anything"
    # a fresh deterministic pool starts from exactly that s_new-equivalent state
    assert s_new == obs_after_reset, (
        "the burned step must return the same observation a fresh reset() returns; "
        f"got {s_new} vs {obs_after_reset}"
    )


def test_the_action_on_the_burned_step_is_discarded():
    """Two identical pools that differ ONLY in the action fed on the burned step stay identical.

    Uses CartPole rather than Atari: every action moves the cart on the very next observation,
    so the control case detects a real action effect within a single 4-step episode.
    """
    n_steps = 3 * (MAX_EPISODE_STEPS + 1)
    burned_at = MAX_EPISODE_STEPS  # 0-based index of the first burned call
    left, right = np.zeros(1, dtype=np.int32), np.ones(1, dtype=np.int32)

    def run(inject_at):
        """Step two CartPole pools with identical actions except at inject_at; return divergence index."""
        a = envpool.make("CartPole-v1", env_type="gym", num_envs=1, seed=0, max_episode_steps=MAX_EPISODE_STEPS)
        b = envpool.make("CartPole-v1", env_type="gym", num_envs=1, seed=0, max_episode_steps=MAX_EPISODE_STEPS)
        reset(a), reset(b)
        for t in range(n_steps):
            oa, ra, _, _ = step(a, left)
            ob, rb, _, _ = step(b, right if t == inject_at else left)
            if digest(oa) != digest(ob) or ra[0] != rb[0]:
                return t
        return None

    assert run(burned_at) is None, "the action on the burned step changed the trajectory; it should be discarded"
    # control: the same injection on a real step MUST change the trajectory, else the test proves nothing
    assert run(burned_at - 1) is not None, "control failed: injecting on a real step changed nothing"


def test_dones_mask_selects_exactly_the_burned_rows():
    """In cleanrl's storage layout, dones[t] == 1 marks exactly the rows envpool fabricated."""
    n_steps = 3 * (MAX_EPISODE_STEPS + 1)
    env = make_deterministic_pool()
    next_obs, next_done = reset(env), np.zeros(1, dtype=bool)
    stored_obs, stored_done, stored_rew, returned_obs = [], [], [], []
    for _ in range(n_steps):
        # cleanrl: obs[t] = next_obs ; dones[t] = next_done ; then step and store the reward
        stored_obs.append(digest(next_obs[0, 3]))
        stored_done.append(bool(next_done[0]))
        next_obs, rew, done, _ = step(env, NOOP)
        stored_rew.append(float(rew[0]))
        returned_obs.append(digest(next_obs[0, 3]))
        next_done = done

    # call MAX_EPISODE_STEPS-1 (0-based) is the terminating step, so its done lands in row
    # MAX_EPISODE_STEPS; the burn then repeats every MAX_EPISODE_STEPS+1 rows
    burned = [t for t in range(n_steps) if stored_done[t]]
    assert burned == [MAX_EPISODE_STEPS + i * (MAX_EPISODE_STEPS + 1) for i in range(len(burned))], burned
    for t in burned:
        assert stored_rew[t] == 0.0, f"row {t} is burned but stores reward {stored_rew[t]}"
        # obs[t] is the terminal observation the previous call returned
        assert stored_obs[t] == returned_obs[t - 1]
        # and the observation this row leads to is the new episode's first observation
        assert returned_obs[t] != stored_obs[t]


# --------------------------------------------------------------------------- the fix itself
GAMMA, INT_GAMMA, LAM = 0.999, 0.99, 0.95


def gae(rewards, curiosity, ext_values, int_values, dones, next_done, nv_ext, nv_int, carry_through):
    """ppo_rnd_envpool's two-stream GAE; carry_through=True applies the burned-row fix."""
    T = len(rewards)
    ext_adv, int_adv = np.zeros_like(rewards), np.zeros_like(curiosity)
    ext_carry = int_carry = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            ext_nnt, ext_nv, int_nv = 1.0 - next_done, nv_ext, nv_int
        else:
            ext_nnt, ext_nv, int_nv = 1.0 - dones[t + 1], ext_values[t + 1], int_values[t + 1]
        ext_delta = rewards[t] + GAMMA * ext_nv * ext_nnt - ext_values[t]
        int_delta = curiosity[t] + INT_GAMMA * int_nv - int_values[t]  # intrinsic return is non-episodic
        ext_adv[t] = ext_delta + GAMMA * LAM * ext_nnt * ext_carry
        int_adv[t] = int_delta + INT_GAMMA * LAM * int_carry
        if carry_through:
            # a burned row must not become the carry for the row before it
            burn = dones[t]
            ext_carry = burn * ext_carry + (1.0 - burn) * ext_adv[t]
            int_carry = burn * int_carry + (1.0 - burn) * int_adv[t]
        else:
            ext_carry, int_carry = ext_adv[t], int_adv[t]
    return ext_adv, int_adv


def test_carry_through_stops_the_burned_row_contaminating_the_intrinsic_advantages():
    """Perturbing a burned row must not move any advantage that survives the batch mask."""
    T, burn_at = 32, 20
    rng = np.random.default_rng(0)
    rewards, curiosity = rng.normal(size=T), rng.uniform(0.5, 1.5, size=T)
    ext_values, int_values = rng.normal(size=T), rng.normal(size=T)
    dones = np.zeros(T)
    dones[burn_at] = 1.0
    rewards[burn_at] = 0.0  # envpool's burned row always carries reward 0
    args = (ext_values, int_values, dones, 0.0, 0.3, 0.4)

    def advantages(bump, carry_through):
        """Advantages with the burned row's intrinsic reward optionally perturbed by `bump`."""
        c = curiosity.copy()
        c[burn_at] += bump
        return gae(rewards, c, *args, carry_through=carry_through)

    # unfixed: the perturbation leaks backward through every earlier row of the rollout
    _, int_bad_a = advantages(0.0, carry_through=False)
    _, int_bad_b = advantages(1.0, carry_through=False)
    leaked = np.nonzero(np.abs(int_bad_a - int_bad_b) > 1e-12)[0]
    assert leaked.min() == 0 and leaked.max() == burn_at, leaked

    # fixed: only the burned row itself moves, and that row is dropped from the batch
    _, int_ok_a = advantages(0.0, carry_through=True)
    _, int_ok_b = advantages(1.0, carry_through=True)
    moved = np.nonzero(np.abs(int_ok_a - int_ok_b) > 1e-12)[0]
    assert moved.tolist() == [burn_at], moved


def test_batch_mask_drops_every_burned_row():
    """The b_inds construction keeps exactly the rows whose dones entry is 0."""
    num_steps, num_envs = 6, 3
    dones = np.zeros((num_steps, num_envs))
    dones[2, 0] = dones[4, 1] = dones[0, 2] = 1.0
    # before: b_inds = arange(18) -> 18 rows, 3 of them fabricated
    # after:  b_inds = the 15 rows with dones == 0, in the same flattened (step, env) order
    b_inds = np.nonzero(dones.reshape(-1) == 0)[0]
    assert len(b_inds) == num_steps * num_envs - 3
    assert set(b_inds) == set(range(18)) - {2 * 3 + 0, 4 * 3 + 1, 0 * 3 + 2}

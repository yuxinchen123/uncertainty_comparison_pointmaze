"""Check the reward-decline study's switches (2026-10-19): each one off changes nothing, each one on
does what its comment says.

Run standalone: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> this_file.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import jax
import jax.numpy as jnp
import numpy as np
from exploration_platform.agents.ppo.config import PPOConfig
from exploration_platform.agents.ppo.update import clip_and_report, clip_per_copy, per_copy_norm
from exploration_platform.envs.pointmaze.jax_pointmaze import EnvState, JaxPointMaze
from exploration_platform.envs.pointmaze.pm_common import EnvConfig, cell_center
from exploration_platform.training import diagnostics as diag
from exploration_platform.training.runner import Runner

SMALL = dict(n_copies=4, n_envs=2, num_steps=16, prime_iterations=2)


def state_arrays(state) -> dict:
    """Every array of a training state that the study's switches could touch, by name."""
    flat, _ = jax.tree_util.tree_flatten_with_path(
        {"params": (state.agent_params, state.bonus_params), "opt": state.opt,
         "env": state.env_state, "obs": state.obs, "bonus_state": state.bonus_state,
         "int": (state.agent_state["int_filter"], state.agent_state["int_rms"])})
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def run(cfg, iterations=3, env_cfg=None, bonus="rnd_next_state"):
    """Train a few iterations; return the final state and the last iteration's metrics."""
    runner = Runner(cfg, bonus=bonus, env_cfg=env_cfg)
    state = runner.prime(runner.init_state(0))
    metrics = None
    for k in range(1, iterations + 1):
        state, metrics = runner.iterate(state, runner.lr_argument(k, iterations))
    return state, metrics


def test_environment():
    """Map checks, the lump-sum goal payment, and termination with the -1 shift."""
    # a start inside a wall is refused; the medium pair used by the study is accepted
    try:
        JaxPointMaze(EnvConfig(map_name="medium"), 1, 1)
        raise AssertionError("medium map with the large map's start was accepted")
    except ValueError as error:
        print(f"PASS refused: {error}")
    medium = EnvConfig(map_name="medium", start_cell=(6, 1), goal_cell=(1, 6))
    JaxPointMaze(medium, 1, 1)

    # remaining_steps needs termination
    try:
        JaxPointMaze(EnvConfig(goal_reward="remaining_steps"), 1, 1)
        raise AssertionError("remaining_steps without termination was accepted")
    except ValueError:
        pass

    # a ball resting on the goal at step 99 of a 150 cap: the step ends the episode and pays
    # 150 - 100 + 1 = 51; the same ball under per_step with shift -1 is paid 0, and a ball away
    # from the goal is paid -1 and the episode continues
    for goal_reward, shift, at_goal, reward, terminated in (
            ("remaining_steps", 0.0, True, 51.0, True),
            ("per_step", -1.0, True, 0.0, True),
            ("per_step", -1.0, False, -1.0, False)):
        cfg = EnvConfig(map_name="umaze", start_cell=(3, 1), goal_cell=(1, 1),
                        max_episode_steps=150, continuing_task=False, reward_shift=shift,
                        goal_reward=goal_reward)
        env = JaxPointMaze(cfg, 1, 1)
        where = cell_center((1, 1) if at_goal else (3, 1), env.rows, env.cols)
        pos = jnp.asarray([[where]], jnp.float32)
        goal = jnp.asarray([[cell_center((1, 1), env.rows, env.cols)]], jnp.float32)
        s = EnvState(pos, jnp.zeros_like(pos), goal, jnp.full((1, 1), 99, jnp.int32),
                     jnp.zeros((1, 1), jnp.int32))
        _, _, r, term, _, _ = env.step(s, jnp.zeros((1, 1, 2), jnp.float32))
        assert float(r[0, 0]) == reward and bool(term[0, 0]) == terminated, (goal_reward, r, term)
    print("PASS environment: lump-sum payment 51, shift -1 pays 0 at the goal and -1 elsewhere")


def test_tracker_against_loop():
    """The episode tracker against a plain loop over random rollouts, carried across two."""
    rng = np.random.default_rng(0)
    T, C, N = 40, 3, 2
    carry = diag.tracker_init(C, N)
    ref = {"reached": np.zeros((C, N), bool), "first": np.zeros((C, N), int),
           "inside": np.zeros((C, N), int), "ret": np.zeros((C, N)), "len": np.zeros((C, N), int)}
    for _ in range(2):
        inside = rng.random((T, C, N)) < 0.2
        reward = inside * 1.0 + rng.normal(size=(T, C, N)) * 0.1
        done = rng.random((T, C, N)) < 0.1
        carry, sums = diag.track_episodes(carry, jnp.asarray(inside), jnp.asarray(reward, jnp.float32),
                                          jnp.asarray(done))
        expect = {k: np.zeros(C) for k in ("episodes", "reached", "first_hit", "inside", "ret",
                                           "length")}
        for t in range(T):
            for c in range(C):
                for n in range(N):
                    ref["len"][c, n] += 1
                    if inside[t, c, n] and not ref["reached"][c, n]:
                        ref["first"][c, n] = ref["len"][c, n]
                    ref["reached"][c, n] |= inside[t, c, n]
                    ref["inside"][c, n] += inside[t, c, n]
                    ref["ret"][c, n] += reward[t, c, n]
                    if done[t, c, n]:
                        expect["episodes"][c] += 1
                        expect["reached"][c] += ref["reached"][c, n]
                        expect["first_hit"][c] += ref["first"][c, n] if ref["reached"][c, n] else 0
                        expect["inside"][c] += ref["inside"][c, n]
                        expect["ret"][c] += ref["ret"][c, n]
                        expect["length"][c] += ref["len"][c, n]
                        for k in ("reached", "first", "inside", "ret", "len"):
                            ref[k][c, n] = 0
        for name, value in expect.items():
            np.testing.assert_allclose(np.asarray(sums[name]), value, rtol=1e-5, atol=1e-4,
                                       err_msg=name)
    print("PASS episode tracker matches a plain loop over two carried rollouts")


def test_small_helpers():
    """Running goal rate, intrinsic weight factor, task-reward scaling, and the clip report."""
    # goal rate: 0.2 + 0.05 x (1/2 - 0.2) = 0.215; a copy with no ended episode keeps its rate
    rate = diag.update_success_rate(jnp.asarray([0.2, 0.3]), jnp.asarray([2.0, 0.0]),
                                    jnp.asarray([1.0, 0.0]), 0.05)
    np.testing.assert_allclose(np.asarray(rate), [0.215, 0.3], rtol=1e-6)

    # linear decay at iteration 3 of 10: 1 - 2/10 = 0.8; with the gate at rate 0.25: 0.6
    cfg = PPOConfig(n_copies=2, int_coef_schedule="linear_decay", total_iterations=10,
                    success_gate_rate=0.05)
    factor = diag.intrinsic_weight_factor(cfg, jnp.asarray(3, jnp.int32), jnp.asarray([0.25, 0.]))
    np.testing.assert_allclose(np.asarray(factor), [0.6, 0.8], rtol=1e-6)
    assert diag.intrinsic_weight_factor(PPOConfig(n_copies=2), jnp.asarray(3), None) is None

    # task-reward scaling: the discounted sum restarts after an episode end, and the divisor is
    # never below 1
    rng = np.random.default_rng(1)
    T, C, N, gamma = 30, 2, 3, 0.9
    r = (rng.random((T, C, N)) < 0.3).astype(np.float32) * 10
    done = (rng.random((T, C, N)) < 0.1).astype(np.float32)
    f0, rms0 = diag.ext_filter_init(C, N)
    f, rms, scaled, std = diag.normalize_task_reward(f0, rms0, jnp.asarray(r), jnp.asarray(done),
                                                     gamma)
    sums, acc = np.zeros((T, C, N)), np.zeros((C, N))
    for t in range(T):
        acc = gamma * acc + r[t]
        sums[t] = acc
        acc = acc * (1 - done[t])
    np.testing.assert_allclose(np.asarray(f), acc, rtol=1e-5)
    flat = sums.transpose(1, 0, 2).reshape(C, -1)
    # the fresh statistics hold mean 0, variance 1 at weight 1e-4, merged with this batch
    n0, n1 = 1e-4, flat.shape[1]
    mean = flat.mean(1)
    var = (1.0 * n0 + flat.var(1) * n1 + mean ** 2 * n0 * n1 / (n0 + n1)) / (n0 + n1)
    expect_std = np.maximum(np.sqrt(var + 1e-8), 1.0)
    np.testing.assert_allclose(np.asarray(std), expect_std, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(scaled), r / expect_std[None, :, None], rtol=1e-5)

    # the clip report: joint by default (identical to clip_per_copy), per network when asked
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    grads = {"agent": {"actor": {"W": jax.random.normal(keys[0], (2, 5)) * 3},
                       "critic": {"W": jax.random.normal(keys[1], (2, 7)) * 0.01}},
             "bonus": {"predictor": {"W": jax.random.normal(keys[2], (2, 4)) * 2}}}
    joint, report = clip_and_report(PPOConfig(n_copies=2), grads, 2)
    for a, b in zip(jax.tree.leaves(joint), jax.tree.leaves(clip_per_copy(grads, 2, 0.5))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    separate, _ = clip_and_report(PPOConfig(n_copies=2, clip_per_network=True), grads, 2)
    for part in (separate["agent"]["actor"], separate["agent"]["critic"], separate["bonus"]):
        assert (np.asarray(per_copy_norm(part, 2)) <= 0.5 + 1e-5).all()
    # the small critic gradient is untouched by its own clip, but shrunk by the joint one
    np.testing.assert_array_equal(np.asarray(separate["agent"]["critic"]["W"]),
                                  np.asarray(grads["agent"]["critic"]["W"]))
    assert (np.abs(np.asarray(joint["agent"]["critic"]["W"]))
            < np.abs(np.asarray(grads["agent"]["critic"]["W"]))).all()
    print("PASS goal rate, weight factor, task-reward scaling, clip report")


def test_diagnostics_do_not_change_training():
    """Diagnostics on: the same training state as off, in both update styles.

    Bit for bit on the processor, and for full-batch updates on the card. On the card the
    epoch-minibatch program with diagnostics is compiled differently enough to move the last bits
    (seen on an RTX A4500, 2026-10-19), so there it must agree to float32 rounding carried over
    three iterations, and the size of the difference is printed.
    """
    on_card = jax.default_backend() == "gpu"
    for style in ("full_batch", "epoch_minibatch"):
        off, _ = run(PPOConfig(update_style=style, **SMALL))
        on, metrics = run(PPOConfig(update_style=style, diagnostics=True, **SMALL))
        a, b = state_arrays(off), state_arrays(on)
        assert a.keys() == b.keys()
        exact = not (on_card and style == "epoch_minibatch")
        worst = 0.0
        for name in a:
            if exact:
                assert a[name].tobytes() == b[name].tobytes(), f"{style}: {name} differs"
            elif a[name].dtype.kind == "f":
                worst = max(worst, float(np.max(np.abs(a[name] - b[name]), initial=0.0)))
                np.testing.assert_allclose(b[name], a[name], rtol=1e-4, atol=1e-5,
                                           err_msg=f"{style}: {name}")
        d = metrics["diagnostics"]
        assert all(np.isfinite(np.asarray(v)).all() for v in d.values()), style
        assert set(d) >= {"logstd_mean", "approx_kl", "episode_episodes", "grad_norm_actor"}
        print(f"PASS {style}: diagnostics leave training "
              f"{'bit-identical' if exact else f'within {worst:.2e} after 3 iterations'}; "
              f"{len(d)} diagnostic fields, all finite")


def test_switches_on():
    """Each training switch runs, stays finite, and does what it says in its first iterations."""
    # linear decay: at iteration 3 of a 10-iteration run the weight is beta x 0.8
    cfg = PPOConfig(diagnostics=True, int_coef=100.0, int_coef_schedule="linear_decay",
                    total_iterations=10, **SMALL)
    _, metrics = run(cfg, iterations=3)
    np.testing.assert_allclose(np.asarray(metrics["diagnostics"]["beta_effective"]), 80.0,
                               rtol=1e-6)

    # the goal gate starts at weight beta (no copy has a goal-reaching record yet)
    cfg = PPOConfig(diagnostics=True, int_coef=10.0, success_gate_rate=0.05, **SMALL)
    _, metrics = run(cfg, iterations=1)
    np.testing.assert_allclose(np.asarray(metrics["diagnostics"]["beta_effective"]), 10.0)

    # the remaining switches and a terminating environment on the medium map
    medium = EnvConfig(map_name="medium", start_cell=(6, 1), goal_cell=(1, 6),
                       max_episode_steps=250, continuing_task=False, reward_shift=-1.0)
    for name, cfg, env_cfg in (
            ("normalize_ext_reward", PPOConfig(diagnostics=True, normalize_ext_reward=True,
                                               **SMALL), None),
            ("clip_per_network", PPOConfig(diagnostics=True, clip_per_network=True, **SMALL),
             None),
            ("terminating medium map", PPOConfig(diagnostics=True, **SMALL), medium)):
        state, metrics = run(cfg, env_cfg=env_cfg)
        values = [np.asarray(v) for v in jax.tree.leaves(state_arrays(state))]
        assert all(np.isfinite(v).all() for v in values if v.dtype.kind == "f"), name
        print(f"PASS {name}: three iterations, every float finite")

    # with no switch at all the agent state holds exactly the two original entries
    runner = Runner(PPOConfig(**SMALL))
    assert set(runner.init_state(0).agent_state) == {"int_filter", "int_rms"}
    print("PASS defaults: the agent state is unchanged")


def test_command_line():
    """The command's defaults name the original environment; the study's flags reach the config."""
    import run_training
    sys.argv = ["run_training.py", "--run-dir", "/tmp/x", "--unit-id", "u"]
    args = run_training.parse_args()
    env_cfg = run_training.build_env_config(args)
    assert env_cfg == EnvConfig(), env_cfg
    assert run_training.env_spec(env_cfg) == "pointmaze_large_cont400_nonoise@1"
    config = run_training.build_config(args)
    assert not config.diagnostics and config.int_coef_schedule == "constant"

    sys.argv += ["--map", "medium", "--start-cell", "6,1", "--goal-cell", "1,6",
                 "--episode-steps", "250", "--terminate-at-goal", "--reward-shift", "-1",
                 "--diagnostics", "--intrinsic-weight-schedule", "linear_decay",
                 "--iterations", "77", "--ent-coef", "0.01", "--gamma-ext", "0.99"]
    args = run_training.parse_args()
    env_cfg = run_training.build_env_config(args)
    assert run_training.env_spec(env_cfg) == (
        "pointmaze_medium_term250_nonoise_start6-1_goal1-6_shift-1@1")
    config = run_training.build_config(args)
    assert (config.total_iterations, config.ent_coef, config.gamma_ext) == (77, 0.01, 0.99)

    # an anneal horizon longer than the run is accepted; a shorter one is refused
    sys.argv += ["--lr-anneal-iterations", "1000"]
    run_training.build_config(run_training.parse_args())
    sys.argv[-1] = "10"
    try:
        run_training.build_config(run_training.parse_args())
        raise AssertionError("an anneal horizon shorter than the run was accepted")
    except ValueError:
        pass
    print("PASS command line: original defaults, study flags reach the configuration")


if __name__ == "__main__":
    test_environment()
    test_tracker_against_loop()
    test_small_helpers()
    test_diagnostics_do_not_change_training()
    test_switches_on()
    test_command_line()
    print("ALL PASS")

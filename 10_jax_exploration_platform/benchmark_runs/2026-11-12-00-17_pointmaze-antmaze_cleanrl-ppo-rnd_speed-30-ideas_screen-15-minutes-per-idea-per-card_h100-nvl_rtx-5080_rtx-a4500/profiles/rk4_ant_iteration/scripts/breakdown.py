"""Where one training iteration of the RK4 AntMaze unit spends its time, at one number of policies.

Three parts, each optional (--parts):

1. `iteration`: the unit's trainer (16 environments per policy, RK4 at 0.01 s x 5, the CleanRL
   update) built as the training command builds it. The whole iteration in chained calls (as the
   training loop calls it) and one call at a time with a wait; then each piece of the iteration as
   its own compiled function (`benchmarks/harness/profile_study_iteration_pieces.py`,
   `build_pieces`: the update and its parts, the rollout scan, the environment alone, the policy
   forward alone, the critic pass, the distillation forward, the reward filter and advantages, ...),
   and the physics alone: the same 128 x 5 batched `mjx.step` calls with nothing else (no fault
   test, reward, reset or observation). The number of CUDA graphs Warp captured during the timed
   rounds is recorded (a capture there would be a hidden cost).
2. `native`: the physics step split into MuJoCo Warp's own stages, run natively (no JAX) on the
   same C model at 16 x copies worlds, each stage captured as its own CUDA graph and replayed:
   position stage, collision detection, constraint build, transmission, velocity, actuation,
   acceleration with the mass-matrix factor, constraint solve; forward (all of them); the
   Runge-Kutta part (stages 2 to 4 and the combination); the whole physics step. Also the kernel
   launches inside each graph, and the solver iterations used per world.
3. `native_variants`: the native physics step with settings that change no number when nothing is
   dropped: the solver loop as a plain loop of 4 iterations instead of the conditional graph loop,
   and smaller pools (constraint rows per environment, contact slots per environment); and, for the
   record only, two settings that change numbers (2 solver iterations, 4 line-search iterations).

The record is rewritten after every piece. Run on a compute node with one card:
  python breakdown.py --copies 128 --parts iteration,native,native_variants --out records/x.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CaptureCounter, build_trainer, chained_seconds, lr_of, machine, median,  # noqa: E402
                    say, throughput, write_record)


def parse_arguments():
    """The shape, the parts to run and where to write."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copies", type=int, required=True)
    parser.add_argument("--parts", default="iteration,native,native_variants")
    parser.add_argument("--native-worlds", default="", help="comma-separated world counts for the native parts; "
                        "default 16 x copies")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=840.0)
    return parser.parse_args()


def adaptive_call_seconds(fn, counter, target_seconds: float = 2.0):
    """Seconds of each call of fn() (a wait after every call), as many calls as fit in about
    `target_seconds`, between 3 and 30; two untimed calls first (compile, then warm). Returns the
    times and the graphs Warp captured during the timed calls."""
    import jax
    for _ in range(2):
        jax.block_until_ready(fn())
    start = time.perf_counter()
    jax.block_until_ready(fn())
    first = time.perf_counter() - start
    before = counter.count
    times = []
    for _ in range(min(30, max(3, int(target_seconds / max(first, 1e-6))))):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    return times, counter.count - before


def physics_only_piece(runner, state):
    """128 x frame_skip batched mjx.step calls at fixed torques, nothing else; a no-argument function."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    from exploration_platform.training.train_step import rollout_unroll

    env, cfg = runner.env, runner.cfg

    # GOAL: the environment's own data with constant torques, as the rollout hands them in
    with jax.enable_x64(False):
        torques = jax.random.uniform(jax.random.PRNGKey(3), (env.B, env.nu), jnp.float32, -1.0, 1.0)
        data = state.env_state.data.replace(ctrl=torques)

    def physics(data):
        """The rollout's physics calls: 128 steps of frame_skip physics steps, unrolled as the rollout."""
        with jax.enable_x64(False):
            def frame(d, _):
                """One batched physics step."""
                return jax.vmap(mjx.step, in_axes=(None, 0))(env.model, d), None

            def environment_step(d, _):
                """frame_skip physics steps, as one environment step runs them."""
                return jax.lax.scan(frame, d, None, length=env.cfg.frame_skip)[0], None
            return jax.lax.scan(environment_step, data, None, length=cfg.num_steps,
                                unroll=rollout_unroll(cfg))[0].qpos

    program = jax.jit(physics)
    return lambda: program(data)


def chained_environment_pieces(runner, state):
    """The environment's 128 steps and the bare physics at the same torques, each as the iteration
    calls them: compiled with the state donated, chained so every call gets the buffers the previous
    call returned. Returns name -> (step function of the state, starting state)."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    from exploration_platform.training.train_step import rollout_unroll

    env, cfg = runner.env, runner.cfg
    unroll = rollout_unroll(cfg)
    torques = jax.random.uniform(jax.random.PRNGKey(5), (cfg.num_steps, cfg.n_copies, cfg.n_envs, env.act_dim),
                                 jnp.float32, -1.0, 1.0)

    def environment(env_state):
        """128 environment steps at the given torques; returns the state (the outputs are computed too)."""
        def body(env_state, action):
            """One environment step, keeping what the rollout keeps."""
            stepped, next_obs, r_ext, term, trunc, final_obs = env.step(env_state, action)
            return stepped, (next_obs, final_obs, r_ext, term, trunc)
        env_state, outputs = jax.lax.scan(body, env_state, torques, unroll=unroll)
        return env_state, outputs

    def physics(data):
        """The same 128 x frame_skip physics steps at the same torques, with nothing else."""
        with jax.enable_x64(False):
            def frame(d, _):
                """One batched physics step."""
                return jax.vmap(mjx.step, in_axes=(None, 0))(env.model, d), None

            def environment_step(d, action):
                """frame_skip physics steps at one torque, as one environment step runs them."""
                d = d.replace(ctrl=jnp.clip(action, -1.0, 1.0).reshape(env.B, env.nu))
                return jax.lax.scan(frame, d, None, length=env.cfg.frame_skip)[0], None
            return jax.lax.scan(environment_step, data, torques, unroll=unroll)[0]

    environment_j = jax.jit(environment, donate_argnums=(0,))
    physics_j = jax.jit(physics, donate_argnums=(0,))
    copy = lambda tree: jax.tree.map(jnp.copy, tree)
    return {
        "environment alone, 128 steps, chained with the state donated":
            (lambda s: environment_j(s)[0], copy(state.env_state)),
        "physics alone at the same torques (128 x 5 physics steps, nothing else), chained with the data donated":
            (physics_j, copy(state.env_state.data)),
    }


def iteration_part(arguments, record, started):
    """Part 1: the whole iteration and its pieces through JAX."""
    import jax
    from profile_study_iteration_pieces import build_pieces

    counter = CaptureCounter()
    build_start = time.perf_counter()
    runner, state, environment = build_trainer(arguments.copies)
    jax.block_until_ready(state)
    record["iteration"] = {"build_and_prime_seconds": time.perf_counter() - build_start,
                           "rollout_unroll": None, "pieces": {}, "not_measured": []}
    part = record["iteration"]
    from exploration_platform.training.train_step import rollout_unroll
    part["rollout_unroll"] = rollout_unroll(runner.cfg)
    part["update_unroll"] = runner.cfg.update_unroll
    part["graph_conditional"] = bool(runner.env.model.opt._impl.graph_conditional)
    say(f"built and primed {arguments.copies} policies x 16 environments in {part['build_and_prime_seconds']:.1f} s")
    piece_state = jax.tree.map(jax.numpy.copy, state)

    # GOAL: two warm-up calls (compile, then compile again for the donated state)
    lr = lr_of(runner, 1)
    step = lambda s: runner.iterate(s, lr)[0]
    warm = []
    for _ in range(2):
        start = time.perf_counter()
        state = step(state)
        jax.block_until_ready(state)
        warm.append(time.perf_counter() - start)
    part["warmup_call_seconds"] = warm
    say(f"warm-up calls {warm[0]:.1f} s and {warm[1]:.2f} s")

    # GOAL: chained rounds as the training loop calls it, with the graph captures counted
    captures_before = counter.count
    rounds = []
    per_round = max(2, round(3.0 / warm[1]))
    for _ in range(7):
        seconds, state = chained_seconds(step, state, per_round)
        rounds.append(seconds)
    part["chained_rounds_seconds"] = rounds
    part["chained_calls_per_round"] = per_round
    part["graph_captures_during_timed_rounds"] = counter.count - captures_before
    waited = []
    for _ in range(4):
        seconds, state = chained_seconds(step, state, 1)
        waited.append(seconds)
    part["one_call_with_wait_seconds"] = waited
    part["throughput"] = throughput(arguments.copies, 16, runner.cfg.num_steps, median(rounds))
    say(f"whole iteration {median(rounds):.3f} s chained, {median(waited):.3f} s with a wait; "
        f"{part['graph_captures_during_timed_rounds']} graph captures while timed")
    write_record(arguments.out, record)

    # GOAL: every piece alone, the physics alone last
    pieces = build_pieces(runner, piece_state, jax.random.PRNGKey(7))
    pieces["physics alone: 128 x 5 batched physics steps, nothing else"] = physics_only_piece(runner, piece_state)
    for name, piece in pieces.items():
        if time.time() - started > arguments.budget_seconds:
            part["not_measured"].append(name)
            continue
        start = time.perf_counter()
        times, captures = adaptive_call_seconds(piece, counter)
        part["pieces"][name] = {"median_seconds": median(times), "least_seconds": min(times),
                                "calls": len(times), "wall_seconds": time.perf_counter() - start,
                                "graph_captures_during_timed_calls": captures}
        say(f"{median(times) * 1e3:10.2f} ms  {name}  ({time.perf_counter() - start:.0f} s, {captures} captures)")
        write_record(arguments.out, record)

    # GOAL: the environment and the bare physics as the iteration calls them (donated, chained)
    for name, (step, own) in chained_environment_pieces(runner, piece_state).items():
        if time.time() - started > arguments.budget_seconds:
            part["not_measured"].append(name)
            continue
        start = time.perf_counter()
        for _ in range(2):
            own = step(own)
            jax.block_until_ready(own)
        before, rounds = counter.count, []
        for _ in range(5):
            seconds, own = chained_seconds(step, own, 2)
            rounds.append(seconds)
        part["pieces"][name] = {"median_seconds": median(rounds), "least_seconds": min(rounds),
                                "rounds_of_two_chained_calls": rounds, "wall_seconds": time.perf_counter() - start,
                                "graph_captures_during_timed_calls": counter.count - before}
        say(f"{median(rounds) * 1e3:10.2f} ms  {name}  ({time.perf_counter() - start:.0f} s, "
            f"{counter.count - before} captures)")
        write_record(arguments.out, record)
    part["warp_contact_pool"] = runner.env.overflow_report(state.env_state)
    del runner, state, piece_state, pieces


def native_model_and_data(worlds: int, contacts_per_world: int = 48, rows_per_world: int = 512):
    """MuJoCo Warp's model and data of the unit's C model at `worlds` worlds, after 60 physics steps
    of random torques from the spawn (so contacts are as in training); returns (m, d, step graph)."""
    import numpy as np
    import warp as wp
    import mujoco
    from mujoco.mjx.third_party import mujoco_warp as mjw
    from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, cell_center
    from exploration_platform.envs.pointmaze.pm_common import MAPS
    from common import unit_configuration

    # GOAL: the C model the unit's environment builds from the configuration file
    environment = unit_configuration(8)[2]
    mj_model = mujoco.MjModel.from_xml_string(build_antmaze_xml(environment))
    if mj_model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("the model does not integrate with RK4")

    # GOAL: batched data at the unit's spawn (the model's qpos0 with the torso moved to the start
    # cell's centre, as WarpAntMaze spawns it), then 60 steps with torques redrawn every 5 steps
    m = mjw.put_model(mj_model)
    d = mjw.make_data(mj_model, nworld=worlds, naconmax=worlds * contacts_per_world, njmax=rows_per_world)
    maze = MAPS[environment.map_name]
    spawn = np.asarray(mj_model.qpos0, np.float32)
    spawn[0:2] = cell_center(environment.start_cell, len(maze), len(maze[0]))
    d.qpos.assign(np.tile(spawn, (worlds, 1)))
    with wp.ScopedCapture() as capture:
        mjw.step(m, d)
    generator = np.random.default_rng(11)
    for block in range(12):
        d.ctrl.assign(generator.uniform(-1, 1, (worlds, mj_model.nu)).astype(np.float32))
        for _ in range(5):
            wp.capture_launch(capture.graph)
    wp.synchronize()
    return mj_model, m, d, capture.graph


def graph_seconds(graph, calls: int = 20, rounds: int = 5) -> list:
    """Seconds per replay of a captured graph, one value per round of `calls` replays."""
    import warp as wp
    wp.capture_launch(graph)
    wp.synchronize()
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(calls):
            wp.capture_launch(graph)
        wp.synchronize()
        times.append((time.perf_counter() - start) / calls)
    return times


class LaunchCounter:
    """Counts kernel launches made while it is active (inside a graph capture: the graph's kernels)."""

    def __enter__(self):
        """Wrap warp.launch and warp.launch_tiled."""
        import warp as wp
        self.count, self.saved = 0, (wp.launch, wp.launch_tiled)

        def counted(original):
            """A launch function that counts, then launches."""
            def launch(*args, **kwargs):
                """Count one launch and make it."""
                self.count += 1
                return original(*args, **kwargs)
            return launch
        wp.launch, wp.launch_tiled = counted(wp.launch), counted(wp.launch_tiled)
        return self

    def __exit__(self, *exc):
        """Put the original launch functions back."""
        import warp as wp
        wp.launch, wp.launch_tiled = self.saved


def native_part(arguments, record, worlds_list):
    """Part 2: the physics step split into MuJoCo Warp's stages, natively."""
    import numpy as np
    import warp as wp
    from mujoco.mjx.third_party.mujoco_warp._src import collision_driver, constraint, smooth, solver
    from mujoco.mjx.third_party.mujoco_warp._src import forward as fwd

    def position_stage(m, d):
        """Kinematics, centre of mass, composite inertia (forward's position stage before collision)."""
        smooth.kinematics(m, d)
        smooth.com_pos(m, d)
        smooth.camlight(m, d)
        smooth.flex(m, d)
        smooth.tendon(m, d)
        smooth.crb(m, d)
        smooth.tendon_armature(m, d)

    stages = {
        "position stage: kinematics, centre of mass, composite inertia": position_stage,
        "collision detection": collision_driver.collision,
        "constraint build": constraint.make_constraint,
        "transmission": smooth.transmission,
        "velocity stage": fwd.fwd_velocity,
        "actuation": fwd.fwd_actuation,
        "acceleration with the mass-matrix factor": lambda m, d: fwd.fwd_acceleration(m, d, factorize=True),
        "constraint solve": solver.solve,
        "forward (every stage above, once)": fwd.forward,
        "Runge-Kutta part: stages 2 to 4 (three more forwards) and the combination": fwd.rungekutta4,
        "whole physics step (forward and the Runge-Kutta part)": fwd.step,
    }
    record["native"] = record.get("native", {})
    for worlds in worlds_list:
        mj_model, m, d, _ = native_model_and_data(worlds)
        entry = {"worlds": worlds, "stages": {}}
        for name, stage in stages.items():
            with LaunchCounter() as launches, wp.ScopedCapture() as capture:
                stage(m, d)
            times = graph_seconds(capture.graph)
            entry["stages"][name] = {"median_seconds": median(times), "rounds_seconds": times,
                                     "kernel_launches": launches.count}
            say(f"native {worlds} worlds: {median(times) * 1e3:8.3f} ms, {launches.count:4d} launches  {name}")
        # GOAL: the solver iterations and pool use of these states
        niter = d.solver_niter.numpy()
        entry["solver_iterations_mean"] = float(niter.mean())
        entry["solver_iterations_largest"] = int(niter.max())
        entry["solver_iterations_histogram"] = np.bincount(niter, minlength=5).tolist()
        entry["constraint_rows_largest"] = int(d.nefc.numpy().max())
        entry["contacts_in_pool"] = int(d.nacon.numpy()[0])
        entry["torso_height_mean"] = float(d.qpos.numpy()[:, 2].mean())
        record["native"][str(worlds)] = entry
        write_record(arguments.out, record)
        del m, d


def native_variants_part(arguments, record, worlds_list):
    """Part 3: the native physics step with settings that change no number when nothing is dropped,
    timed in alternating rounds against the unit's setting."""
    import copy
    import warp as wp
    from mujoco.mjx.third_party import mujoco_warp as mjw

    record["native_variants"] = record.get("native_variants", {})
    for worlds in worlds_list:
        graphs = {}
        # GOAL: the unit's setting and the plain-loop solver on the same data
        mj_model, m, d, step_graph = native_model_and_data(worlds)
        graphs["unit's setting: 48 contact slots and 512 rows per environment, conditional solver loop"] = step_graph
        m_plain = copy.copy(m)
        m_plain.opt = copy.copy(m.opt)
        m_plain.opt.graph_conditional = False
        with wp.ScopedCapture() as capture:
            mjw.step(m_plain, d)
        graphs["solver loop as a plain loop of 4 iterations"] = capture.graph
        # GOAL: settings that change numbers (measured and reported only), on the same data
        for label, field, value in (("changes numbers: 2 solver iterations", "iterations", 2),
                                    ("changes numbers: 4 line-search iterations", "ls_iterations", 4)):
            m_other = copy.copy(m)
            m_other.opt = copy.copy(m.opt)
            setattr(m_other.opt, field, value)
            with wp.ScopedCapture() as capture:
                mjw.step(m_other, d)
            graphs[label] = capture.graph
        # GOAL: smaller pools, each on its own data advanced the same way
        keep = []
        for slots, rows in ((48, 128), (16, 512), (16, 128)):
            _, m2, d2, graph = native_model_and_data(worlds, contacts_per_world=slots, rows_per_world=rows)
            keep.append((m2, d2))
            graphs[f"{slots} contact slots and {rows} rows per environment"] = graph
        # GOAL: alternating rounds, order reversed every second round
        rounds = {name: [] for name in graphs}
        names = list(graphs)
        for r in range(8):
            for name in (names if r % 2 == 0 else names[::-1]):
                rounds[name].extend(graph_seconds(graphs[name], calls=10, rounds=1))
        reference = median(rounds[names[0]])
        entry = {name: {"median_seconds": median(times), "rounds_seconds": times,
                        "time_against_unit_setting": median(times) / reference}
                 for name, times in rounds.items()}
        entry["dropped_anything"] = {f"{slots} contact slots and {rows} rows per environment":
                                     int(dd.overflow.numpy().max()) if hasattr(dd, "overflow") else None
                                     for (slots, rows), (_, dd) in zip(((48, 128), (16, 512), (16, 128)), keep)}
        for name, values in entry.items():
            if "median_seconds" in values:
                say(f"variant {worlds} worlds: {values['median_seconds'] * 1e3:8.3f} ms "
                    f"({values['time_against_unit_setting']:.3f})  {name}")
        record["native_variants"][str(worlds)] = entry
        write_record(arguments.out, record)


def main():
    """Run the requested parts at one number of policies and write the record."""
    arguments = parse_arguments()
    started = time.time()
    parts = arguments.parts.split(",")
    worlds = ([int(w) for w in arguments.native_worlds.split(",")] if arguments.native_worlds
              else [16 * arguments.copies])
    record = {"record": "RK4 AntMaze iteration breakdown", "copies": arguments.copies,
              "environments_per_policy": 16, "machine": machine(), "parts": parts}
    write_record(arguments.out, record)
    if "iteration" in parts:
        iteration_part(arguments, record, started)
    if "native" in parts:
        native_part(arguments, record, worlds)
    if "native_variants" in parts:
        native_variants_part(arguments, record, worlds)
    record["wall_seconds"] = time.time() - started
    write_record(arguments.out, record)
    say(f"wrote {arguments.out} after {record['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()

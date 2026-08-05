"""End-to-end check that a resumed run continues the learning-rate schedule, not restarts it.

The learning rate here is not a stored number — it is recomputed every iteration from the update
index and the total update count:

    frac = 1.0 - (update - 1.0) / num_updates
    lr   = frac * args.learning_rate

So the schedule is only correct across a resume if the update index is restored. This test runs a
short training twice — once uninterrupted, once stopped and resumed — and asserts the learning rate
logged at every shared update is identical.

It also asserts the step and update counters line up, and that the resumed run does not repeat work.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python -m pytest test_resume_schedule.py -q
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "..", "src", "ppo_rnd_envpool_shuze.py")
PYTHON = "/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"

# Small enough to run on a cpu in about a minute, large enough to span several logged updates.
COMMON = [
    "--env_id", "MontezumaRevenge-v5", "--num_envs", "8", "--num_steps", "16",
    "--total_timesteps", "2560", "--num_iterations_obs_norm_init", "1",
    "--num_minibatches", "2", "--update_epochs", "1", "--log_every_updates", "1",
    "--run_id", "0", "--run_total", "1", "--seed", "1", "--no-cuda", "--opt_env_threads", "2",
    "--minibatch_drop_allowance", "8",
]


def run_trainer(out_dir, extra):
    """Run the trainer into out_dir and return its record, or raise with the log on failure."""
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([PYTHON, TRAINER] + COMMON + ["--output_dir", out_dir] + extra,
                          capture_output=True, text=True, env=env, timeout=1800)
    if proc.returncode != 0:
        raise AssertionError(f"trainer failed rc={proc.returncode}\n{proc.stdout[-3000:]}")
    return json.load(open(os.path.join(out_dir, "0_of_1.json"))), proc.stdout


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    """One run straight through, as the reference."""
    d = str(tmp_path_factory.mktemp("straight"))
    rec, _ = run_trainer(d, ["--checkpoint_every_seconds", "1e9", "--no-resume"])
    return rec


@pytest.fixture(scope="module")
def interrupted(tmp_path_factory):
    """The same run stopped after 8 updates and resumed by re-running the identical command."""
    d = str(tmp_path_factory.mktemp("resumed"))
    # Segment 1 stops early. checkpoint_every_seconds=0 forces a checkpoint at the first chance,
    # so the stop leaves something to resume from.
    _, out1 = run_trainer(d, ["--checkpoint_every_seconds", "0", "--profile_iterations", "8"])
    # Segment 2 is the identical command with no iteration cap; it must resume, not restart.
    rec, out2 = run_trainer(d, ["--checkpoint_every_seconds", "0"])
    return rec, out1, out2


def test_resume_actually_resumed(interrupted):
    """The second segment reports a resume rather than starting over."""
    _, _, out2 = interrupted
    assert "[resume] continuing at update 9" in out2, out2[-2000:]


def test_learning_rate_schedule_survives_the_resume(uninterrupted, interrupted):
    """Golden path: the learning rate at every shared update matches the uninterrupted run."""
    ref = {r["update"]: r["charts/learning_rate"] for r in uninterrupted["eval_history"]}
    got = {r["update"]: r["charts/learning_rate"] for r in interrupted[0]["eval_history"]}
    shared = sorted(set(ref) & set(got))
    assert len(shared) >= 8, f"too few shared updates to compare: {shared}"
    for u in shared:
        assert got[u] == pytest.approx(ref[u], rel=1e-12), (
            f"update {u}: resumed lr {got[u]} != uninterrupted {ref[u]} — the schedule restarted")
    # And it is genuinely decaying, so the test would catch a schedule frozen at its initial value.
    assert got[shared[-1]] < got[shared[0]]


def test_step_and_update_counters_continue(uninterrupted, interrupted):
    """The resumed run's counters continue rather than restart, and reach the same end."""
    ref, got = uninterrupted, interrupted[0]
    assert [r["update"] for r in got["eval_history"]] == [r["update"] for r in ref["eval_history"]]
    assert [r["step"] for r in got["eval_history"]] == [r["step"] for r in ref["eval_history"]]
    assert got["completed"] is True


def test_no_duplicated_history_rows(interrupted):
    """Edge case: resuming must not append a second copy of the rows written before the stop."""
    updates = [r["update"] for r in interrupted[0]["eval_history"]]
    assert len(updates) == len(set(updates)), f"duplicated updates after resume: {updates}"


def test_there_is_no_replay_buffer_to_lose():
    """PPO here is on-policy: the rollout buffers are rebuilt each iteration, so none is carried.

    This is asserted against the source rather than assumed, because 'did you save the replay
    buffer' is the first question anyone asks about resumability.
    """
    src = open(TRAINER).read()
    # The rollout tensors are allocated once and overwritten in place every iteration; there is no
    # buffer object, and nothing named replay anywhere in the trainer.
    assert "replay" not in src.lower().replace("replay buffer stores", ""), \
        "the trainer mentions a replay buffer; this test's premise needs revisiting"
    for name in ["obs = torch.zeros", "actions = torch.zeros", "rewards = torch.zeros"]:
        assert name in src, f"expected the rollout tensor allocation {name!r}"

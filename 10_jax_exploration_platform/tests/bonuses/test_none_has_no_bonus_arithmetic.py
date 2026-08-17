"""Selecting no bonus really removes it: the compiled program contains none of its arithmetic.

Selecting the bonus before `jax.jit` is only worth doing if it is real. The way to see that it is
real is to look at the program the compiler produced. Random network distillation is built here
with two deliberately odd network widths — 193 hidden and 97 features — that appear nowhere else
in this program: the observation is 4 wide, the agent's layers are 64 and 2 and 1, and the batch
shapes are powers of two. Every array shape carrying a 193 or a 97 is therefore this bonus's, and
counting them in the optimized program answers the question directly.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_none_has_no_bonus_arithmetic.py

The same file also runs on the graphics card (drop JAX_PLATFORMS), through the serval05 lock.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.rnd.config import RNDConfig  # noqa: E402
from exploration_platform.bonuses.registry import rnd_preset  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

# widths that cannot arise from anything else in this program
MARKED = rnd_preset(RNDConfig(hidden=193, feature_dim=97))
MARKS = ("193", "97")
CONFIG = dict(n_copies=4, n_envs=4, num_steps=32, prime_iterations=1, update_style="full_batch")


def compiled_text(bonus) -> str:
    """The optimized program for one iteration of PPO with this bonus."""
    runner = Runner(PPOConfig(**CONFIG), bonus=bonus)
    return runner.iterate.lower(runner.init_state(), 3e-4).compile().as_text()


def shapes_with(text: str, marks) -> dict:
    """How many array shapes in the program carry each marked dimension.

    before: a line 'f32[4,512,97]{2,1,0} dot(...)'
    after:  the shape 'f32[4,512,97]' is counted under the mark '97'
    """
    found = re.findall(r"\bf(?:32|64)\[[0-9,]*\]", text)
    return {mark: sum(1 for shape in found
                      if mark in re.findall(r"[0-9]+", shape)) for mark in marks}


def test_the_marked_widths_are_there_when_the_bonus_is():
    """The control: with the bonus selected, its network widths are all over the program."""
    counts = shapes_with(compiled_text(MARKED), MARKS)
    print(f"PPO + random network distillation: shapes carrying {MARKS} = {counts}")
    assert all(n > 0 for n in counts.values()), \
        "the marked widths are absent even WITH the bonus, so this test proves nothing"
    print("ok test_the_marked_widths_are_there_when_the_bonus_is")


def test_no_bonus_arithmetic_without_the_bonus():
    """With no bonus selected, not one array shape of the bonus's networks survives."""
    text = compiled_text("none")
    counts = shapes_with(text, MARKS)
    print(f"PPO + none: shapes carrying {MARKS} = {counts}")
    assert all(n == 0 for n in counts.values()), \
        f"the program without a bonus still holds shapes of the bonus's networks: {counts}"

    # and it is smaller in the way that matters: fewer matrix multiplications
    with_bonus = len(re.findall(r"\bdot\(", compiled_text(MARKED)))
    without = len(re.findall(r"\bdot\(", text))
    print(f"matrix multiplications in the program: {with_bonus} with the bonus, "
          f"{without} without")
    assert without < with_bonus, \
        "removing the bonus did not remove any matrix multiplication"
    print("ok test_no_bonus_arithmetic_without_the_bonus")


def test_none_trains_and_its_intrinsic_reward_is_exactly_zero():
    """The arm still runs: the loss is finite and the intrinsic reward is zero at every step."""
    runner = Runner(PPOConfig(**CONFIG), bonus="none")
    state = runner.prime(runner.init_state(run_seed=3))
    for iteration in range(1, 4):
        state, metrics = runner.iterate(state, runner.lr_argument(iteration, 3))
        loss = float(np.asarray(metrics["loss"]))
        intrinsic = np.asarray(metrics["rint_mean"])
        assert loss == loss, "loss is not a number"
        assert (intrinsic == 0.0).all(), f"the intrinsic reward is not zero: {intrinsic}"
    print(f"three iterations of PPO with no bonus: last loss {loss:.6f}, "
          f"intrinsic reward exactly zero on every copy")
    print("ok test_none_trains_and_its_intrinsic_reward_is_exactly_zero")


if __name__ == "__main__":
    test_the_marked_widths_are_there_when_the_bonus_is()
    test_no_bonus_arithmetic_without_the_bonus()
    test_none_trains_and_its_intrinsic_reward_is_exactly_zero()

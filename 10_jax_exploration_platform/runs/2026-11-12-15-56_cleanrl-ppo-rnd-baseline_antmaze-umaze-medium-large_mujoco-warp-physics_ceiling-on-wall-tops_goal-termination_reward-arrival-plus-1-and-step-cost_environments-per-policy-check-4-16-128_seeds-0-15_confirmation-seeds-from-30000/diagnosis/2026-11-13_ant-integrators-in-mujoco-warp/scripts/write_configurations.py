"""Write one configuration file per physics setting into ../configurations/ (standard library only).

Each file is the U-maze stable-physics example file with the setting's integrator, physics time step and
physics steps per environment step written in (`integrator_options.with_setting`); every other value is the
example file's. These files are inputs of the speed measurement only; nothing under configs/ changes.
"""
from pathlib import Path

from integrator_options import NAMES, SETTINGS, with_setting

PLATFORM = Path("/p/rlprojects/RND/10_jax_exploration_platform")
SOURCE = PLATFORM / ("configs/experiments/antmaze_umaze_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-"
                     "metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_"
                     "example_steps-204800.toml")


def main():
    """One file per setting, with a header line naming the source and the change."""
    out = Path(__file__).resolve().parent.parent / "configurations"
    out.mkdir(exist_ok=True)
    text = SOURCE.read_text()
    for slug in SETTINGS:
        header = (f"# Integrator check of 2026-11-13: {NAMES[slug]}. Copy of {SOURCE.name}\n"
                  f"# with integrator, physics_time_step and frame_skip of [environment] set by scripts/write_configurations.py.\n")
        (out / f"antmaze_umaze_{slug}_guard-30-and-100.toml").write_text(header + with_setting(text, slug))
        print("wrote", out / f"antmaze_umaze_{slug}_guard-30-and-100.toml")


if __name__ == "__main__":
    main()

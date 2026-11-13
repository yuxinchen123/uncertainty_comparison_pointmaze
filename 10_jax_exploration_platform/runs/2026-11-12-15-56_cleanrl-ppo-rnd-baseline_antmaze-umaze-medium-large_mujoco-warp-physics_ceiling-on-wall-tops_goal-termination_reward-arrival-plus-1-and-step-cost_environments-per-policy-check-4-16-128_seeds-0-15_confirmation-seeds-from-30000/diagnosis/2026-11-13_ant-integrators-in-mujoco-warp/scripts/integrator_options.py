"""The five physics settings this check compares, and what each script needs to run them in MuJoCo Warp.

A setting changes three fields of the environment's configuration and nothing else: the integrator, the
physics time step and the number of physics steps per environment step (one environment step is 0.05 s in
all five). Solver and line-search iterations stay at 4 and 8, the fault guard at 30 m/s and 100 rad/s.

MJX 3.11.0 refuses a model whose integrator is `implicit` in `mujoco.mjx._src.io._put_option`, although
MuJoCo Warp implements it (`mujoco_warp/_src/forward.py`, `implicit`) and the Warp step reads the
integrator as a plain number (`mujoco/mjx/warp/forward.py`, `_m.opt.integrator = opt__integrator`).
`allow_full_implicit` is the wrapper of the speed study (`profiles/antmaze_warp_blow_up/units/
full_implicit_in_warp.py`): the option is put as `implicitfast` and its integrator number then set to
`implicit`. It changes nothing for any other integrator.

MuJoCo Warp's `implicit` step as shipped in MJX 3.11.0 adds the turning-rate (Coriolis and centrifugal) term of
the velocity derivative with the wrong sign: `forward.implicit` calls `derivative.deriv_rne_vel(m, d, d.qLU,
flg_subtract=True)`, which forms M - dt*D_smooth - dt*D_rne, where C MuJoCo forms M - dt*D_smooth + dt*D_rne
(`mjd_rne_vel` subtracts the term from qDeriv, and qLU = M - dt*qDeriv). Measured in
`implicit_four_way.py`: Warp's step minus its implicitfast step is the negative of C's (cosine -0.9999, length
ratio 0.997). `correct_full_implicit_sign` flips that one flag; the settings whose slug starts with
`implicit-sign-corrected` run with it.

Both patches live only in the scripts of this folder; the environment code and the installed library are not
changed.
"""
import re

# slug: (integrator as the model file names it, physics time step in seconds, physics steps per environment step)
SETTINGS = {
    "implicitfast_time-step-0.005_frame-skip-10": ("implicitfast", 0.005, 10),
    "implicit_time-step-0.005_frame-skip-10": ("implicit", 0.005, 10),
    "rk4_time-step-0.005_frame-skip-10": ("RK4", 0.005, 10),
    "rk4_time-step-0.01_frame-skip-5": ("RK4", 0.01, 5),
    "implicit_time-step-0.01_frame-skip-5": ("implicit", 0.01, 5),
    "implicit-sign-corrected_time-step-0.005_frame-skip-10": ("implicit", 0.005, 10),
    "implicit-sign-corrected_time-step-0.01_frame-skip-5": ("implicit", 0.01, 5),
}
NAMES = {
    "implicitfast_time-step-0.005_frame-skip-10": "implicitfast, 0.005 s x 10 (current)",
    "implicit_time-step-0.005_frame-skip-10": "implicit as shipped, 0.005 s x 10",
    "rk4_time-step-0.005_frame-skip-10": "RK4, 0.005 s x 10",
    "rk4_time-step-0.01_frame-skip-5": "RK4, 0.01 s x 5",
    "implicit_time-step-0.01_frame-skip-5": "implicit as shipped, 0.01 s x 5",
    "implicit-sign-corrected_time-step-0.005_frame-skip-10": "implicit with the sign corrected, 0.005 s x 10",
    "implicit-sign-corrected_time-step-0.01_frame-skip-5": "implicit with the sign corrected, 0.01 s x 5",
}
CURRENT = "implicitfast_time-step-0.005_frame-skip-10"
SIGN_CORRECTED_PREFIX = "implicit-sign-corrected_"


def correct_full_implicit_sign():
    """Make MuJoCo Warp's `implicit` step add the turning-rate derivative term with C MuJoCo's sign."""
    from mujoco.mjx.third_party.mujoco_warp._src import derivative

    if getattr(derivative.deriv_rne_vel, "sign_corrected", False):
        return
    shipped = derivative.deriv_rne_vel

    def deriv_rne_vel_with_c_sign(m, d, out, flg_subtract=False):
        """The shipped function with the opposite flag: its one caller (`forward.implicit`) passes True."""
        return shipped(m, d, out, flg_subtract=not flg_subtract)

    deriv_rne_vel_with_c_sign.sign_corrected = True
    derivative.deriv_rne_vel = deriv_rne_vel_with_c_sign


def prepare(slug):
    """The patches a setting needs, applied before any Warp model is built."""
    allow_full_implicit()
    if slug.startswith(SIGN_CORRECTED_PREFIX):
        correct_full_implicit_sign()


def allow_full_implicit():
    """Make `mjx.put_model(..., impl="warp")` accept a model whose integrator is `implicit`."""
    import mujoco
    from mujoco.mjx._src import io, types

    if getattr(io._put_option, "allows_full_implicit", False):
        return
    put_option = io._put_option
    full, fast = int(mujoco.mjtIntegrator.mjINT_IMPLICIT), int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)

    def put_option_allowing_full_implicit(option, impl, impl_fields=None):
        """MJX's option of a C option; for the Warp implementation `implicit` is carried as its number."""
        if impl != types.Impl.WARP or int(option.integrator) != full:
            return put_option(option, impl, impl_fields)
        option.integrator = fast
        try:
            put = put_option(option, impl, impl_fields)
        finally:
            option.integrator = full
        return put.replace(integrator=full)

    put_option_allowing_full_implicit.allows_full_implicit = True
    io._put_option = put_option_allowing_full_implicit


def with_setting(text, slug):
    """A configuration file's text with the setting's three environment values written in.

    before: '[environment]\\n...frame_skip = 10\\nintegrator = "implicitfast"\\n...physics_time_step = 0.005\\n'
    after (slug rk4_time-step-0.01_frame-skip-5): 'frame_skip = 5', 'integrator = "RK4"', 'physics_time_step = 0.01'
    Each of the three lines must occur exactly once in the [environment] section.
    """
    integrator, time_step, frame_skip = SETTINGS[slug]
    head, environment = text.split("[environment]\n", 1)
    rest = ""
    if "\n[" in environment:
        environment, rest = environment.split("\n[", 1)
        rest = "\n[" + rest
    for field, value in (("integrator", f'"{integrator}"'), ("physics_time_step", repr(time_step)),
                         ("frame_skip", str(frame_skip))):
        environment, count = re.subn(rf"(?m)^{field} = .*$", f"{field} = {value}", environment)
        if count != 1:
            raise ValueError(f"{field} occurs {count} times in the [environment] section")
    return head + "[environment]\n" + environment + rest


def apply_to_config(cfg, slug):
    """An AntMazeConfig with the setting's three values."""
    from dataclasses import replace
    integrator, time_step, frame_skip = SETTINGS[slug]
    return replace(cfg, integrator=integrator, physics_time_step=time_step, frame_skip=frame_skip)

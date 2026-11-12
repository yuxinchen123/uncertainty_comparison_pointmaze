"""Knobs of the random-network-distillation bonus."""
from dataclasses import dataclass, replace

REWARD_DIVISORS = ("running_std", "frozen_after_priming", "none")
BONUS_FORMS = ("half_squared_error", "error_length")
# the predictor's own learning rate is given as a multiple of this rate
PREDICTOR_RATE_UNIT = 1e-4


@dataclass(frozen=True)
class RNDConfig:
    """The two network widths, and the options of the study that started 2026-11-11.

    Every option's default is the bonus as it always was, so a run that sets none of them
    compiles the program it always did.
    """
    feature_dim: int = 128   # width of the target's output, which the predictor tries to match
    hidden: int = 256        # width of both networks' hidden layers

    # ---- batch 1 of the PointMaze and AntMaze study: the start, the scale and the decay of the
    # bonus (2026-11-12). One value for the whole process, never one per copy ----
    even_start: bool = False             # the predictor's output layer starts at zero (weights
                                         # and bias) and the target's output is scaled to unit
                                         # length per input, so the first error is the same at
                                         # every input (half squared error 0.5, length 1)
    reward_divisor: str = "running_std"  # what the trainer divides the bonus by.
                                         # "running_std": the running standard deviation of the
                                         # bonus's discounted sum, as always.
                                         # "frozen_after_priming": the same statistic gathered
                                         # over the random-action priming rollouts and never
                                         # moved afterwards. "none": no division
    predictor_learning_rate_multiple: float = 0.0  # 0: the predictor is stepped by the policy's
                                         # Adam, at the policy's rate, inside the joint gradient
                                         # clip, as always. m > 0: the predictor has its own Adam
                                         # at the constant rate m x 1e-4 (never annealed), and its
                                         # gradient is left out of the clip
    bonus_form: str = "half_squared_error"  # "error_length": the bonus is the length of the
                                         # error vector instead of half its square. The
                                         # predictor's loss is the mean squared error either way
    fast_last_layer_step: float = 0.0    # eta. 0: off. eta in (0, 1]: when a rollout is scored,
                                         # its steps are visited in time order, and after a step
                                         # is scored a temporary copy of the predictor's output
                                         # layer takes the smallest change that multiplies the
                                         # error of that step's states by 1 - eta. The copy is
                                         # dropped after the rollout
    scheduled_last_layer_decay: float = 0.0  # a. 0: off. a > 0: the predictor's body stays at
                                         # its random start and is never trained; its output
                                         # layer starts at zero, the target has unit length, and
                                         # each visit moves the visited state's squared error
                                         # one place along (1 + m)^(-a). The fitted layer is kept
                                         # between rollouts. Sets even_start and
                                         # reward_divisor "none"
    record_bonus_on_maze_grid: bool = False  # no change to training: per logging window and
                                         # policy, record the bonus at one probe observation per
                                         # open maze cell (cell centre, zero velocity), the
                                         # divisor, and the visits to each cell in the window


def resolved_rnd_config(rnd_cfg: RNDConfig) -> RNDConfig:
    """Refuse option values and combinations that are not built; fill in what one option implies.

    before: RNDConfig(scheduled_last_layer_decay=0.5)
    after:  the same with even_start=True and reward_divisor="none", which the schedule is
            defined with (its squared error starts at 1 and is at most about 1, so the
            schedule's own numbers are the bonus)
    """
    # each option's own range
    if rnd_cfg.reward_divisor not in REWARD_DIVISORS:
        raise ValueError(f"unknown reward_divisor {rnd_cfg.reward_divisor!r}; choose from "
                         f"{list(REWARD_DIVISORS)}")
    if rnd_cfg.bonus_form not in BONUS_FORMS:
        raise ValueError(f"unknown bonus_form {rnd_cfg.bonus_form!r}; choose from "
                         f"{list(BONUS_FORMS)}")
    if rnd_cfg.predictor_learning_rate_multiple < 0:
        raise ValueError("predictor_learning_rate_multiple must be zero (the policy's optimizer) "
                         f"or positive; it is {rnd_cfg.predictor_learning_rate_multiple}")
    if not 0 <= rnd_cfg.fast_last_layer_step <= 1:
        raise ValueError("fast_last_layer_step must be between 0 (off) and 1 (the exact fit); "
                         f"it is {rnd_cfg.fast_last_layer_step}")
    if rnd_cfg.scheduled_last_layer_decay < 0:
        raise ValueError("scheduled_last_layer_decay must be zero (off) or positive; it is "
                         f"{rnd_cfg.scheduled_last_layer_decay}")
    if rnd_cfg.scheduled_last_layer_decay == 0:
        return rnd_cfg

    # the scheduled last layer replaces the predictor's training and defines its own bonus
    if rnd_cfg.fast_last_layer_step > 0 or rnd_cfg.predictor_learning_rate_multiple > 0:
        raise ValueError("scheduled_last_layer_decay trains no predictor, so it cannot be "
                         "combined with fast_last_layer_step or "
                         "predictor_learning_rate_multiple")
    if rnd_cfg.bonus_form != "half_squared_error":
        raise ValueError("scheduled_last_layer_decay defines the bonus as the squared error; "
                         "leave bonus_form at its default")
    if rnd_cfg.reward_divisor == "frozen_after_priming":
        raise ValueError("scheduled_last_layer_decay is defined with no divisor; leave "
                         "reward_divisor at its default or set it to 'none'")
    return replace(rnd_cfg, even_start=True, reward_divisor="none")

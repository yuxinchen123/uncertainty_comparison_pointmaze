"""Knobs of the random-network-distillation bonus."""
from dataclasses import dataclass, replace

REWARD_DIVISORS = ("running_std", "frozen_after_priming", "held_after_fraction",
                   "held_after_first_arrival", "none")
BONUS_FORMS = ("half_squared_error", "error_length")
INPUT_STATISTICS = ("running", "frozen_after_priming")
INPUT_FEATURES = ("raw", "random_fourier")
SCHEDULED_LAST_LAYER_FEATURES = ("body", "random_fourier")
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
                                         # moved afterwards. "held_after_fraction": the running
                                         # statistic until the fraction divisor_hold_fraction of
                                         # the run's planned iterations, and the value it holds
                                         # then for the rest of the run.
                                         # "held_after_first_arrival": per policy, the running
                                         # statistic up to and including the rollout in which
                                         # the policy first reaches the goal, and the value it
                                         # holds then for the rest of the run; a policy that
                                         # never arrives keeps the running statistic. Needs
                                         # the trainer's episode log and episodes that end at
                                         # the goal. "none": no division
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

    # ---- batch 2 of the same study: where the bonus points at the start, and features that keep
    # the fit local (2026-11-12). One value for the whole process, never one per copy ----
    input_clip: float = 5.0              # c. The whitened input is clipped to +-c before the
                                         # networks read it. 0: no clip
    input_statistics: str = "running"    # the mean and variance the input is whitened with.
                                         # "running": moved on by every rollout, as always.
                                         # "frozen_after_priming": gathered over the
                                         # random-action priming rollouts and never moved
                                         # afterwards, so a place keeps the same whitened input
                                         # for the whole run
    input_features: str = "raw"          # what the target and the predictor read. "raw": the
                                         # whitened input x. "random_fourier": phi(x) =
                                         # sqrt(2 / D) cos(Omega x + beta), D features, the rows
                                         # of Omega normal with standard deviation 1 / l, beta
                                         # uniform on [0, 2 pi), drawn once per policy and fixed.
                                         # Two inputs at distance d have features with inner
                                         # product about exp(-d^2 / (2 l^2)). Sets
                                         # input_statistics "frozen_after_priming" and no clip
    fourier_feature_count: int = 256     # D, the number of random Fourier features
    fourier_length_scale: float = 1.0    # l, in units of the whitened input
    scheduled_last_layer_features: str = "body"  # what the scheduled last layer fits its output
                                         # layer on. "body": the last hidden features of the
                                         # predictor's random body, with a bias input.
                                         # "random_fourier": phi(x) scaled to unit length, with
                                         # no bias input and no predictor body at all; the target
                                         # network still reads x. Sets input_statistics
                                         # "frozen_after_priming" and no clip
    divisor_hold_fraction: float = 0.0   # f, read with reward_divisor "held_after_fraction"
                                         # only, where 0 < f < 1: iterations numbered from 0
                                         # below f x the run's planned number of iterations
                                         # divide as "running_std" does; from that iteration on
                                         # the divisor's statistics are not moved


def resolved_rnd_config(rnd_cfg: RNDConfig) -> RNDConfig:
    """Refuse option values and combinations that are not built; fill in what one option implies.

    before: RNDConfig(scheduled_last_layer_decay=0.5)
    after:  the same with even_start=True and reward_divisor="none", which the schedule is
            defined with (its squared error starts at 1 and is at most about 1, so the
            schedule's own numbers are the bonus)
    before: RNDConfig(input_features="random_fourier")
    after:  the same with input_statistics="frozen_after_priming" and input_clip=0.0 (the
            features are bounded, and statistics that move would move the features of a place)
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
    if rnd_cfg.reward_divisor == "held_after_fraction":
        if not 0 < rnd_cfg.divisor_hold_fraction < 1:
            raise ValueError("divisor_hold_fraction must be above 0 and below 1 with "
                             "reward_divisor='held_after_fraction'; it is "
                             f"{rnd_cfg.divisor_hold_fraction}")
    elif rnd_cfg.divisor_hold_fraction != 0:
        raise ValueError("divisor_hold_fraction is read only with "
                         "reward_divisor='held_after_fraction'; the divisor is "
                         f"{rnd_cfg.reward_divisor!r}")
    if rnd_cfg.input_clip < 0:
        raise ValueError(f"input_clip must be zero (no clip) or positive; it is "
                         f"{rnd_cfg.input_clip}")
    if rnd_cfg.input_statistics not in INPUT_STATISTICS:
        raise ValueError(f"unknown input_statistics {rnd_cfg.input_statistics!r}; choose from "
                         f"{list(INPUT_STATISTICS)}")
    if rnd_cfg.input_features not in INPUT_FEATURES:
        raise ValueError(f"unknown input_features {rnd_cfg.input_features!r}; choose from "
                         f"{list(INPUT_FEATURES)}")
    if rnd_cfg.scheduled_last_layer_features not in SCHEDULED_LAST_LAYER_FEATURES:
        raise ValueError("unknown scheduled_last_layer_features "
                         f"{rnd_cfg.scheduled_last_layer_features!r}; choose from "
                         f"{list(SCHEDULED_LAST_LAYER_FEATURES)}")
    if rnd_cfg.fourier_feature_count < 1:
        raise ValueError("fourier_feature_count must be at least 1; it is "
                         f"{rnd_cfg.fourier_feature_count}")
    if rnd_cfg.fourier_length_scale <= 0:
        raise ValueError("fourier_length_scale must be positive; it is "
                         f"{rnd_cfg.fourier_length_scale}")
    scheduled = rnd_cfg.scheduled_last_layer_decay > 0
    fourier_input = rnd_cfg.input_features == "random_fourier"
    fourier_layer = rnd_cfg.scheduled_last_layer_features == "random_fourier"

    # random Fourier features: where they are read, and the input they are defined on
    defaults = RNDConfig()
    if fourier_layer and not scheduled:
        raise ValueError("scheduled_last_layer_features names what the scheduled last layer is "
                         "fitted on; set scheduled_last_layer_decay above zero with it")
    if fourier_input and scheduled:
        raise ValueError("input_features='random_fourier' is built for the trained predictor; "
                         "the scheduled last layer reads random Fourier features with "
                         "scheduled_last_layer_features='random_fourier'")
    if fourier_input or fourier_layer:
        if rnd_cfg.input_clip not in (defaults.input_clip, 0.0):
            raise ValueError("random Fourier features are defined on the whitened input with no "
                             "clip; leave input_clip at its default or set it to 0")
        rnd_cfg = replace(rnd_cfg, input_statistics="frozen_after_priming", input_clip=0.0)
    elif (rnd_cfg.fourier_feature_count != defaults.fourier_feature_count
          or rnd_cfg.fourier_length_scale != defaults.fourier_length_scale):
        raise ValueError("fourier_feature_count and fourier_length_scale are read only with "
                         "input_features='random_fourier' or "
                         "scheduled_last_layer_features='random_fourier'; neither is set")
    if not scheduled:
        return rnd_cfg

    # the scheduled last layer replaces the predictor's training and defines its own bonus
    if rnd_cfg.fast_last_layer_step > 0 or rnd_cfg.predictor_learning_rate_multiple > 0:
        raise ValueError("scheduled_last_layer_decay trains no predictor, so it cannot be "
                         "combined with fast_last_layer_step or "
                         "predictor_learning_rate_multiple")
    if rnd_cfg.bonus_form != "half_squared_error":
        raise ValueError("scheduled_last_layer_decay defines the bonus as the squared error; "
                         "leave bonus_form at its default")
    if rnd_cfg.reward_divisor not in ("running_std", "none"):
        raise ValueError("scheduled_last_layer_decay is defined with no divisor; leave "
                         "reward_divisor at its default or set it to 'none'")
    return replace(rnd_cfg, even_start=True, reward_divisor="none")

"""Random network distillation on the next observation, expressed through the bonus hooks.

The bonus of a state is half the squared distance between a fixed random target network's
features of that state and a trained predictor's. The predictor is trained on the states the
agent actually reached, so the distance shrinks where the agent has been and stays large where it
has not.

Trainable parameters: the predictor. Statistics: the running mean and variance of the network's
input, so a whitened input keeps the same scale as the agent moves through the maze. The
predictor's own error is also the loss the optimizer minimises, and it rides the agent's Adam —
one optimizer over {agent, bonus}, which is what the frozen 09_parallelization baseline did.
"""
from ...statistics import rms_init, rms_update
from ..protocol import BonusFunctions
from .config import RNDConfig
from .networks import features, init_predictor, init_target, whiten


def build(cfg, rnd_cfg: RNDConfig, n_copies: int, base_seed: int, copy_seed_index):
    """Bind this bonus to one run's copy count, seeds and statistics setting."""
    # the target is a constant of the program, not a parameter: it is drawn once here and never
    # appears in any gradient
    target = init_target(rnd_cfg, n_copies, base_seed, copy_seed_index)

    def init():
        """The predictor's weights, and fresh statistics for the four observation dimensions."""
        return ({"predictor": init_predictor(rnd_cfg, n_copies, base_seed, copy_seed_index)},
                {"obs_rms": rms_init(n_copies, 4)})

    def score(params, state, next_obs):
        """Intrinsic reward of [C, M, 4] next observations -> [C, M]."""
        tf, pf = features(target, params["predictor"], whiten(next_obs, state["obs_rms"]))
        return 0.5 * ((pf - tf) ** 2).sum(-1)

    def prime(params, state, next_obs_flat):
        """Warm-up: fold a rollout of random actions into the input statistics, nothing else."""
        return {"obs_rms": rms_update(state["obs_rms"], next_obs_flat, cfg.batch_stats_f32)}

    def post_rollout(params, state, next_obs_flat, scored):
        """Reward the rollout, then move the statistics on and whiten the update batch's input.

        The reward uses the statistics as they stood at the START of the iteration — the same
        whitening the agent's actions were taken under — while the batch the update will see is
        whitened with the statistics that now include this rollout.
        """
        reward = score(params, state, next_obs_flat) if scored is None else scored
        obs_rms = rms_update(state["obs_rms"], next_obs_flat, cfg.batch_stats_f32)
        return {"obs_rms": obs_rms}, reward, {"rnd_input": whiten(next_obs_flat, obs_rms)}

    def loss(params, batch):
        """Per-copy predictor error on the update batch, [C]."""
        tf, pf = features(target, params["predictor"], batch["rnd_input"])
        return ((pf - tf) ** 2).mean(axis=2).mean(axis=1)

    def metrics(params, state):
        """This family reports nothing beyond the intrinsic reward the composer already logs."""
        return {}

    return BonusFunctions(name="rnd_next_state", init=init, prime=prime, rollout_step=score,
                          post_rollout=post_rollout, loss=loss, metrics=metrics)

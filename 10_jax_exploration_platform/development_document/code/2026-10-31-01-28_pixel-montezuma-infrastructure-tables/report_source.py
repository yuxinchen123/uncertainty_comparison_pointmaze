"""Human-reviewed sources and fixed protocol for the pixel infrastructure report.

This file contains specifications, never measured performance. Measurement rows
come only from result files explicitly indexed in the campaign study registry.
"""

from pathlib import Path


CAMPAIGN_ROOT = Path(
    "/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/"
    "benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_"
    "environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_"
    "steps-per-copy-2000000000_seed-start-1"
)
CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
JAXTARI_COMMIT = "04ab9aec318c72162e2e53503a594a489d7e162b"
CLEANRL_SOURCE = (
    "https://github.com/vwxyzjn/cleanrl/blob/" + CLEANRL_COMMIT
    + "/cleanrl/ppo_rnd_envpool.py"
)
JAXTARI_SOURCE = "https://github.com/k4ntz/JAXtari/tree/" + JAXTARI_COMMIT
ENVPOOL_REGISTRATION_SOURCE = (
    "https://github.com/sail-sg/envpool/blob/v0.6.6/envpool/atari/registration.py#L26-L35"
)
ENVPOOL_EPISODE_CAP_SOURCE = (
    "https://github.com/sail-sg/envpool/blob/v0.6.6/envpool/atari/atari_env.h#L168-L194"
)

# Each row is (category, parameter, value). Values use literal LaTeX, not data
# extracted from output logs; the generator checks the numeric resolved settings.
ENVIRONMENT_ROWS = (
    ("Game and observations", "game", "JAXtari Montezuma's Revenge; pixel observations"),
    (None, "actions", r"$18$ categorical actions"),
    (None, "observed image", r"$4\times84\times84$ unsigned-byte grayscale pixels"),
    (None, "action repeat", r"one action held for $4$ native game frames"),
    (None, "sticky action probability", r"$0.25$"),
    ("Image processing", "frame processing order",
     "grayscale each native frame, take the maximum of the last two held-action frames, "
     "area-average resize, and convert to unsigned bytes"),
    (None, "frame history", r"four processed frames, ordered from oldest to newest"),
    (None, "full-game reset history", "repeat the initial processed frame four times"),
    ("Episodes and rewards", "learning reward", "sign of the game reward"),
    (None, "reported game reward", "unclipped full-game score, recorded separately"),
    (None, "life loss", "an extrinsic episode boundary; the intrinsic return continues"),
    (None, "reset actions", r"$0$--$29$ native no-operation frames, then one native fire action"),
    (None, "reset timing", "native immediate reset within the transition"),
    (None, "life-loss history", "preserve the transition history and append one reset frame; "
     "a full game ending during reset replaces the history"),
    (None, "episode cap, matching EnvPool", r"$108{,}000$ held-action transitions per full game, "
     "including transitions before a life-loss reset"),
)

# Source and implementation share the fixed algorithm settings. Native immediate
# resets are the separately stated environment adaptation, not a new learner rule.
LEARNER_ROWS = (
    ("Batch and budget", "training budget per policy copy", r"$2$ billion environment transitions"),
    (None, "parallel environments per copy", r"$128$"),
    (None, "rollout steps per environment", r"$128$"),
    (None, "transitions per update per copy", r"$16{,}384$"),
    (None, "epochs per update", r"$4$"),
    (None, "minibatches per epoch", r"$4$, each containing $4{,}096$ transitions"),
    (None, "hyperparameter configurations", r"$1$ fixed setting; no hyperparameter search in this study"),
    ("Optimizer and policy objective", "Adam learning rate", r"$10^{-4}$, linearly reduced over the training budget"),
    (None, "Adam numerical constant", r"$10^{-5}$"),
    (None, "Adam moment coefficients", r"$0.9$ and $0.999$"),
    (None, "policy clip coefficient", r"$0.1$"),
    (None, "entropy coefficient", r"$0.001$"),
    (None, "value coefficient", r"$0.5$"),
    (None, "joint gradient norm limit", r"$0.5$ across policy and predictor parameters"),
    (None, "advantage normalization", "within each minibatch; sample standard deviation with correction one"),
    ("Returns and distillation", "extrinsic discount", r"$0.999$"),
    (None, "intrinsic discount", r"$0.99$"),
    (None, "generalized advantage estimation parameter", r"$0.95$"),
    (None, "intrinsic and extrinsic advantage weights", r"$1$ and $2$, respectively"),
    (None, "predictor update fraction", r"$0.25$, using a random row mask"),
    (None, "value losses", "clipped extrinsic value loss; unclipped intrinsic mean-squared error"),
    (None, "return boundaries", "extrinsic return stops at an episode boundary; intrinsic return is non-episodic"),
)

NETWORK_ROWS = (
    ("Networks", "policy convolution channels", r"$4\to32\to64\to64$; kernels $8,4,3$; strides $4,2,1$; rectified linear activations"),
    (None, "policy shared dense layers", r"$3{,}136\to256\to448$ with rectified linear activations"),
    (None, "policy output branches", r"$448$-unit actor layer and $18$ logits; residual $448$-unit critic layer and separate extrinsic and intrinsic value heads"),
    (None, "distillation input", r"the newest $84\times84$ grayscale frame"),
    (None, "distillation convolution channels", r"$1\to32\to64\to64$; kernels $8,4,3$; strides $4,2,1$; leaky rectified linear slope $0.01$"),
    (None, "predictor dense layers", r"$3{,}136\to512\to512\to512$; rectified linear activations between layers"),
    (None, "frozen target dense layer", r"$3{,}136\to512$"),
    ("Normalization and numerical precision", "policy pixel scaling", r"divide unsigned-byte values by $255$"),
    (None, "observation normalization warmup", r"$50$ random rollouts of $128$ steps, outside the training budget"),
    (None, "distillation observation normalization", r"per-pixel running mean and variance, then clip to $[-5,5]$"),
    (None, "network arithmetic", r"$32$-bit floating point, highest matrix-multiplication precision; no reduced-precision modes"),
    (None, "running statistics", r"$64$-bit floating point in the JAX implementation"),
)

RESOLVED_SETTINGS = {
    "environments_per_copy": 128, "rollout_steps": 128, "minibatches": 4, "epochs": 4,
    "learning_rate": 0.0001, "anneal_learning_rate": True, "adam_epsilon": 0.00001,
    "extrinsic_discount": 0.999, "intrinsic_discount": 0.99,
    "generalized_advantage_estimation_parameter": 0.95, "policy_clip_coefficient": 0.1,
    "entropy_coefficient": 0.001, "value_coefficient": 0.5, "joint_gradient_norm_limit": 0.5,
    "predictor_update_fraction": 0.25, "intrinsic_advantage_weight": 1.0,
    "extrinsic_advantage_weight": 2.0, "normalization_warmup_rollouts": 50,
    "network_precision": "float32", "statistics_precision": "float64",
    "allow_reduced_precision": False,
}

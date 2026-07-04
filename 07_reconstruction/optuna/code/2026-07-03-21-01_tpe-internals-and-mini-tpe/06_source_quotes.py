"""Print exact source of the TPE functions that back the report's claims (optuna 4.9.0).

These are verbatim quotes from the installed package via inspect.getsource, so the report can
cite exact code rather than paraphrase.
"""

import inspect

from optuna.samplers._tpe import sampler as S
from optuna.samplers._tpe import parzen_estimator as P


def show(title, obj):
    """Print a labeled block of a function/method's exact installed source."""
    print("=" * 78)
    print(title)
    print("-" * 78)
    print(inspect.getsource(obj))


if __name__ == "__main__":
    # Split-size and weighting functions.
    show("default_gamma  (n_below = number of 'good' trials)", S.default_gamma)
    show("default_weights  (per-observation down-weighting)", S.default_weights)
    # The candidate/acquisition pipeline: sample from below, score log l - log g, argmax.
    show("TPESampler._sample  (draws candidates from below, then compares)", S.TPESampler._sample)
    show("TPESampler._compute_acquisition_func  (log l - log g)",
         S.TPESampler._compute_acquisition_func)
    show("TPESampler._compare  (argmax of acquisition)", S.TPESampler._compare)
    # The numeric mixture: one component per observation + one prior component, plus sigmas.
    show("_ParzenEstimator._calculate_numerical_distributions  (mixture + magic clip)",
         P._ParzenEstimator._calculate_numerical_distributions)
    show("_ParzenEstimator._calculate_categorical_distributions  (weighted probs + prior)",
         P._ParzenEstimator._calculate_categorical_distributions)

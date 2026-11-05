"""Describe measured execution choices without changing scientific hyperparameters."""

from copy import deepcopy
from itertools import combinations
import math
from statistics import median


BASELINE = {"name": "baseline", "changes": {},
            "rationale": "Reference execution with 128 independent training copies."}

# Each candidate changes one execution choice; no statistical seed target is changed.
# Before: one baseline configuration. After: twenty separately measured configurations.
INITIAL_IDEAS = (
    {"name": "rollout_loop_unroll_four", "changes": {"rollout_unroll": 4},
     "rationale": "Reduce loop overhead when collecting transitions."},
    {"name": "update_loop_unroll_four", "changes": {"update_unroll": 4},
     "rationale": "Reduce loop overhead across optimizer updates."},
    {"name": "precompute_all_state_predictions", "changes": {"precompute_all_states": True},
     "rationale": "Reuse predictions when a rollout revisits the same states."},
    {"name": "all_state_minibatch_predictions", "changes": {"all_states_minibatch": True},
     "rationale": "Predict each state once within a minibatch and gather its visits."},
    {"name": "reuse_rollout_values", "changes": {"hoist_values": True},
     "rationale": "Avoid evaluating unchanged rollout values a second time."},
    {"name": "reuse_rollout_log_probabilities", "changes": {"hoist_log_probabilities": True},
     "rationale": "Reuse the recorded action log probabilities."},
    {"name": "reuse_count_bonus_terms", "changes": {"hoist_bonus": True},
     "rationale": "Reuse state bonus terms within the collected rollout."},
    {"name": "batch_random_key_derivation", "changes": {"batch_random_keys": True},
     "rationale": "Derive independent random streams in a single batched operation."},
    {"name": "flat_parameter_storage", "changes": {"flat_parameters": True},
     "rationale": "Reduce the number of optimizer arrays using flat parameter storage."},
    {"name": "single_optimizer_update_loop", "changes": {"flatten_update_scan": True},
     "rationale": "Use one loop across epochs and minibatches."},
    {"name": "gather_minibatches_before_updates", "changes": {"pre_gather_minibatches": True},
     "rationale": "Move repeated minibatch gathers outside the update loop."},
    {"name": "donate_training_state", "changes": {"donate_state": True},
     "rationale": "Allow the compiler to reuse consumed state buffers."},
    {"name": "parallel_advantage_sum", "changes": {"associative_gae": True},
     "rationale": "Compute the same generalized advantage sum with a parallel scan."},
    {"name": "reciprocal_square_root_bonus", "changes": {"rsqrt_bonus": True},
     "rationale": "Use a reciprocal square root operation for the count bonus."},
    {"name": "fused_optimizer_arithmetic", "changes": {"fuse_optimizer": True},
     "rationale": "Combine independent optimizer arithmetic into fewer operations."},
    {"name": "flat_state_prediction_gather", "changes": {"flat_state_gather": True},
     "rationale": "Gather state predictions through one flattened index."},
    {"name": "training_copies_32", "changes": {"n_copies": 32},
     "rationale": "Measure lower per-copy latency with 32 physical training copies."},
    {"name": "training_copies_64", "changes": {"n_copies": 64},
     "rationale": "Measure lower per-copy latency with 64 physical training copies."},
    {"name": "training_copies_256", "changes": {"n_copies": 256},
     "rationale": "Measure greater aggregate throughput with 256 training copies."},
    {"name": "training_copies_512", "changes": {"n_copies": 512},
     "rationale": "Measure greater aggregate throughput with 512 training copies."},
)


def initial_ideas(include_baseline=True):
    """Return independent JSON-compatible registry entries for the first measurements."""
    # A caller may attach results without modifying this module's source registry.
    entries = ((BASELINE,) if include_baseline else ()) + INITIAL_IDEAS
    return deepcopy(list(entries))


def resolve_idea(name, adaptive_ideas=()):
    """Find one named execution choice and reject missing or repeated names."""
    # Match full descriptive names; short numeric codes are never accepted.
    matches = [entry for entry in initial_ideas() + list(adaptive_ideas)
               if entry["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one execution choice named {name!r}; found {len(matches)}")
    return deepcopy(matches[0])


def _case_rows(records):
    """Flatten aggregate benchmark documents into their individual measurement cases."""
    # Before: {idea: name, cases: [{policy_kind: tabular, ...}]}.
    # After: [{idea: name, policy_kind: tabular, ...}].
    rows = []
    for record in records:
        cases = record.get("cases", [record])
        for case in cases:
            rows.append({**record, **case, "idea": record["idea"]})
    return rows


def choose_adaptive_ideas(records, count=10):
    """Choose new combinations from measured, correct initial execution choices."""
    # Require measurements rather than inventing successful constituents.
    if count < 1 or count > 10:
        raise ValueError("the adaptive batch must contain between one and ten ideas")
    rows = _case_rows(records)
    baselines = {(row["policy_kind"], row["geometry"]): row for row in rows
                 if row["idea"] == "baseline" and row.get("completed", False)}
    if not baselines:
        raise ValueError("adaptive ideas require completed baseline measurements")

    # Compare the same policy and geometry; unmatched or incorrect cases are ineligible.
    ratios = {}
    for row in rows:
        key = (row["policy_kind"], row["geometry"])
        if (row["idea"] == "baseline" or not row.get("completed", False)
                or not row.get("correctness", {}).get("passed", False) or key not in baselines):
            continue
        ratio = row["steps_per_second_total"] / baselines[key]["steps_per_second_total"]
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("measured throughput ratios must be finite and positive")
        ratios.setdefault(row["idea"], []).append(ratio)
    required_case_count = len(baselines)
    eligible = {entry["name"]: entry for entry in initial_ideas(False)
                if len(ratios.get(entry["name"], [])) == required_case_count}
    ranked = sorted(eligible, key=lambda name: (-median(ratios[name]), name))
    execution = [name for name in ranked if "n_copies" not in eligible[name]["changes"]]
    if len(execution) < 2:
        raise ValueError("adaptive combinations require two correct execution choices on every baseline case")

    # Prefer the best constituents, but still measure every combination independently.
    pairs = list(combinations(execution[:6], 2))
    pairs.sort(key=lambda pair: (-sum(math.log(median(ratios[name])) for name in pair), pair))
    if len(pairs) < count:
        for copies in (32, 64, 256, 512):
            pairs.append((execution[0], f"training_copies_{copies}"))
    if len(pairs) < count:
        raise ValueError("not enough measured constituents for the requested adaptive batch")

    # The reason names measured values and explicitly leaves the combined gain unmeasured.
    result = []
    for first, second in pairs[:count]:
        if second not in eligible:
            raise ValueError(f"adaptive copy choice {second!r} has no complete correct measurement")
        changes = {**eligible[first]["changes"], **eligible[second]["changes"]}
        reason = (f"The measured aggregate throughput ratios were {median(ratios[first]):.4f} "
                  f"for {first.replace('_', ' ')} and {median(ratios[second]):.4f} for "
                  f"{second.replace('_', ' ')}. The combined execution remains unmeasured.")
        result.append({"name": first + "_with_" + second, "changes": changes,
                       "rationale": reason, "constituents": [first, second], "phase": "adaptive"})
    return result

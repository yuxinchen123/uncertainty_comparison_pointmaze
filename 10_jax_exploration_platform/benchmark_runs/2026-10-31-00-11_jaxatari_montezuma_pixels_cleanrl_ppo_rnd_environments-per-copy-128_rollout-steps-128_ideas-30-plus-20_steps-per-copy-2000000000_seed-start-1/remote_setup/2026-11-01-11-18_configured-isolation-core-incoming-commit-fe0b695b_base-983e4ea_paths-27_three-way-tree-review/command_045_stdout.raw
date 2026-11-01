"""Keep the thirty agreed hypotheses and twenty unchosen follow-up slots explicit.

A catalog entry is a hypothesis, not a result. Parameter trials belong to the same
hypothesis. Entries without a distinct implementation remain pending and cannot
be counted as completed experiments or selected for training.
"""

import argparse
import ast
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_HARDWARE = ("h100", "jaguar03_rtx_a4500")


@dataclass(frozen=True)
class Idea:
    """One distinct initial hypothesis, with concrete changes or an honest pending status."""

    position: int
    name: str
    hypothesis: str
    mechanism: str
    learner: dict
    environment: dict
    execution: dict
    implementation_pending: bool = False
    implementation_note: str = ""
    parameter_trials: tuple = ()
    additional_checks: tuple = ()
    reference_execution: dict | None = None


def idea(position, name, hypothesis, mechanism, *, learner=None, environment=None,
         execution=None, pending=False, note="", trials=(), checks=()):
    """Construct a catalog row without sharing mutable change dictionaries."""
    return Idea(position, name, hypothesis, mechanism, dict(learner or {}),
                dict(environment or {}), dict(execution or {}), pending,
                note, tuple(trials), tuple(checks))


# These are the agreed thirty mechanisms. A pending row is deliberately not an alias
# for a nearby implemented optimization. For example, gathering a frame stack is
# not described as a circular buffer, and caching targets is not palette fusion.
INITIAL_IDEAS = (
    idea(1, "cache_room_backgrounds", "Cache static room backgrounds between renders.",
         "Avoid rebuilding unchanged room geometry while drawing dynamic objects normally.",
         environment={"cache_room_backgrounds": True},
         checks=("exact pixels across multiple rooms, dark rooms, inventory and lava states",)),
    idea(2, "render_final_two_game_states", "Render only the two held-action states needed for pooling.",
         "Carry two game states through frame skipping instead of rendering every native frame.",
         environment={"render_final_two_only": True},
         checks=("exact early-game-over and final-two-frame pooling results",)),
    idea(3, "unroll_four_frame_game_loop", "Unroll the fixed four-frame held-action loop.",
         "Expose independent work within the native-frame loop to the compiler.",
         environment={"frame_skip_unroll": 4}, trials=({"frame_skip_unroll": 2},
                                                     {"frame_skip_unroll": 4})),
    idea(4, "nested_policy_environment_batching", "Compare nested and flat game-state batching.",
         "Change batching layout without sharing state or weights across policy copies.",
         environment={"batch_layout": "nested"}),
    idea(5, "fuse_pixel_preprocessing", "Fuse grayscale, maximum pooling, area resize and byte output.",
         "Avoid intermediate image buffers while preserving the agreed preprocessing order.",
         environment={"fuse_pixel_preprocessing": True},
         note="Requires a distinct preprocessing implementation, not an unchanged JIT wrapper.",
         checks=("exact uint8 observations on supplied native frames",)),
    idea(6, "combine_palette_and_sprite_drawing", "Use grayscale palette lookup after ordered sprite drawing.",
         "Keep the original ordered palette-index rasterization, then use a precomputed grayscale "
         "palette to avoid creating an RGB intermediate image.",
         environment={"combine_palette_and_sprite_drawing": True},
         checks=("exact pixels for overlapping sprites and dynamic room objects",)),
    idea(7, "store_rollout_observations_as_bytes", "Keep rollout observations as bytes until network input.",
         "Reduce rollout storage and memory transfers relative to upstream float32 buffers.",
         learner={"rollout_observation_storage": "uint8"}),
    idea(8, "reconstruct_stacks_from_individual_frames", "Store individual frames and reconstruct reset-aware stacks.",
         "Avoid storing four overlapping frames for every rollout row.",
         learner={"rollout_frame_storage": "indexed_frames", "rollout_observation_storage": "uint8"},
         note="The indexed bank inherently stores uint8 pixels: 2*T+8 frames per lane, "
              "rather than 4*T stacked frames. Both representation changes are recorded.",
         checks=("stack reconstruction across life loss, full reset and rollout boundaries",)),
    idea(9, "circular_frame_stack", "Replace frame-stack concatenation with a circular buffer.",
         "Move stack indices rather than repeatedly copying earlier frames.",
         environment={"stack_update": "circular"},
         note="Compare canonical chronological frames; the physical circular representation differs.",
         checks=("exact network-visible chronological stacks after every reset",)),
    idea(10, "fuse_minibatch_image_preparation", "Fuse minibatch gather, image conversion and normalization.",
         "Read selected image bytes once before producing normalized network inputs.",
         learner={"fuse_minibatch_image_preparation": True}),
    idea(11, "donate_training_buffers", "Donate old learner buffers for reuse.",
         "Permit reuse of parameter, optimizer and environment allocations.",
         learner={"donate_state": True}, checks=("independently owned input buffers",)),
    idea(12, "recompute_network_activations", "Recompute selected activations during differentiation.",
         "Exchange repeated arithmetic for smaller saved activation buffers.",
         learner={"recompute_network_activations": True}),
    idea(13, "channels_last_convolutions", "Compare channels-first and channels-last convolutions.",
         "Change convolution layout while retaining original float32 weights and outputs.",
         learner={"network_layout": "nhwc"}),
    idea(14, "group_independent_convolutions", "Compare vectorized and grouped independent convolutions.",
         "Give the compiler one grouped operation with separate weights for every policy.",
         learner={"convolution_batching": "grouped"}),
    idea(15, "einsum_independent_dense_layers", "Compare equivalent dense-layer batch operations.",
         "Change matrix batching without mixing policy parameters or rows.",
         learner={"dense_batching": "einsum"}),
    idea(16, "fuse_network_bias_and_activation", "Fuse network bias and activation with linear operations.",
         "Remove intermediate network arrays if the reference compiler has not already done so.",
         learner={"fuse_network_bias_and_activation": True},
         note="An existing compiler fusion alone is not a distinct measured candidate."),
    idea(17, "cache_frozen_training_target_features", "Cache frozen RND training targets within one update.",
         "Reuse targets on current frames under updated observation statistics across minibatches.",
         learner={"cache_target_features": True},
         checks=("current training frames and updated statistics; predictor recomputed each step",)),
    idea(18, "batch_rollout_intrinsic_rewards", "Calculate rollout intrinsic rewards after collection.",
         "Batch next-frame RND calculations using unchanged parameters and pre-rollout statistics.",
         learner={"hoist_rollout_bonus": True}),
    idea(19, "flat_parameter_and_optimizer_buffers", "Keep per-policy parameters and moments in flat update buffers.",
         "Pack once across all minibatches, retaining independent per-policy clipping.",
         learner={"flat_parameters": True}),
    idea(20, "fuse_gradient_clipping_and_adam", "Fuse per-policy gradient norm, clipping and Adam work.",
         "Combine elementwise optimizer work within each minibatch.",
         learner={"fuse_optimizer": True}),
    idea(21, "single_sixteen_minibatch_scan", "Run all sixteen minibatch updates in one compiled loop.",
         "Flatten the epoch and minibatch loops while retaining each epoch's permutation.",
         learner={"flatten_update_scan": True},
         note="The reference already compiles nested loops; a second JIT wrapper is not a candidate."),
    idea(22, "unroll_minibatch_update_loop", "Tune update-loop unrolling.",
         "Expose adjacent minibatch work without changing its order or number of updates.",
         learner={"update_unroll": 2}, trials=({"update_unroll": 2}, {"update_unroll": 4})),
    idea(23, "gather_epoch_minibatches_in_advance", "Compare on-demand and pre-gathered minibatches.",
         "Prepare an epoch's independent policy minibatches before scanning updates.",
         learner={"pre_gather_minibatches": True}),
    idea(24, "associative_advantage_scan", "Compare reverse and associative advantage calculations.",
         "Use parallel composition of the same discounted recurrences without changing masks.",
         learner={"associative_gae": True}),
    idea(25, "separate_rollout_and_update_compilation", "Compare one compiled iteration with separate compiled phases.",
         "Limit compilation regions while retaining the same rollout, preparation and update.",
         execution={"compilation_mode": "separate_phases"}),
    idea(26, "unroll_policy_rollout_loop", "Tune the 128-step policy rollout loop's unrolling.",
         "Expose adjacent policy rollout work while retaining every transition.",
         learner={"rollout_unroll": 4}, trials=({"rollout_unroll": 2},
                                              {"rollout_unroll": 4}, {"rollout_unroll": 8})),
    idea(27, "group_complete_training_iterations", "Run several complete iterations in one compiled call.",
         "Reduce host dispatches while preserving iteration counters and annealing positions.",
         execution={"iterations_per_call": 5}, trials=({"iterations_per_call": 2},
                                                     {"iterations_per_call": 5})),
    idea(28, "batch_quantity_keyed_random_generation", "Batch random draws using the same quantity-specific keys.",
         "Reduce repeated random-generation dispatch without shifting any existing stream.",
         learner={"batch_quantity_keyed_random_generation": True},
         checks=("identical action, permutation and predictor-mask keys",)),
    Idea(29, "accumulate_scheduled_reports_on_device", "Accumulate scheduled reports and episode summaries on the GPU.",
         "Read reports only when due while preserving their window positions and totals.",
         {}, {}, {"report_readout": "scheduled", "report_every_iterations": 10},
         implementation_note="A separate driver-overhead comparison: both variants emit the same rows; "
                             "the reference downloads every update and the candidate downloads only the window.",
         reference_execution={"report_readout": "per_update", "report_every_iterations": 10}),
    idea(30, "combine_running_statistic_reductions", "Combine normalization reductions and running-statistic updates.",
         "Compute equivalent float64 moments without changing source axes, counts or update order.",
         learner={"moment_reduction": "raw_moments"},
         checks=("float64 moment comparison under large means and small variance",)),
)
IDEAS_BY_NAME = {entry.name: entry for entry in INITIAL_IDEAS}


def source_capabilities(platform_root=PLATFORM_ROOT):
    """Inspect constructor fields without importing JAX or creating any device arrays."""
    config_path = platform_root / "src/exploration_platform/atari_pixels/config.py"
    environment_path = platform_root / "src/exploration_platform/atari_pixels/environment.py"
    configuration = ast.parse(config_path.read_text())
    environment = ast.parse(environment_path.read_text())
    config_class = next(node for node in configuration.body
                        if isinstance(node, ast.ClassDef) and node.name == "PixelPPOConfig")
    fields = {node.target.id for node in config_class.body
              if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    consumed = set()
    for path in config_path.parent.glob("*.py"):
        if path == config_path:
            continue
        consumed.update(node.attr for node in ast.walk(ast.parse(path.read_text()))
                        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load))
    environment_class = next(node for node in environment.body
                             if isinstance(node, ast.ClassDef) and node.name == "AtariPixelEnvironment")
    constructor = next(node for node in environment_class.body
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    arguments = {node.arg for node in constructor.args.args + constructor.args.kwonlyargs}
    return {"learner": fields & consumed, "environment": arguments,
            "execution": {"compilation_mode", "iterations_per_call", "report_readout",
                          "report_every_iterations"}}


def implementation_status(entry, capabilities=None):
    """Refuse placeholders and unavailable switches instead of recording an unchanged benchmark."""
    capabilities = source_capabilities() if capabilities is None else capabilities
    missing = {scope: sorted(set(getattr(entry, scope)) - capabilities[scope])
               for scope in ("learner", "environment", "execution")}
    missing = {scope: names for scope, names in missing.items() if names}
    return {"state": ("pending_implementation" if entry.implementation_pending or missing
                       else "ready_for_correctness_check"),
            "missing_switches": missing, "note": entry.implementation_note}


def canonical_signature(learner, environment, execution):
    """Give concrete configurations an exact identity independent of hypothesis names."""
    return json.dumps({"learner": learner, "environment": environment,
                       "execution": execution}, sort_keys=True, separators=(",", ":"))


def trial_changes(entry, trial_index=None):
    """Resolve one parameter trial without turning it into another hypothesis."""
    changes = {scope: dict(getattr(entry, scope)) for scope in ("learner", "environment", "execution")}
    if trial_index is None:
        return changes
    if trial_index < 0 or trial_index >= len(entry.parameter_trials):
        raise ValueError("parameter trial index is outside this hypothesis's listed trials")
    for name, value in entry.parameter_trials[trial_index].items():
        scopes = [scope for scope in changes if name in changes[scope]]
        if len(scopes) != 1:
            raise ValueError(f"parameter trial {name!r} does not belong to exactly one change scope")
        changes[scopes[0]][name] = value
    return changes


def registry_idea(name, registry=None):
    """Resolve an initial hypothesis or one of the twenty later specified mechanisms."""
    if name in IDEAS_BY_NAME:
        return IDEAS_BY_NAME[name]
    if registry is None:
        raise ValueError("an additional hypothesis needs its persisted study registry")
    matches = [entry for batch in registry["adaptive_batches"] for entry in batch["hypotheses"]
               if entry["name"] == name]
    if len(matches) != 1 or matches[0]["state"] == "not_yet_chosen":
        raise ValueError("the additional hypothesis has not been specified exactly once")
    entry = matches[0]
    return Idea(entry["position"], name, entry["hypothesis"], entry["mechanism"],
                entry["learner"], entry["environment"], entry["execution"],
                parameter_trials=tuple(entry.get("parameter_trials", ())),
                additional_checks=tuple(entry["correctness_checks"]),
                reference_execution=entry.get("reference_execution"))


def new_registry(reference_commit, platform_root=PLATFORM_ROOT):
    """Build the initial study record while leaving all twenty adaptive hypotheses unchosen."""
    capabilities = source_capabilities(platform_root)
    return {
        "schema_version": 1, "reference_commit": reference_commit,
        "required_hardware": list(REQUIRED_HARDWARE),
        "initial_ideas": [{**asdict(entry), "implementation": implementation_status(entry, capabilities),
                           "results": {hardware: [] for hardware in REQUIRED_HARDWARE}}
                          for entry in INITIAL_IDEAS],
        "adaptive_batches": [{"batch": batch + 1, "reference_commit": None,
                              "profiling_evidence": None,
                              "hypotheses": [{"position": 31 + batch * 5 + offset,
                                              "name": None, "hypothesis": None,
                                              "state": "not_yet_chosen", "results": {}}
                                             for offset in range(5)]}
                             for batch in range(4)],
    }


def measured_hypotheses(registry):
    """Count mechanisms with valid results on both GPU types, never their parameter trials."""
    completed = []
    for entry in registry["initial_ideas"]:
        if all(any(result.get("measurement_valid") and result.get("correctness_passed")
                   and result.get("implementation_distinct")
                   for result in entry["results"].get(hardware, []))
               for hardware in REQUIRED_HARDWARE):
            completed.append(entry["name"])
    for batch in registry["adaptive_batches"]:
        for entry in batch["hypotheses"]:
            if entry["name"] and all(any(result.get("measurement_valid")
                                         and result.get("correctness_passed")
                                         and result.get("implementation_distinct")
                                         for result in entry["results"].get(hardware, []))
                                     for hardware in REQUIRED_HARDWARE):
                completed.append(entry["name"])
    return completed


def refresh_implementations(registry, platform_root=PLATFORM_ROOT):
    """Refresh availability after code changes without changing any experiment result."""
    capabilities = source_capabilities(platform_root)
    for entry in registry["initial_ideas"]:
        specification = IDEAS_BY_NAME[entry["name"]]
        entry.update(asdict(specification))
        entry["implementation"] = implementation_status(specification, capabilities)
    return registry


def investigated_hypotheses(registry):
    """Count finished numerical rejections as investigated, without calling them measured gains."""
    entries = list(registry["initial_ideas"])
    entries.extend(entry for batch in registry["adaptive_batches"] for entry in batch["hypotheses"])
    finished = []
    for entry in entries:
        if not entry["name"]:
            continue
        for hardware in REQUIRED_HARDWARE:
            records = entry["results"].get(hardware, [])
            valid = any(row.get("implementation_distinct") and (
                row.get("measurement_valid") and row.get("correctness_passed")
                or row.get("correctness_attempted") and row.get("failure_stage") == "correctness"
                and row.get("decision") == "rejected") for row in records)
            if not valid:
                break
        else:
            finished.append(entry["name"])
    return finished


def record_result(registry, result, result_path):
    """Index a durable measurement once, keeping invalid and rejected results in the record."""
    hardware = result["hardware_class"]
    if hardware not in REQUIRED_HARDWARE:
        raise ValueError("only the two specified GPU classes belong in this registry")
    entries = list(registry["initial_ideas"])
    entries.extend(entry for batch in registry["adaptive_batches"] for entry in batch["hypotheses"])
    matches = [entry for entry in entries if entry["name"] == result["idea"]]
    if len(matches) != 1:
        raise ValueError("the measurement must name one specified hypothesis")
    before, after = result.get("reference"), result.get("candidate")
    distinct = bool(before and after and canonical_signature(**before) != canonical_signature(**after))
    if result.get("measurement_valid") and not distinct:
        raise ValueError("an unchanged configuration cannot be counted as a valid experiment")
    summary_fields = ("idea", "copies", "hardware_class", "decision", "correctness_passed",
                      "measurement_valid", "median_speedup", "median_candidate_transitions_per_second_total",
                      "paired_rounds", "contaminated", "measurement_scope")
    summary_fields += ("correctness_attempted", "failure_stage")
    summary = {name: result[name] for name in summary_fields if name in result}
    summary.update(result_path=str(result_path), implementation_distinct=distinct)
    records = matches[0]["results"].setdefault(hardware, [])
    if any(row["result_path"] == str(result_path) for row in records):
        raise ValueError("this result file is already indexed; retain reruns in separate files")
    records.append(summary)
    return registry


def register_adaptive_batch(registry, hypotheses, profiling_evidence, reference_commit):
    """Choose exactly five new mechanisms only after the preceding phase has measured results."""
    if len(hypotheses) != 5 or not profiling_evidence or not reference_commit:
        raise ValueError("an adaptive batch needs five hypotheses, profiling evidence and a fixed reference")
    completed = set(investigated_hypotheses(registry))
    if not set(IDEAS_BY_NAME).issubset(completed):
        raise ValueError("all thirty initial mechanisms need completed attempts on both GPU types first")
    chosen_names = set(IDEAS_BY_NAME)
    target = None
    for batch in registry["adaptive_batches"]:
        names = [entry["name"] for entry in batch["hypotheses"]]
        if all(name is None for name in names):
            target = batch
            break
        if not set(names).issubset(completed):
            raise ValueError("the preceding adaptive batch has unfinished measurements")
        chosen_names.update(names)
    if target is None:
        raise ValueError("all twenty adaptive slots are already assigned")
    required = {"name", "hypothesis", "mechanism", "learner", "environment", "execution",
                "correctness_checks", "measurement_settings", "acceptance_criteria"}
    existing_changes = {}
    for entry in INITIAL_IDEAS:
        for trial_index in (None, *range(len(entry.parameter_trials))):
            existing_changes[canonical_signature(**trial_changes(entry, trial_index))] = entry.name
    batch_changes = set()
    updates = []
    for position, hypothesis in zip(target["hypotheses"], hypotheses):
        if not required.issubset(hypothesis) or not hypothesis["name"] or not hypothesis["hypothesis"]:
            raise ValueError("each adaptive hypothesis needs its mechanism, switches and verification specification")
        if hypothesis["name"] in chosen_names:
            raise ValueError("an adaptive hypothesis cannot rename or repeat an earlier mechanism")
        if not any(hypothesis[scope] for scope in ("learner", "environment", "execution")):
            raise ValueError("an adaptive hypothesis needs a concrete distinct implementation")
        signature = canonical_signature(**{scope: hypothesis[scope]
                                            for scope in ("learner", "environment", "execution")})
        if signature in batch_changes:
            raise ValueError("two additional names cannot describe an identical configuration in one batch")
        if signature in existing_changes and not (hypothesis.get("interaction_reason")
                                                   and hypothesis.get("related_prior_hypotheses")):
            raise ValueError("an earlier setting or parameter trial needs an explicitly motivated interaction, "
                             "not another hypothesis name")
        batch_changes.add(signature)
        chosen_names.add(hypothesis["name"])
        updates.append((position, hypothesis))
    for position, hypothesis in updates:
        position.update(hypothesis, state="specified", results={hardware: [] for hardware in REQUIRED_HARDWARE})
    target.update(reference_commit=reference_commit, profiling_evidence=profiling_evidence)
    return registry


def summarize_pairs(rounds, *, correctness_passed, contaminated=False):
    """Select only repeatable matched-copy improvements using an exact one-sided sign check."""
    if not rounds:
        raise ValueError("paired timing results are empty")
    speedups = []
    for pair in rounds:
        reference, candidate = pair["reference"], pair["candidate"]
        if (reference["copies"], reference["iterations"]) != (candidate["copies"], candidate["iterations"]):
            raise ValueError("paired results must use identical copy and update counts")
        if min(reference["seconds"], candidate["seconds"]) <= 0:
            raise ValueError("timed durations must be positive")
        speedups.append(reference["seconds"] / candidate["seconds"])
    wins = sum(value > 1.0 for value in speedups)
    count = len(speedups)
    probability = sum(math.comb(count, successes) for successes in range(wins, count + 1)) / 2 ** count
    median = statistics.median(speedups)
    valid = correctness_passed and not contaminated and count >= 7
    decision = ("accepted" if valid and median > 1.0 and probability <= 0.05
                else "inconclusive" if valid and median > 1.0 else "rejected")
    return {"decision": decision, "measurement_valid": bool(valid),
            "correctness_passed": bool(correctness_passed), "contaminated": bool(contaminated),
            "paired_rounds": count, "candidate_winning_rounds": wins,
            "median_speedup": median, "paired_speedups": speedups,
            "one_sided_sign_probability": probability,
            "selection_rule": "at least seven valid pairs, median improvement, sign probability <= 0.05"}


def matched_hardware_selection(results, copies):
    """Keep H100 and A4500 selections separate and compare only the same policy-copy count."""
    selected = {}
    for hardware in REQUIRED_HARDWARE:
        candidates = [row for row in results if row.get("hardware_class") == hardware
                      and row.get("copies") == copies and row.get("measurement_valid")
                      and row.get("decision") == "accepted"]
        selected[hardware] = (max(candidates, key=lambda row: row["median_candidate_transitions_per_second_total"])
                              if candidates else None)
    return selected


def main():
    """Print the honest implementation catalog or create a new study registry explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-registry", type=Path)
    parser.add_argument("--reference-commit", default="unspecified")
    args = parser.parse_args()
    registry = new_registry(args.reference_commit)
    if args.write_registry:
        if args.reference_commit == "unspecified" or args.write_registry.exists():
            raise ValueError("a new registry needs a reference commit and a previously unused path")
        args.write_registry.write_text(json.dumps(registry, indent=2) + "\n")
    else:
        for entry in registry["initial_ideas"]:
            print(f'{entry["position"]:2d}  {entry["name"]:46}  {entry["implementation"]["state"]}')
        print("20 additional slots remain unchosen until initial measurements and profiling exist")


if __name__ == "__main__":
    main()

"""Point 1: one Optuna trial per hyperparameter cell + intermediate seed reports + a CUSTOM pruner
that implements the user's exact rule: prune a cell once n>=10 seeds and mean + 1.96*sd/sqrt(n) < 35.

Each trial is one of the 12 cells (chosen by GridSampler). Inside the trial we loop seed_index and
draw that seed's reward, keep the running list, report the running mean at step=n, and after 10 seeds
ask trial.should_prune(). The custom pruner reads the running (mean, sd, n) off the FrozenTrial and
returns True when the upper end of the 95% interval has dropped below 35. We run the whole simulation
at 3 base seeds and report the range of total cost and of the surviving set.
"""

import numpy as np
import optuna

import sim

optuna.logging.set_verbosity(optuna.logging.WARNING)


class UpperConfidenceBoundPruner(optuna.pruners.BasePruner):
    """Custom pruner: prune a cell when it has >=min_seeds seeds and its 95% upper bound < threshold."""

    def __init__(self, threshold, min_seeds, z=sim.Z_95):
        # Store the user's rule constants (the absolute bar, the warmup count, the z-multiplier).
        self.threshold = threshold
        self.min_seeds = min_seeds
        self.z = z

    def prune(self, study, trial):
        # The one method BasePruner requires. It gets the live study and a FrozenTrial snapshot;
        # it must return True (Optuna will mark the trial for pruning) or False.
        # Read the running seed count from how many intermediate values have been reported.
        n = len(trial.intermediate_values)
        # Do not judge a cell before it has the warmup number of seeds.
        if n < self.min_seeds:
            return False
        # The running mean is the value reported at the latest step; sd came in via a user attr.
        mean = trial.intermediate_values[max(trial.intermediate_values)]
        sd = trial.user_attrs["running_sd"]
        # Apply the user's rule: prune when the 95% upper confidence bound is below the bar.
        upper = mean + self.z * sd / np.sqrt(n)
        return upper < self.threshold


def make_objective(base_seed, records):
    """Build the objective for one base seed; records[cell] collects the per-cell outcome."""

    def objective(trial):
        # Each trial is one cell; loop seeds, report the running mean, and cooperate with the pruner.
        cell = trial.suggest_int("cell", 0, sim.N_CELLS - 1)
        rewards = []
        for seed_index in range(sim.MAX_SEEDS):
            # Draw this seed's fixed reward and update the running statistics.
            rewards.append(sim.draw_reward(base_seed, cell, seed_index))
            mean, sd, upper = sim.upper_ci(rewards)
            n = len(rewards)
            # Expose sd to the pruner via a user attr and report the running mean as the step value.
            trial.set_user_attr("running_sd", sd)
            trial.report(mean, step=n)
            # After the warmup count, let the pruner decide; cooperate by raising TrialPruned.
            if n >= sim.MIN_SEEDS and trial.should_prune():
                records[cell] = {"cell": cell, "seeds": n, "mean": mean, "sd": sd,
                                 "upper": upper, "status": "PRUNED"}
                raise optuna.TrialPruned()
        # Cell survived to the 50-seed cap: record it as completed and return the final mean.
        records[cell] = {"cell": cell, "seeds": len(rewards), "mean": mean, "sd": sd,
                         "upper": upper, "status": "COMPLETED"}
        return mean

    return objective


def run_one(base_seed):
    """Run all 12 cells once under the custom pruner; return the per-cell records dict."""
    # GridSampler over the single 'cell' parameter guarantees every one of the 12 cells runs once.
    records = {}
    search_space = {"cell": list(range(sim.N_CELLS))}
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space, seed=base_seed),
        pruner=UpperConfidenceBoundPruner(threshold=sim.THRESHOLD, min_seeds=sim.MIN_SEEDS),
    )
    study.optimize(make_objective(base_seed, records), n_trials=sim.N_CELLS)
    return records, study


def main():
    # Header describing the rule and the reference full-grid cost.
    full_grid_cost = sim.N_CELLS * sim.MAX_SEEDS
    print(f"Custom pruner rule: prune when n>=10 and mean + 1.96*sd/sqrt(n) < {sim.THRESHOLD}")
    print(f"True survivors (target mean>=35): cells {sim.true_survivors()}")
    print(f"Full grid cost (12 cells x 50 seeds) = {full_grid_cost} seed-runs\n")

    totals = []
    survivor_sets = []
    for base_seed in (0, 1, 2):
        records, study = run_one(base_seed)
        # Per-cell table for this base seed.
        print(f"===== base_seed {base_seed} =====")
        print(f"{'cell':>4} {'true_mean':>9} {'seeds':>5} {'mean':>7} {'upper95':>8} {'status':>10}")
        total = 0
        survivors = []
        for cell in range(sim.N_CELLS):
            r = records[cell]
            total += r["seeds"]
            if r["status"] == "COMPLETED":
                survivors.append(cell)
            print(f"{cell:>4} {sim.TRUE_MEANS[cell]:>9.1f} {r['seeds']:>5} {r['mean']:>7.2f} "
                  f"{r['upper']:>8.2f} {r['status']:>10}")
        # Compare the surviving set against the true survivors and list any disagreement.
        true_set = set(sim.true_survivors())
        wrong_keep = sorted(set(survivors) - true_set)
        wrong_prune = sorted(true_set - set(survivors))
        print(f"total seed-runs used: {total}  (full grid would use {full_grid_cost})")
        print(f"surviving cells: {survivors}")
        print(f"wrong keep (kept but true mean<35): {wrong_keep}")
        print(f"wrong prune (pruned but true mean>=35): {wrong_prune}\n")
        totals.append(total)
        survivor_sets.append(tuple(survivors))

    # Range summary across the 3 base seeds.
    print("===== summary across base seeds 0,1,2 =====")
    print(f"total seed-runs used: min {min(totals)}  max {max(totals)}  values {totals}")
    print(f"full grid cost: {full_grid_cost}")
    print(f"surviving sets: {survivor_sets}")


if __name__ == "__main__":
    main()

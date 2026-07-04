"""Point 4: SuccessiveHalvingPruner and HyperbandPruner with resource = number of seeds.

SuccessiveHalvingPruner runs rungs at resource = min_resource * reduction_factor**rung. At each rung
it promotes only the top 1/reduction_factor fraction of the trials that reached that rung and prunes
the rest. We use resource = seed count: report the running mean at each seed and cooperate by raising
TrialPruned when should_prune() is True. We report seeds used per cell and which cells reach 50 seeds.
HyperbandPruner is constructed and its key arguments printed; it wraps successive halving in several
brackets that start at different min_resource values.
"""

import optuna

import sim

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Successive halving parameters for the demo: first rung at 10 seeds, keep the top half each rung.
SHA_MIN_RESOURCE = 10
SHA_REDUCTION_FACTOR = 2


def objective(trial):
    # One cell per trial; report the running mean at each seed and let successive halving decide.
    cell = trial.suggest_int("cell", 0, sim.N_CELLS - 1)
    rewards = []
    for seed_index in range(sim.MAX_SEEDS):
        # Draw this seed's reward, report the running mean as the value at this resource level.
        rewards.append(sim.draw_reward(0, cell, seed_index))
        mean = sum(rewards) / len(rewards)
        trial.report(mean, step=len(rewards))
        # Cooperate: at a rung where this cell is not in the promoted top fraction, stop it.
        if trial.should_prune():
            raise optuna.TrialPruned()
    return mean


def run_pruner(pruner):
    # Run the 12 cells under one pruner; enqueue in order so cell 0 (the strong cell) runs first.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=pruner,
    )
    for cell in range(sim.N_CELLS):
        study.enqueue_trial({"cell": cell})
    study.optimize(objective, n_trials=sim.N_CELLS)
    return study


def report(study):
    # Print the per-cell table and return (total seeds, list of cells that reached the 50-seed cap).
    print(f"{'cell':>4} {'true_mean':>9} {'seeds':>5} {'last_mean':>9} {'reached_50':>10} {'state':>18}")
    total = 0
    reached_50 = []
    for t in sorted(study.trials, key=lambda t: t.params["cell"]):
        cell = t.params["cell"]
        seeds = len(t.intermediate_values)
        last_mean = t.intermediate_values[max(t.intermediate_values)]
        got_50 = seeds >= sim.MAX_SEEDS
        if got_50:
            reached_50.append(cell)
        total += seeds
        print(f"{cell:>4} {sim.TRUE_MEANS[cell]:>9.1f} {seeds:>5} {last_mean:>9.2f} "
              f"{str(got_50):>10} {str(t.state):>18}")
    return total, reached_50


def main():
    # Print the key constructor arguments of both pruners (the resource schedule knobs).
    sha = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=SHA_MIN_RESOURCE, reduction_factor=SHA_REDUCTION_FACTOR)
    hb = optuna.pruners.HyperbandPruner(
        min_resource=SHA_MIN_RESOURCE, max_resource=sim.MAX_SEEDS,
        reduction_factor=SHA_REDUCTION_FACTOR)
    print("SuccessiveHalvingPruner key args: "
          f"min_resource={SHA_MIN_RESOURCE}, reduction_factor={SHA_REDUCTION_FACTOR}, "
          "min_early_stopping_rate=0, bootstrap_count=0")
    print("  rungs (resource=seed count): "
          f"{[SHA_MIN_RESOURCE * SHA_REDUCTION_FACTOR**k for k in range(3)]} ... "
          "top 1/2 promoted at each rung")
    print("HyperbandPruner key args: "
          f"min_resource={SHA_MIN_RESOURCE}, max_resource={sim.MAX_SEEDS}, "
          f"reduction_factor={SHA_REDUCTION_FACTOR}, bootstrap_count=0")
    # The inner successive-halving brackets are built lazily, only once the first trial runs.
    print(f"  Hyperband inner brackets before any trial runs: {len(hb._pruners)} (built lazily)\n")

    # Successive halving demonstration.
    print("----- SuccessiveHalvingPruner -----")
    sha_study = run_pruner(sha)
    sha_total, sha_reached = report(sha_study)
    print(f"total seed-runs used: {sha_total}  (full grid would use {sim.N_CELLS * sim.MAX_SEEDS})")
    print(f"cells reaching 50 seeds: {sha_reached}\n")

    # Hyperband demonstration on the same 12 cells.
    print("----- HyperbandPruner -----")
    hb_study = run_pruner(hb)
    hb_total, hb_reached = report(hb_study)
    print(f"total seed-runs used: {hb_total}  (full grid would use {sim.N_CELLS * sim.MAX_SEEDS})")
    print(f"cells reaching 50 seeds: {hb_reached}")
    print(f"Hyperband inner brackets after the study ran: {len(hb._pruners)}")
    print("Hyperband adds over successive halving: it runs several successive-halving brackets, each")
    print("starting at a different first-rung budget, hedging against picking one wrong min_resource.")


if __name__ == "__main__":
    main()

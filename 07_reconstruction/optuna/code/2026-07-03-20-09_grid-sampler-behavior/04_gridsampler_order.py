"""Verify GridSampler visit order: deterministic across fresh studies with the same seed?
Cartesian-product order or shuffled? Also compare two different seeds."""
import optuna

# small 2 x 3 x 2 = 12 combination space so the full order prints compactly
SEARCH_SPACE = {
    "normalization": ["none", "unit"],
    "ridge": [1e-6, 1e-4, 1e-2],
    "clip": ["inf", "5"],
}


def run_order(seed):
    """Run the whole grid once with the given sampler seed; return the visit-order list of tuples."""
    # fresh study + seeded GridSampler
    order = []
    sampler = optuna.samplers.GridSampler(SEARCH_SPACE, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # objective records the exact tuple in the order trials run
    def objective(trial):
        # suggest each axis and append the visited tuple
        t = (
            trial.suggest_categorical("normalization", SEARCH_SPACE["normalization"]),
            trial.suggest_categorical("ridge", SEARCH_SPACE["ridge"]),
            trial.suggest_categorical("clip", SEARCH_SPACE["clip"]),
        )
        order.append(t)
        return 0.0

    study.optimize(objective, n_trials=None)
    return order


def cartesian_order():
    """Return the plain nested-loop Cartesian-product order for reference comparison."""
    # nested loops in dict-key order: normalization outer, ridge middle, clip inner
    out = []
    for n in SEARCH_SPACE["normalization"]:
        for r in SEARCH_SPACE["ridge"]:
            for c in SEARCH_SPACE["clip"]:
                out.append((n, r, c))
    return out


def main():
    """Compare two fresh studies at the same seed, and the order vs Cartesian product."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("optuna version:", optuna.__version__)

    # two independent runs with the SAME seed
    order_a = run_order(seed=42)
    order_b = run_order(seed=42)
    print("\nseed=42 run A order:")
    for i, t in enumerate(order_a):
        print(f"  {i:2d} {t}")
    print("seed=42 run B order equals run A order:", order_a == order_b)

    # is the seed=42 order the plain Cartesian-product order?
    cart = cartesian_order()
    print("\nCartesian-product order:")
    for i, t in enumerate(cart):
        print(f"  {i:2d} {t}")
    print("seed=42 order equals Cartesian-product order:", order_a == cart)

    # a DIFFERENT seed for contrast
    order_c = run_order(seed=99)
    print("\nseed=99 run order:")
    for i, t in enumerate(order_c):
        print(f"  {i:2d} {t}")
    print("seed=99 order equals seed=42 order:", order_c == order_a)
    print("seed=99 order equals Cartesian-product order:", order_c == cart)


if __name__ == "__main__":
    main()

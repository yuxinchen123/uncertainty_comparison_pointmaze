## Purpose

Retrain the 11 unique parameter settings shown in historical Figures 1.2 and 1.3 after correcting the intrinsic value used at episode resets. Keep the historical 1000-million-step results and add a corrected 100-million-step section.

## Key hyperparameters

128 copies per configuration, paired seed indices 0 through 127; full-batch proximal policy optimization, 128 rollout steps, four environments per copy. The exact per-algorithm pairs are in parameter_settings.json. Each copy trains for 195325 iterations (100006400 transitions). Recording windows have 200 iterations, except the final 125-iteration window; both span complete episode-clock cycles. No checkpoints.

## Code and config changes

Intrinsic returns continue across resets and bootstrap from the actual reset observation. Immediate intrinsic rewards still score the final transition observation. Task-return bootstrap and task advantage boundaries retain their prior definitions. Predictor hidden widths are 256 and 128, with 128 outputs. Select execution options from ten bounded serval05 measurements, then test numerical agreement. Generic UVA jobs claim units from one shared queue; no unit is assigned to an account in advance.

## Git state

Commit will be recorded before submission. Existing changes in 08_cleanrl_ppo_rnd and RLeXplore belong to other sessions.

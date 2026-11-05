## Purpose

Run one reviewed metadata controller for one-copy common pixel PPO plus Random Network Distillation timing on the approved serval02 card. Scientific results remain in the existing data unit.

## Key hyperparameters

Seven alternating pairs of ten complete updates per arm; 128 environments, 128 rollout steps, four epochs, four minibatches and fifty normalization warmups. Seed1 and the original two-billion-transition annealing horizon remain unchanged. Networks and Adam use float32; statistics use float64.

## Code and config changes

Only room background caching and final-two-frame rendering differ between the scientific arms. The metadata starter preserves the reviewed process, metadata barrier and custody functions; only its actual inner publication constant changes. The controller owns no standing GPU descriptor.

## Git state

Reviewed controller source:765480d990cff5ae556914e3b74afba0472f64ea. Clean measurement helper source:69f1b5075530418adbe02c1e433c5647848a61ba. Original numerical source:d28e6735e79788f618acb0ebfde8609db1f776af. Actual inner publication:61b57745832bb640ff5c66019edf0a4aca34ced4. The integration checkout contains these newly prepared transport records pending review and publication. No remote action occurred during preparation.

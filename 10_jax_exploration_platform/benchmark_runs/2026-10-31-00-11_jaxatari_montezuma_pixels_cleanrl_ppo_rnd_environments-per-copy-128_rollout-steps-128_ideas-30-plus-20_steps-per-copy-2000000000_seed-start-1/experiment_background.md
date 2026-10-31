# Pixel Montezuma infrastructure study

Hand-written launch record. Created October 31, 2026, Pacific Time.

## 1. Purpose

Merge the missing RND environment branches, reproduce original CleanRL proximal policy optimization with Random Network Distillation using JAXtari pixel observations, measure thirty specified infrastructure ideas and twenty adaptive ideas on H100 and RTX A4500, select independent-policy packing, compare one with eight jaguar03 GPUs, and train the selected implementation for two billion steps per policy copy.

## 2. Key hyperparameters

Each independent policy uses 128 environments, 128 rollout steps, four minibatches, four epochs, learning rate 0.0001 with linear annealing, extrinsic discount 0.999, intrinsic discount 0.99, generalized advantage estimation parameter 0.95, policy clip 0.1, entropy coefficient 0.001, value coefficient 0.5, joint gradient norm limit 0.5, predictor update fraction 0.25, intrinsic advantage weight 1, extrinsic advantage weight 2, and fifty random warmup rollouts. Networks use full float32 calculations; running statistics retain original float64 precision.

The long-run budget is 122,070 full updates, or 1,999,994,880 training transitions per policy copy, with warmup additional. Copy count is measured per GPU type; stop when doubling copies adds less than ten percent aggregate throughput after confirmation. No hyperparameter search or extra seed repetitions is planned.

## 3. Code and config changes

The source reference is CleanRL commit fe8d8a03c41a7ef5b523e2e354bd01c363e786bb. The new game source is JAXtari commit 04ab9aec318c72162e2e53503a594a489d7e162b. Native immediate episode resets are an approved game-interface difference. PointMaze configurations are explicit and do not set the Atari algorithm.

The first thirty ideas are listed in the saved plan. The next twenty are chosen in four batches of five from measurements and profiling after the initial thirty. Only verified, measured changes enter the final implementation.

For this study the user authorizes direct serval02 card 1 use even while their own work occupies serval05. Card 0 remains excluded, card availability and mode checks remain required, and the appropriate shared lock is mandatory. The earlier duration restrictions are waived for this study; checkpoints are required for interruption recovery. Other users' work is never interrupted.

Latest placement instruction: short H100 measurements may use serval02 or serval05; the final long H100 runs may use ACES, NCSA DeltaAI, or MIT. Existing ACES and DeltaAI connections on UVA portal05 are authorized for use. Choose final placement from measured run duration, available capacity and allocation balance. Rebuild environments for DeltaAI's ARM processors. Do not repeatedly attempt unavailable connections.

## 4. Git state

Initial current branch: Use-RLexplore-RND at 0c93d43132d288e240a36a09a50618248d6bd808.
Incoming branch: fused-atari at adf6861a7406c6235942d52f9276ad42aa8bcef6; includes fused-antmaze.
Integration branch: merge-atari-cleanrl.
First pixel baseline code commit: d5d63d9. Each submitted unit records its full execution commit in the submission ledger and reads an immutable archive of that commit.

Original-checkout modified historical logs, notebook edits in RLeXplore, and an untracked completion sentinel are preserved separately. No numerical work is run on login nodes.

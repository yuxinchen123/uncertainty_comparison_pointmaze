# Earlier training steps

Verified from the original records on 2026-10-31.

- Earlier pixel Atari: 128 environments per policy, 1,999,994,880 trained transitions per completed policy, or **15,624,960 trained transitions per environment**. Observation normalization uses another **6,400 transitions per environment**, outside the recorded training step counter. Of 150 raw policy records, 115 are complete and 35 are incomplete.
- Earlier object-based JAX: one environment per policy and 1,024 independent policies per setting. Both the RND and no-bonus settings completed **2,048,000 trained transitions per environment**. The RND setting has another **1,280 normalization transitions per environment**; the no-bonus setting has no normalization warmup.

The pixel evidence is the original run's `configs.jsonl` and raw `data/2026-08-05-22-30_ablation/local/*_of_150.json` records under:

```text
/p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-22-30_cleanrl_run_2_ablation_ppo-rnd_montezuma-v5__arm1-orig_arm2-noclip_arm3-prop1_arm4-shallow_arm5-all__2e9step_seed1-30__log200_gradstats_ckpt1h__gpu-gnolim-resv
```

The warmup exclusion is implemented in `08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py`. The older run's `analysis/run_metrics.json` is stale; the counts above use the original raw records.

The object evidence is `config_resolved.yaml`, `summary.json`, and the final `unit_complete` records in both `data/unit-rnd_next_state.jsonl` and `data/unit-none.jsonl` under:

```text
/p/rlprojects/RND_worktrees/fused-atari/10_jax_exploration_platform/runs/2026-08-18-04-43_jax-ppo_montezuma-object-centric-sticky_full-batch_rollout-128_envs-per-copy-1_steps-per-copy-2M_bonus-rnd-next-state-none_learning-rate-3e-4_intrinsic-weight-1_copies-1024_first-montezuma-batch
```

The object warmup behavior is defined in `training/runner.py` and `training/train_step.py` in that original worktree. This verification did not resume or change either earlier study.

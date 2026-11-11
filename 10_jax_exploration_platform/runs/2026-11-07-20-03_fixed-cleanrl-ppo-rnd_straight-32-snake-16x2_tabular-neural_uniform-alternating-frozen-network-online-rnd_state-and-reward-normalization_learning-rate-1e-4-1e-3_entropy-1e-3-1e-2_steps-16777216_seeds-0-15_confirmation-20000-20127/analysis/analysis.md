# Fixed CleanRL PPO-RND settings on the straight and snake grids

Generated 2026-11-11 15:12 PT (times are Pacific, converted from machines running Eastern). Phase: complete. Finished settings: 4768.

## Contents

| Section | First written | Last changed | Status |
|---|---|---|---|
| [Metrics](#metrics) | 2026-11-07 21:49 PT | 2026-11-09 15:29 PT | unread |
| [Settings](#settings) | 2026-11-07 21:49 PT | 2026-11-09 19:56 PT | unread |
| [Confirmation, chosen configuration of each condition](#confirmation-chosen-configuration-of-each-condition) | 2026-11-07 21:49 PT | 2026-11-11 15:09 PT | unread |
| [Confirmation, learning rate 1e-4, entropy 0.001, best intrinsic weight](#confirmation-learning-rate-1e-4-entropy-0-001-best-intrinsic-weight) | 2026-11-07 21:49 PT | 2026-11-11 15:09 PT | unread |
| [Trimmed means: how many seeds to remove, and whether the results change](#trimmed-means-how-many-seeds-to-remove-and-whether-the-results-change) | 2026-11-09 15:29 PT | 2026-11-11 11:23 PT | unread |
| [Development](#development) | 2026-11-07 21:49 PT | 2026-11-11 15:13 PT | unread |
| [Decay: online RND (coordinate- and tabular-RND) against the count rule](#decay-online-rnd-coordinate-and-tabular-rnd-against-the-count-rule) | 2026-11-09 19:56 PT | 2026-11-11 14:59 PT | unread |
| [Records](#records) | 2026-11-07 21:49 PT | 2026-11-11 15:13 PT | unread |

[Read with section colors](analysis.html). No section marks itself read.

## Metrics

Define $K$ as the number of policies of a configuration, $T_i$ as the training transition at which policy $i$ first arrives at the goal cell, and $B = 16{,}777{,}216$ as the budget. Configurations are chosen by the mean capped first-goal time $M(B) = \dfrac{1}{K} \sum_{i=1}^{K} \min(T_i, B)$, with ties broken by the goal discovery rate $F(B) = \dfrac{1}{K} \sum_{i=1}^{K} \mathbf{1}[T_i \le B]$. Coverage is the fraction of path cells reached in training by $B$. The final-policy goal rate is the fraction of 32 evaluation episodes (actions sampled from the final policy, no learning) that reach the goal; the greedy column uses one episode with the most likely action. Intervals are 95% bootstrap intervals and standard errors are bootstrap standard errors (2,000 resamples); network conditions resample field seeds (four policies share a field), the others resample policy seeds. The trimmed mean drops the 10 smallest and 10 largest capped first-goal times of 128 seeds (1 of 16 in development) before averaging; it is defined, and the number removed is discussed, in the section on trimmed means. Step counts are written in thousands (8.11K = 8,110) when every value in a table is at least 1,000.

## Settings

All conditions use CleanRL PPO-RND's scalar settings: rollout 128, 4 epochs, 4 minibatches, extrinsic discount 0.999, intrinsic discount 0.99, GAE 0.95, clip 0.1, value coefficient 0.5, maximum gradient norm 0.5, Adam epsilon 1e-5, extrinsic advantage weight 2, linear learning-rate annealing. Swept per condition: learning rate {1e-4, 1e-3} × entropy {0.001, 0.01} × intrinsic weight 2^-6 … 2^6 (13 values). Count-based fields are scaled to average 1 over cells and multiplied by $N^{-a}$, where $N$ is the cell's visits and $a$ the decay exponent: $a = 0.5$ (division by √visits) except the uniform rows that name another exponent. Online RND uses the raw CleanRL prediction error (half the summed squared error of 512 features). Coordinate-RND networks read the 2-d cell coordinates; tabular-RND networks read a one-hot cell vector, with the same deeper layers. State normalization whitens the coordinate-RND inputs with running statistics, started by 50 random-action rollouts of 128 steps (it does not apply to one-hot inputs); reward normalization divides the bonus by the running standard deviation of its discounted sum. Tabular-PPO keeps a table of 4 action logits and 2 values per cell; coordinate-PPO is the neural policy, which reads the coordinates. Development: 16 seeds per configuration; confirmation: 128 fresh seeds (20000-20127), and 1,024 (20000-21023) for online tabular-RND with reward normalization on, the auto research's baseline. The uniform fields with decay exponents 2, 1 and 0.25 and the tabular-RND rows come from the extension run `runs/2026-11-09-19-40_fixed-cleanrl-ppo-rnd-extension_…`, on the same seeds and settings.

## Confirmation, chosen configuration of each condition

### straight grid, 32 cells, tabular-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -2, reward normalization on | **5.49K** ± 0.23K [5.02K, 5.94K] | **5.42K** ± 0.24K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 128 |
| uniform field, decay exponent -2, reward normalization off | <u>5.71K</u> ± 0.25K [5.22K, 6.19K] | <u>5.68K</u> ± 0.27K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -1, reward normalization on | 5.81K ± 0.25K [5.33K, 6.30K] | 5.73K ± 0.27K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^0 | 128 |
| uniform field, decay exponent -1, reward normalization off | 6.17K ± 0.29K [5.63K, 6.74K] | 5.98K ± 0.29K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| online tabular-RND, reward normalization on | 7.55K ± 0.13K [7.30K, 7.79K] | 7.46K ± 0.14K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-3 | 1024 |
| online tabular-RND, reward normalization off | 7.66K ± 0.35K [7.01K, 8.35K] | 7.63K ± 0.38K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 8.11K ± 0.45K [7.26K, 9.01K] | 7.74K ± 0.46K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | 8.59K ± 0.61K [7.46K, 9.91K] | 7.79K ± 0.44K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 11.59K ± 0.80K [9.99K, 13.19K] | 10.91K ± 0.83K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| frozen tabular-RND field, reward normalization on | 12.98K ± 1.90K [9.72K, 17.16K] | 9.29K ± 0.96K | **100.0%** | **100.0%** | **100.0%** | 98.4% | learning rate 0.001, entropy 0.001, weight 2^0 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 13.86K ± 0.93K [11.94K, 15.61K] | 13.10K ± 1.02K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-3 | 128 |
| frozen tabular-RND field, reward normalization off | 14.29K ± 1.52K [11.33K, 17.29K] | 9.74K ± 1.19K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization on, reward normalization on | 15.67K ± 0.92K [13.74K, 17.35K] | 14.86K ± 1.05K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^4 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 16.16K ± 1.05K [13.98K, 18.18K] | 15.09K ± 1.08K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-5 | 128 |
| alternating high/low field (64:1), reward normalization on | 17.00K ± 1.26K [14.60K, 19.51K] | 15.24K ± 1.22K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| alternating high/low field (64:1), reward normalization off | 17.08K ± 1.27K [14.66K, 19.53K] | 15.34K ± 1.23K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | 19.44K ± 1.57K [16.44K, 22.59K] | 17.11K ± 1.64K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.01, weight 2^-3 | 128 |
| no bonus | 21.47K ± 1.74K [18.23K, 25.09K] | 19.11K ± 1.78K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| uniform field, decay exponent -0.25, reward normalization off | 25.37K ± 8.33K [10.82K, 44.23K] | 10.35K ± 0.70K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 34.93K ± 9.05K [19.93K, 54.15K] | 18.56K ± 2.76K | **100.0%** | **100.0%** | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 61.45K ± 12.32K [39.51K, 87.44K] | 36.89K ± 7.44K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-3 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 425.87K ± 141.66K [184.34K, 731.43K] | 93.68K ± 38.78K | <u>99.2%</u> | <u>100.0%</u> | 93.9% | 93.0% | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 878.40K ± 316.33K [307.03K, 1,557.81K] | 75.83K ± 106.49K | 96.9% | 99.9% | 18.0% | 17.2% | learning rate 0.0001, entropy 0.01, weight 2^-4 | 128 |

### straight grid, 32 cells, coordinate-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -1, reward normalization on | **2.41K** ± 0.18K [2.07K, 2.79K] | **2.15K** ± 0.18K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | <u>2.98K</u> ± 0.18K [2.63K, 3.33K] | 2.79K ± 0.19K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^4 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 3.14K ± 0.19K [2.77K, 3.52K] | 2.98K ± 0.21K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 128 |
| frozen tabular-RND field, reward normalization off | 3.41K ± 0.28K [2.87K, 3.96K] | 3.17K ± 0.29K | **100.0%** | **100.0%** | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.001, entropy 0.001, weight 2^-3 | 128 |
| uniform field, decay exponent -2, reward normalization off | 3.45K ± 0.21K [3.05K, 3.87K] | 3.36K ± 0.24K | **100.0%** | **100.0%** | 98.4% | 98.4% | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| frozen tabular-RND field, reward normalization on | 3.52K ± 0.22K [3.07K, 3.95K] | 3.23K ± 0.24K | **100.0%** | **100.0%** | 96.1% | 96.1% | learning rate 0.001, entropy 0.001, weight 2^0 | 128 |
| online tabular-RND, reward normalization on | 4.59K ± 0.05K [4.49K, 4.70K] | 4.57K ± 0.04K | **100.0%** | **100.0%** | 97.2% | 97.2% | learning rate 0.001, entropy 0.01, weight 2^-4 | 1024 |
| uniform field, decay exponent -0.25, reward normalization off | 4.87K ± 0.30K [4.31K, 5.48K] | 4.57K ± 0.29K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -1, reward normalization off | 5.30K ± 2.37K [2.60K, 10.35K] | <u>2.69K</u> ± 0.21K | **100.0%** | **100.0%** | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | 6.11K ± 0.27K [5.58K, 6.64K] | 6.10K ± 0.30K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization on, reward normalization on | 7.82K ± 0.36K [7.15K, 8.54K] | 7.51K ± 0.32K | **100.0%** | **100.0%** | 89.1% | 89.1% | learning rate 0.001, entropy 0.01, weight 2^5 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 8.01K ± 0.46K [7.11K, 8.93K] | 7.71K ± 0.46K | **100.0%** | **100.0%** | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.001, entropy 0.01, weight 2^-3 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 10.83K ± 0.30K [10.29K, 11.47K] | 10.68K ± 0.24K | **100.0%** | **100.0%** | 94.5% | 94.5% | learning rate 0.001, entropy 0.01, weight 2^4 | 128 |
| online tabular-RND, reward normalization off | 55.84K ± 29.42K [4.93K, 120.54K] | 4.84K ± 0.23K | **100.0%** | **100.0%** | 97.7% | 97.7% | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 161.00K ± 130.19K [16.72K, 436.69K] | 12.05K ± 0.52K | <u>99.2%</u> | <u>99.8%</u> | 98.4% | 98.4% | learning rate 0.001, entropy 0.001, weight 2^-4 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 167.87K ± 127.86K [25.95K, 439.38K] | 23.49K ± 2.78K | <u>99.2%</u> | <u>99.8%</u> | 96.9% | 96.9% | learning rate 0.001, entropy 0.01, weight 2^-3 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 170.59K ± 129.36K [29.44K, 442.83K] | 24.53K ± 4.96K | <u>99.2%</u> | 99.7% | 95.3% | 95.3% | learning rate 0.001, entropy 0.01, weight 2^-5 | 128 |
| uniform field, decay exponent -2, reward normalization on | 180.70K ± 133.43K [24.37K, 465.10K] | 3.32K ± 5.59K | <u>99.2%</u> | 99.8% | 98.4% | 98.4% | learning rate 0.001, entropy 0.01, weight 2^-6 | 128 |
| no bonus | 572.44K ± 230.51K [212.29K, 1,110.38K] | 68.09K ± 40.96K | 97.7% | 99.7% | 96.9% | 96.9% | learning rate 0.001, entropy 0.01, weight 2^0 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 873.77K ± 350.83K [291.92K, 1,642.66K] | 33.71K ± 102.31K | 95.3% | 99.1% | 85.2% | 85.2% | learning rate 0.001, entropy 0.01, weight 2^-5 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 1,767.57K ± 407.83K [1,000.40K, 2,575.93K] | 541.10K ± 431.63K | 89.8% | 98.2% | 59.4% | 47.7% | learning rate 0.001, entropy 0.001, weight 2^1 | 128 |
| alternating high/low field (64:1), reward normalization off | 5,774.27K ± 624.17K [4,588.13K, 6,991.07K] | 5,290.09K ± 739.75K | 72.7% | 90.8% | 9.2% | 7.8% | learning rate 0.001, entropy 0.001, weight 2^5 | 128 |
| alternating high/low field (64:1), reward normalization on | 6,995.25K ± 649.69K [5,710.35K, 8,235.32K] | 6,737.17K ± 770.01K | 64.8% | 87.7% | 32.8% | 32.0% | learning rate 0.001, entropy 0.001, weight 2^0 | 128 |

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -2, reward normalization off | **10.38K** ± 0.37K [9.66K, 11.12K] | <u>10.39K</u> ± 0.38K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 128 |
| uniform field, decay exponent -2, reward normalization on | <u>10.44K</u> ± 0.41K [9.63K, 11.24K] | **10.29K** ± 0.42K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| uniform field, decay exponent -1, reward normalization off | 12.81K ± 0.53K [11.73K, 13.83K] | 12.63K ± 0.56K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| uniform field, decay exponent -1, reward normalization on | 13.77K ± 0.56K [12.69K, 14.84K] | 13.55K ± 0.57K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| online tabular-RND, reward normalization off | 18.20K ± 0.85K [16.57K, 19.90K] | 17.84K ± 0.85K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-1 | 128 |
| online tabular-RND, reward normalization on | 18.25K ± 0.26K [17.74K, 18.76K] | 17.97K ± 0.26K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-3 | 1024 |
| online coordinate-RND, state normalization on, reward normalization on | 20.35K ± 1.16K [18.08K, 22.63K] | 19.80K ± 1.20K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^2 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 21.91K ± 1.47K [19.10K, 24.74K] | 20.76K ± 1.47K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-1 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 31.94K ± 3.11K [26.04K, 38.18K] | 26.81K ± 3.29K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | 33.53K ± 3.27K [27.25K, 40.03K] | 27.60K ± 3.52K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 36.29K ± 2.34K [31.81K, 41.03K] | 35.43K ± 2.63K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-5 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 37.30K ± 1.88K [33.70K, 40.95K] | 36.88K ± 2.06K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^5 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 56.72K ± 4.10K [48.99K, 64.83K] | 54.38K ± 4.09K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-5 | 128 |
| frozen tabular-RND field, reward normalization off | 74.68K ± 5.74K [63.07K, 85.90K] | 71.06K ± 6.52K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-2 | 128 |
| frozen tabular-RND field, reward normalization on | 77.28K ± 5.06K [67.03K, 87.19K] | 75.54K ± 5.86K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 91.42K ± 8.55K [76.04K, 109.18K] | 82.93K ± 7.13K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-4 | 128 |
| no bonus | 119.18K ± 10.06K [99.67K, 139.25K] | 104.65K ± 9.79K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| alternating high/low field (64:1), reward normalization off | 120.15K ± 10.48K [99.26K, 141.02K] | 110.97K ± 11.82K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^0 | 128 |
| uniform field, decay exponent -0.25, reward normalization off | 139.32K ± 39.84K [76.67K, 231.75K] | 73.14K ± 6.78K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.01, weight 2^-3 | 128 |
| alternating high/low field (64:1), reward normalization on | 146.97K ± 14.56K [119.63K, 175.54K] | 126.41K ± 16.07K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | 252.15K ± 38.14K [180.05K, 329.65K] | 165.55K ± 38.74K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 1,261.16K ± 82.91K [1,103.90K, 1,424.77K] | 1,176.51K ± 79.52K | **100.0%** | **100.0%** | 81.1% | <u>0.0%</u> | learning rate 0.001, entropy 0.01, weight 2^6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 2,315.62K ± 315.02K [1,765.75K, 2,993.57K] | 1,662.77K ± 165.41K | <u>96.9%</u> | <u>99.5%</u> | <u>96.1%</u> | <u>0.0%</u> | learning rate 0.001, entropy 0.01, weight 2^2 | 128 |

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -1, reward normalization on | **63.87K** ± 9.07K [47.93K, 82.69K] | <u>41.90K</u> ± 3.78K | **100.0%** | **100.0%** | 82.8% | 82.0% | learning rate 0.001, entropy 0.01, weight 2^4 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | <u>75.49K</u> ± 8.88K [61.61K, 95.64K] | 64.22K ± 4.23K | **100.0%** | **100.0%** | 89.8% | 82.0% | learning rate 0.001, entropy 0.01, weight 2^1 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 78.74K ± 17.35K [49.46K, 117.99K] | 45.07K ± 3.78K | **100.0%** | **100.0%** | 29.3% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| uniform field, decay exponent -0.25, reward normalization off | 82.08K ± 16.98K [58.29K, 119.19K] | 59.60K ± 4.65K | **100.0%** | **100.0%** | 22.6% | 18.0% | learning rate 0.001, entropy 0.001, weight 2^0 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 163.71K ± 57.81K [70.64K, 294.24K] | 51.70K ± 5.22K | **100.0%** | **100.0%** | 36.7% | 35.9% | learning rate 0.001, entropy 0.001, weight 2^-1 | 128 |
| online tabular-RND, reward normalization on | 164.07K ± 31.30K [112.05K, 236.03K] | 91.89K ± 1.63K | <u>99.7%</u> | 100.0% | 50.6% | 50.0% | learning rate 0.0001, entropy 0.01, weight 2^-6 | 1024 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 175.98K ± 130.89K [34.19K, 455.56K] | **35.82K** ± 1.59K | 99.2% | 99.9% | 69.4% | 24.2% | learning rate 0.001, entropy 0.01, weight 2^1 | 128 |
| uniform field, decay exponent -1, reward normalization off | 190.61K ± 36.54K [128.10K, 268.92K] | 103.82K ± 17.62K | **100.0%** | **100.0%** | 44.2% | 42.2% | learning rate 0.001, entropy 0.01, weight 2^1 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 235.87K ± 51.66K [142.81K, 345.05K] | 112.37K ± 11.98K | **100.0%** | **100.0%** | 95.3% | 89.1% | learning rate 0.001, entropy 0.01, weight 2^4 | 128 |
| online coordinate-RND, state normalization on, reward normalization on | 250.56K ± 37.67K [194.94K, 332.20K] | 185.81K ± 10.14K | **100.0%** | **100.0%** | <u>98.4%</u> | <u>97.7%</u> | learning rate 0.0001, entropy 0.01, weight 2^-1 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 396.02K ± 19.89K [357.71K, 436.22K] | 381.09K ± 19.66K | **100.0%** | **100.0%** | **99.2%** | **99.2%** | learning rate 0.0001, entropy 0.01, weight 2^0 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | 436.69K ± 199.58K [106.86K, 860.99K] | 50.85K ± 5.85K | 98.4% | 99.9% | 27.3% | 22.7% | learning rate 0.001, entropy 0.001, weight 2^3 | 128 |
| online tabular-RND, reward normalization off | 488.29K ± 195.82K [139.40K, 923.95K] | 51.28K ± 9.95K | 98.4% | 99.9% | 28.1% | 28.1% | learning rate 0.001, entropy 0.01, weight 2^4 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 643.88K ± 133.74K [450.77K, 954.70K] | 437.14K ± 27.61K | 99.2% | <u>100.0%</u> | 78.9% | 78.9% | learning rate 0.0001, entropy 0.01, weight 2^-3 | 128 |
| frozen tabular-RND field, reward normalization off | 1,176.01K ± 240.64K [713.81K, 1,661.79K] | 515.98K ± 127.43K | 97.7% | 99.7% | 25.9% | 14.1% | learning rate 0.001, entropy 0.01, weight 2^3 | 128 |
| frozen tabular-RND field, reward normalization on | 2,183.62K ± 385.14K [1,475.11K, 2,968.80K] | 1,126.39K ± 332.13K | 93.8% | 99.0% | 7.0% | 1.6% | learning rate 0.001, entropy 0.01, weight 2^6 | 128 |
| no bonus | 8,805.72K ± 730.18K [7,441.39K, 10,258.24K] | 8,882.43K ± 865.34K | 49.2% | 93.4% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.01, weight 2^0 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 11,442.45K ± 548.84K [10,349.01K, 12,481.37K] | 12,003.34K ± 643.48K | 45.3% | 86.3% | 13.3% | 10.2% | learning rate 0.001, entropy 0.01, weight 2^-4 | 128 |
| uniform field, decay exponent -2, reward normalization on | 11,483.30K ± 633.53K [10,214.06K, 12,642.79K] | 12,055.48K ± 750.22K | 36.7% | 87.5% | 3.1% | 3.1% | learning rate 0.001, entropy 0.01, weight 2^1 | 128 |
| uniform field, decay exponent -2, reward normalization off | 12,420.71K ± 593.43K [11,229.81K, 13,589.27K] | 13,164.73K ± 699.88K | 30.5% | 85.2% | 0.8% | 0.8% | learning rate 0.001, entropy 0.01, weight 2^-1 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 13,164.14K ± 545.19K [12,020.73K, 14,221.75K] | 13,945.73K ± 623.97K | 31.2% | 80.2% | 12.5% | 12.5% | learning rate 0.001, entropy 0.01, weight 2^-2 | 128 |
| alternating high/low field (64:1), reward normalization on | 13,657.69K ± 548.50K [12,579.69K, 14,698.54K] | 14,632.41K ± 649.31K | 21.1% | 75.1% | 0.0% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^2 | 128 |
| alternating high/low field (64:1), reward normalization off | 14,202.12K ± 503.10K [13,204.11K, 15,196.27K] | 15,276.94K ± 583.64K | 19.5% | 73.1% | 0.0% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^4 | 128 |

## Confirmation, learning rate 1e-4, entropy 0.001, best intrinsic weight

### straight grid, 32 cells, tabular-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -2, reward normalization on | **13.67K** ± 0.86K [12.02K, 15.42K] | **12.93K** ± 0.92K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -2, reward normalization off | <u>14.02K</u> ± 0.88K [12.37K, 15.78K] | <u>13.28K</u> ± 0.94K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -1, reward normalization off | 15.37K ± 1.06K [13.36K, 17.51K] | 14.30K ± 1.12K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -1, reward normalization on | 15.74K ± 1.10K [13.55K, 17.91K] | 14.54K ± 1.14K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen tabular-RND field, reward normalization on | 16.83K ± 1.35K [14.27K, 19.59K] | 15.32K ± 1.33K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online tabular-RND, reward normalization on | 16.94K ± 1.29K [14.42K, 19.42K] | 15.75K ± 1.37K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | 16.94K ± 1.20K [14.62K, 19.29K] | 15.70K ± 1.30K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 16.99K ± 1.23K [14.62K, 19.38K] | 15.60K ± 1.30K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| alternating high/low field (64:1), reward normalization on | 17.00K ± 1.26K [14.60K, 19.51K] | 15.24K ± 1.22K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| alternating high/low field (64:1), reward normalization off | 17.08K ± 1.27K [14.66K, 19.53K] | 15.34K ± 1.23K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen tabular-RND field, reward normalization off | 17.23K ± 1.36K [14.55K, 19.91K] | 15.30K ± 1.38K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online tabular-RND, reward normalization off | 17.24K ± 1.27K [14.81K, 19.71K] | 16.07K ± 1.35K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 18.60K ± 1.45K [15.75K, 21.35K] | 17.18K ± 1.54K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 18.84K ± 1.44K [16.02K, 21.62K] | 17.55K ± 1.56K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.25, reward normalization off | 18.96K ± 1.50K [16.08K, 21.96K] | 16.83K ± 1.48K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | 19.00K ± 1.53K [16.06K, 22.03K] | 16.75K ± 1.54K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| no bonus | 21.47K ± 1.74K [18.23K, 25.09K] | 19.11K ± 1.78K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 21.87K ± 2.35K [17.35K, 26.59K] | 18.24K ± 2.20K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization on, reward normalization on | 22.64K ± 2.26K [18.42K, 27.05K] | 20.09K ± 2.50K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-2 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 34.93K ± 9.05K [19.93K, 54.15K] | 18.56K ± 2.76K | **100.0%** | **100.0%** | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 39.90K ± 9.32K [24.31K, 60.16K] | 20.62K ± 3.56K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 425.87K ± 141.66K [184.34K, 731.43K] | 93.68K ± 38.78K | <u>99.2%</u> | <u>100.0%</u> | 93.9% | 93.0% | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 802.95K ± 264.74K [328.81K, 1,362.28K] | 69.81K ± 38.61K | 96.1% | 99.8% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^-1 | 128 |

### straight grid, 32 cells, coordinate-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -1, reward normalization off | **14.54K** ± 0.78K [13.04K, 16.05K] | 14.42K ± 0.87K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^1 | 128 |
| uniform field, decay exponent -1, reward normalization on | <u>14.88K</u> ± 0.75K [13.41K, 16.30K] | 14.82K ± 0.82K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| uniform field, decay exponent -2, reward normalization on | 15.82K ± 2.91K [12.31K, 22.37K] | <u>13.49K</u> ± 0.60K | **100.0%** | **100.0%** | 98.4% | 98.4% | learning rate 0.0001, entropy 0.001, weight 2^6 | 128 |
| online tabular-RND, reward normalization on | 16.06K ± 0.74K [14.54K, 17.48K] | 16.74K ± 0.85K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-4 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | 21.74K ± 1.06K [19.63K, 23.78K] | 22.20K ± 1.20K | **100.0%** | **100.0%** | 66.4% | 66.4% | learning rate 0.0001, entropy 0.001, weight 2^6 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 21.88K ± 1.05K [19.82K, 23.83K] | 22.35K ± 1.19K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online tabular-RND, reward normalization off | 23.06K ± 1.33K [20.42K, 25.67K] | 23.72K ± 1.54K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 25.07K ± 0.86K [23.35K, 26.68K] | 26.13K ± 0.98K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-5 | 128 |
| online coordinate-RND, state normalization on, reward normalization on | 26.73K ± 0.88K [25.01K, 28.44K] | 28.07K ± 1.01K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^3 | 128 |
| uniform field, decay exponent -2, reward normalization off | 30.01K ± 16.93K [11.84K, 65.39K] | **12.66K** ± 0.53K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^3 | 128 |
| frozen tabular-RND field, reward normalization off | 31.34K ± 8.36K [19.78K, 49.47K] | 21.16K ± 1.52K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-1 | 128 |
| uniform field, decay exponent -0.25, reward normalization off | 32.15K ± 1.62K [28.84K, 35.13K] | 32.95K ± 1.86K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-3 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | 35.42K ± 1.68K [32.08K, 38.52K] | 36.62K ± 1.95K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-5 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 42.94K ± 1.66K [39.62K, 46.12K] | 44.76K ± 1.67K | **100.0%** | **100.0%** | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen tabular-RND field, reward normalization on | 45.31K ± 19.51K [21.17K, 87.59K] | 23.04K ± 3.86K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-1 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 46.72K ± 1.53K [43.54K, 49.46K] | 49.26K ± 1.72K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-4 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 134.84K ± 6.01K [123.15K, 145.72K] | 136.87K ± 6.32K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 227.66K ± 129.59K [88.69K, 494.34K] | 101.42K ± 5.75K | <u>99.2%</u> | <u>99.7%</u> | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.0001, entropy 0.001, weight 2^-4 | 128 |
| no bonus | 2,426.77K ± 509.04K [1,471.97K, 3,456.74K] | 1,322.65K ± 597.69K | 86.7% | 97.9% | 86.7% | 86.7% | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 5,011.25K ± 625.61K [3,707.11K, 6,187.97K] | 4,385.72K ± 741.46K | 70.3% | 96.1% | 25.8% | 25.8% | learning rate 0.0001, entropy 0.001, weight 2^-3 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 5,536.05K ± 852.45K [3,966.06K, 7,234.06K] | 5,007.70K ± 1,010.31K | 67.2% | 95.2% | 35.7% | 13.3% | learning rate 0.0001, entropy 0.001, weight 2^5 | 128 |
| alternating high/low field (64:1), reward normalization off | 11,868.63K ± 638.12K [10,644.69K, 13,096.74K] | 12,513.00K ± 756.26K | 35.2% | 84.9% | 32.0% | 32.0% | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| alternating high/low field (64:1), reward normalization on | 13,216.25K ± 609.75K [12,018.77K, 14,362.58K] | 14,110.18K ± 722.65K | 22.7% | 82.0% | 4.7% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^6 | 128 |

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -2, reward normalization on | **35.65K** ± 1.75K [32.19K, 39.03K] | **35.34K** ± 1.89K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -2, reward normalization off | <u>37.59K</u> ± 1.84K [33.92K, 41.13K] | <u>37.35K</u> ± 1.99K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -1, reward normalization off | 41.06K ± 2.09K [36.85K, 45.15K] | 40.59K ± 2.28K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -1, reward normalization on | 45.12K ± 2.42K [40.29K, 49.69K] | 44.54K ± 2.70K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online coordinate-RND, state normalization on, reward normalization on | 55.98K ± 3.15K [49.69K, 62.02K] | 55.29K ± 3.60K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| online tabular-RND, reward normalization on | 58.38K ± 3.54K [51.39K, 65.19K] | 58.16K ± 3.94K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^2 | 128 |
| online tabular-RND, reward normalization off | 61.16K ± 3.80K [53.73K, 68.67K] | 60.25K ± 4.11K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-3 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 66.53K ± 4.96K [56.73K, 76.53K] | 62.92K ± 5.34K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-5 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 67.82K ± 4.75K [58.98K, 77.07K] | 63.79K ± 4.98K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-2 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 68.36K ± 4.92K [58.85K, 77.85K] | 64.96K ± 5.42K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 82.76K ± 7.58K [68.41K, 98.35K] | 69.51K ± 7.94K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | 87.22K ± 8.64K [71.17K, 104.57K] | 70.55K ± 8.94K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.25, reward normalization off | 104.58K ± 24.91K [71.26K, 159.32K] | 73.26K ± 6.46K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-5 | 128 |
| frozen tabular-RND field, reward normalization on | 108.83K ± 11.67K [86.07K, 131.12K] | 90.10K ± 12.09K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| no bonus | 119.18K ± 10.06K [99.67K, 139.25K] | 104.65K ± 9.79K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| frozen tabular-RND field, reward normalization off | 120.13K ± 10.07K [100.48K, 139.86K] | 99.79K ± 11.05K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| alternating high/low field (64:1), reward normalization off | 143.00K ± 15.49K [113.83K, 174.53K] | 117.86K ± 16.23K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 143.12K ± 18.44K [110.81K, 181.84K] | 124.20K ± 13.95K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| alternating high/low field (64:1), reward normalization on | 146.97K ± 14.56K [119.63K, 175.54K] | 126.41K ± 16.07K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 155.12K ± 24.76K [113.64K, 208.80K] | 122.02K ± 15.89K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | 252.15K ± 38.14K [180.05K, 329.65K] | 165.55K ± 38.74K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 1,289.58K ± 151.87K [1,011.61K, 1,608.67K] | 984.04K ± 122.52K | **100.0%** | **100.0%** | <u>99.2%</u> | <u>99.2%</u> | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 2,142.21K ± 349.97K [1,514.82K, 2,877.22K] | 1,113.31K ± 271.12K | <u>94.5%</u> | <u>99.6%</u> | 13.8% | 10.9% | learning rate 0.0001, entropy 0.001, weight 2^-1 | 128 |

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| online tabular-RND, reward normalization on | **99.28K** ± 9.06K [85.70K, 119.77K] | <u>93.38K</u> ± 4.42K | **100.0%** | **100.0%** | 92.2% | 92.2% | learning rate 0.0001, entropy 0.001, weight 2^1 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization on | <u>106.54K</u> ± 16.67K [81.43K, 144.65K] | **78.20K** ± 4.58K | **100.0%** | **100.0%** | 49.2% | 49.2% | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 107.92K ± 6.29K [96.01K, 120.05K] | 99.72K ± 5.65K | **100.0%** | **100.0%** | 92.2% | 92.2% | learning rate 0.0001, entropy 0.001, weight 2^-2 | 128 |
| online tabular-RND, reward normalization off | 108.66K ± 7.02K [95.57K, 122.74K] | 102.57K ± 7.70K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^2 | 128 |
| uniform field, decay exponent -0.5, reward normalization off | 236.86K ± 128.55K [92.03K, 509.36K] | 97.97K ± 5.05K | <u>99.2%</u> | <u>100.0%</u> | 40.6% | 39.8% | learning rate 0.0001, entropy 0.001, weight 2^5 | 128 |
| online coordinate-RND, state normalization on, reward normalization off | 497.91K ± 127.82K [322.49K, 791.24K] | 307.48K ± 33.21K | <u>99.2%</u> | 99.9% | 85.9% | 72.7% | learning rate 0.0001, entropy 0.001, weight 2^5 | 128 |
| online coordinate-RND, state normalization on, reward normalization on | 673.10K ± 209.06K [294.84K, 1,150.62K] | 237.36K ± 30.64K | 97.7% | 99.8% | 91.4% | 88.3% | learning rate 0.0001, entropy 0.001, weight 2^3 | 128 |
| uniform field, decay exponent -0.5, reward normalization on | 764.71K ± 286.74K [242.57K, 1,403.56K] | 104.43K ± 40.18K | 96.1% | 99.7% | <u>93.0%</u> | <u>93.0%</u> | learning rate 0.0001, entropy 0.001, weight 2^-2 | 128 |
| online coordinate-RND, state normalization off, reward normalization off | 1,470.09K ± 309.70K [880.82K, 2,115.77K] | 549.76K ± 117.70K | 94.5% | 99.1% | 87.5% | 87.5% | learning rate 0.0001, entropy 0.001, weight 2^2 | 128 |
| online coordinate-RND, state normalization off, reward normalization on | 1,947.28K ± 402.49K [1,211.05K, 2,790.61K] | 750.67K ± 361.78K | 91.4% | 97.9% | 44.5% | 44.5% | learning rate 0.0001, entropy 0.001, weight 2^-3 | 128 |
| uniform field, decay exponent -0.25, reward normalization on | 2,262.45K ± 503.48K [1,312.32K, 3,288.63K] | 1,127.54K ± 576.93K | 87.5% | 98.0% | 69.8% | 66.4% | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| uniform field, decay exponent -1, reward normalization on | 2,796.97K ± 517.36K [1,823.38K, 3,822.99K] | 1,761.02K ± 612.48K | 84.4% | 98.3% | 16.4% | 16.4% | learning rate 0.0001, entropy 0.001, weight 2^-4 | 128 |
| uniform field, decay exponent -0.25, reward normalization off | 3,090.46K ± 565.81K [2,046.10K, 4,239.32K] | 2,108.91K ± 669.55K | 82.8% | 97.9% | 69.8% | 68.8% | learning rate 0.0001, entropy 0.001, weight 2^-2 | 128 |
| uniform field, decay exponent -1, reward normalization off | 5,767.02K ± 697.01K [4,446.17K, 7,172.93K] | 5,280.91K ± 825.98K | 67.2% | 97.2% | 16.4% | 16.4% | learning rate 0.0001, entropy 0.001, weight 2^-1 | 128 |
| frozen tabular-RND field, reward normalization on | 6,237.76K ± 675.42K [4,963.74K, 7,544.48K] | 5,838.96K ± 800.49K | 63.3% | 94.4% | 27.3% | 27.3% | learning rate 0.0001, entropy 0.001, weight 2^-5 | 128 |
| frozen tabular-RND field, reward normalization off | 6,775.45K ± 878.68K [5,094.44K, 8,573.03K] | 6,476.23K ± 1,041.39K | 60.2% | 94.5% | 3.1% | 0.8% | learning rate 0.0001, entropy 0.001, weight 2^4 | 128 |
| no bonus | 10,200.08K ± 732.00K [8,757.06K, 11,685.38K] | 10,535.00K ± 867.48K | 39.8% | 89.9% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| uniform field, decay exponent -2, reward normalization on | 13,421.77K ± 587.08K [12,189.66K, 14,554.86K] | 14,352.42K ± 695.41K | 20.3% | 84.7% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^6 | 128 |
| uniform field, decay exponent -2, reward normalization off | 13,645.20K ± 574.35K [12,471.71K, 14,691.48K] | 14,616.96K ± 680.29K | 18.8% | 83.6% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^1 | 128 |
| alternating high/low field (64:1), reward normalization off | 14,408.25K ± 481.46K [13,459.87K, 15,296.69K] | 15,520.83K ± 551.55K | 16.4% | 76.9% | 4.7% | 4.7% | learning rate 0.0001, entropy 0.001, weight 2^-6 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 15,080.45K ± 403.89K [14,170.68K, 15,861.90K] | 16,315.84K ± 419.66K | 10.2% | 73.2% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^1 | 128 |
| alternating high/low field (64:1), reward normalization on | 15,237.37K ± 423.65K [14,352.62K, 16,022.35K] | 16,504.23K ± 391.16K | 9.4% | 70.7% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^0 | 128 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 15,338.37K ± 389.65K [14,553.50K, 16,122.79K] | 16,622.50K ± 334.22K | 8.6% | 73.2% | 0.0% | 0.0% | learning rate 0.0001, entropy 0.001, weight 2^-2 | 128 |

## Trimmed means: how many seeds to remove, and whether the results change

Define $v_{(1)} \le \dots \le v_{(K)}$ as the sorted capped first-goal times $\min(T_i, B)$ of one configuration. The trimmed mean removing $k$ seeds from each end is $M_k(B) = \dfrac{1}{K-2k} \sum_{i=k+1}^{K-k} v_{(i)}$; $M_0(B) = M(B)$.

**The extreme values are on the high side.** Among the four best conditions of each case (by $M(B)$, 16 conditions), up to 13 of 128 seeds lie more than three interquartile ranges above the third quartile, and none lie that far below the first quartile. A seed that never reaches the goal counts as $B$ = 16,777,216, so one such seed adds 131.1K to a 128-seed mean: a single failure can decide the ranking of conditions whose typical seed needs 10K to 100K transitions.

**Chosen: 10 seeds from each end of 128** (7.8% per end, 108 seeds kept; 1 per end of the 16 development seeds). Removing 2 or 4 leaves the single failures of several conditions in the mean. Between 6 and 10 removed per end, the trimmed means of these 16 conditions still move by up to 9.7%; between 10 and 13 (10% per end) by up to 3.3%. Removing more than about 10% per end removes every failure of conditions that fail on up to that many seeds, so they look as good as conditions that never fail. Trimming also removes exactly the never-found seeds that $M(B)$ is meant to count, so configurations are still chosen by the untrimmed $M(B)$; the trimmed mean is reported beside it.

### straight grid, 32 cells, tabular-PPO

| seeds removed from each end | best condition | its trimmed mean | second condition | its trimmed mean |
|---|---|---|---|---|
| 0 | uniform field, decay exponent -2, reward normalization on | 5.49K | uniform field, decay exponent -2, reward normalization off | 5.71K |
| 2 | uniform field, decay exponent -2, reward normalization on | 5.46K | uniform field, decay exponent -2, reward normalization off | 5.70K |
| 4 | uniform field, decay exponent -2, reward normalization on | 5.44K | uniform field, decay exponent -2, reward normalization off | 5.70K |
| 6 | uniform field, decay exponent -2, reward normalization on | 5.44K | uniform field, decay exponent -2, reward normalization off | 5.69K |
| 8 | uniform field, decay exponent -2, reward normalization on | 5.43K | uniform field, decay exponent -2, reward normalization off | 5.69K |
| 10 | uniform field, decay exponent -2, reward normalization on | 5.42K | uniform field, decay exponent -2, reward normalization off | 5.68K |
| 13 | uniform field, decay exponent -2, reward normalization on | 5.41K | uniform field, decay exponent -2, reward normalization off | 5.67K |
| 16 | uniform field, decay exponent -2, reward normalization on | 5.40K | uniform field, decay exponent -2, reward normalization off | 5.65K |
| 20 | uniform field, decay exponent -2, reward normalization on | 5.39K | uniform field, decay exponent -2, reward normalization off | 5.64K |

The best condition is the same for every number removed, up to 20 per end. With 10 removed, uniform field, decay exponent -2, reward normalization on leads uniform field, decay exponent -2, reward normalization off by 0.26K; their standard errors are 0.24K and 0.27K.

### straight grid, 32 cells, coordinate-PPO

| seeds removed from each end | best condition | its trimmed mean | second condition | its trimmed mean |
|---|---|---|---|---|
| 0 | uniform field, decay exponent -1, reward normalization on | 2.41K | uniform field, decay exponent -0.5, reward normalization off | 2.98K |
| 2 | uniform field, decay exponent -1, reward normalization on | 2.34K | uniform field, decay exponent -1, reward normalization off | 2.87K |
| 4 | uniform field, decay exponent -1, reward normalization on | 2.27K | uniform field, decay exponent -1, reward normalization off | 2.81K |
| 6 | uniform field, decay exponent -1, reward normalization on | 2.22K | uniform field, decay exponent -1, reward normalization off | 2.76K |
| 8 | uniform field, decay exponent -1, reward normalization on | 2.18K | uniform field, decay exponent -1, reward normalization off | 2.72K |
| 10 | uniform field, decay exponent -1, reward normalization on | 2.15K | uniform field, decay exponent -1, reward normalization off | 2.69K |
| 13 | uniform field, decay exponent -1, reward normalization on | 2.10K | uniform field, decay exponent -1, reward normalization off | 2.65K |
| 16 | uniform field, decay exponent -1, reward normalization on | 2.06K | uniform field, decay exponent -1, reward normalization off | 2.62K |
| 20 | uniform field, decay exponent -1, reward normalization on | 2.00K | uniform field, decay exponent -1, reward normalization off | 2.58K |

The best condition is the same for every number removed, up to 20 per end. With 10 removed, uniform field, decay exponent -1, reward normalization on leads uniform field, decay exponent -1, reward normalization off by 0.54K; their standard errors are 0.18K and 0.21K.

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

| seeds removed from each end | best condition | its trimmed mean | second condition | its trimmed mean |
|---|---|---|---|---|
| 0 | uniform field, decay exponent -2, reward normalization off | 10.38K | uniform field, decay exponent -2, reward normalization on | 10.44K |
| 2 | uniform field, decay exponent -2, reward normalization off | 10.33K | uniform field, decay exponent -2, reward normalization on | 10.39K |
| 4 | uniform field, decay exponent -2, reward normalization off | 10.35K | uniform field, decay exponent -2, reward normalization on | 10.35K |
| 6 | uniform field, decay exponent -2, reward normalization on | 10.34K | uniform field, decay exponent -2, reward normalization off | 10.37K |
| 8 | uniform field, decay exponent -2, reward normalization on | 10.32K | uniform field, decay exponent -2, reward normalization off | 10.38K |
| 10 | uniform field, decay exponent -2, reward normalization on | 10.29K | uniform field, decay exponent -2, reward normalization off | 10.39K |
| 13 | uniform field, decay exponent -2, reward normalization on | 10.25K | uniform field, decay exponent -2, reward normalization off | 10.40K |
| 16 | uniform field, decay exponent -2, reward normalization on | 10.23K | uniform field, decay exponent -2, reward normalization off | 10.42K |
| 20 | uniform field, decay exponent -2, reward normalization on | 10.20K | uniform field, decay exponent -2, reward normalization off | 10.43K |

The best condition differs from the untrimmed one at 6, 8, 10, 13, 16, 20 removed per end. With 10 removed, uniform field, decay exponent -2, reward normalization on leads uniform field, decay exponent -2, reward normalization off by 0.10K; their standard errors are 0.42K and 0.38K. It is not the untrimmed best (uniform field, decay exponent -2, reward normalization off): 0 of its 128 seeds never reached the goal, against 0; each such seed adds 131.1K to an untrimmed mean.

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

| seeds removed from each end | best condition | its trimmed mean | second condition | its trimmed mean |
|---|---|---|---|---|
| 0 | uniform field, decay exponent -1, reward normalization on | 63.87K | uniform field, decay exponent -0.25, reward normalization on | 75.49K |
| 2 | frozen coordinate-RND field, state normalization on, reward normalization off | 37.72K | uniform field, decay exponent -1, reward normalization on | 54.26K |
| 4 | frozen coordinate-RND field, state normalization on, reward normalization off | 36.27K | uniform field, decay exponent -1, reward normalization on | 49.92K |
| 6 | frozen coordinate-RND field, state normalization on, reward normalization off | 36.03K | uniform field, decay exponent -1, reward normalization on | 46.42K |
| 8 | frozen coordinate-RND field, state normalization on, reward normalization off | 35.90K | uniform field, decay exponent -1, reward normalization on | 43.81K |
| 10 | frozen coordinate-RND field, state normalization on, reward normalization off | 35.82K | uniform field, decay exponent -1, reward normalization on | 41.90K |
| 13 | frozen coordinate-RND field, state normalization on, reward normalization off | 35.72K | uniform field, decay exponent -1, reward normalization on | 40.51K |
| 16 | frozen coordinate-RND field, state normalization on, reward normalization off | 35.80K | uniform field, decay exponent -1, reward normalization on | 39.61K |
| 20 | frozen coordinate-RND field, state normalization on, reward normalization off | 35.90K | uniform field, decay exponent -1, reward normalization on | 39.15K |

The best condition differs from the untrimmed one at 2, 4, 6, 8, 10, 13, 16, 20 removed per end. With 10 removed, frozen coordinate-RND field, state normalization on, reward normalization off leads uniform field, decay exponent -1, reward normalization on by 6.07K; their standard errors are 1.59K and 3.78K. It is not the untrimmed best (uniform field, decay exponent -1, reward normalization on): 1 of its 128 seeds never reached the goal, against 0; each such seed adds 131.1K to an untrimmed mean.

## Development

### straight grid, 32 cells, tabular-PPO

Best configuration of each condition (1148 of 1148 settings finished):

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -2, reward normalization on | **6.09K** ± 0.58K [4.94K, 7.18K] | **6.10K** ± 0.61K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| uniform field, decay exponent -2, reward normalization off | <u>6.20K</u> ± 0.56K [5.05K, 7.27K] | <u>6.30K</u> ± 0.61K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -1, reward normalization on | 6.82K ± 0.74K [5.38K, 8.27K] | 6.91K ± 0.82K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -1, reward normalization off | 6.99K ± 0.72K [5.58K, 8.41K] | 7.09K ± 0.79K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| online tabular-RND, reward normalization on | 7.17K ± 0.36K [6.60K, 8.01K] | 7.24K ± 0.41K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-3 | 16 |
| online tabular-RND, reward normalization off | 8.11K ± 0.50K [7.17K, 9.02K] | 8.18K ± 0.50K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -0.5, reward normalization on | 8.68K ± 1.24K [6.42K, 11.25K] | 8.50K ± 1.36K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| online coordinate-RND, state normalization off, reward normalization off | 9.96K ± 2.02K [7.15K, 14.49K] | 9.54K ± 2.01K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| uniform field, decay exponent -0.5, reward normalization off | 10.39K ± 1.42K [7.64K, 13.11K] | 10.26K ± 1.56K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| frozen tabular-RND field, reward normalization on | 14.77K ± 5.64K [7.17K, 27.65K] | 10.07K ± 4.65K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -0.25, reward normalization off | 14.99K ± 3.12K [9.73K, 21.78K] | 13.41K ± 3.14K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| online coordinate-RND, state normalization on, reward normalization on | 15.86K ± 2.87K [10.46K, 21.26K] | 15.41K ± 2.97K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^4 | 16 |
| online coordinate-RND, state normalization on, reward normalization off | 17.01K ± 1.99K [12.78K, 21.02K] | 16.82K ± 2.20K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-5 | 16 |
| online coordinate-RND, state normalization off, reward normalization on | 18.11K ± 2.40K [14.23K, 23.60K] | 17.56K ± 2.49K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-3 | 16 |
| frozen tabular-RND field, reward normalization off | 18.45K ± 4.57K [11.00K, 28.90K] | 14.10K ± 4.01K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 19.64K ± 2.95K [14.33K, 25.26K] | 18.66K ± 3.05K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -0.25, reward normalization on | 22.52K ± 3.88K [15.43K, 30.59K] | 20.97K ± 3.91K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 16 |
| alternating high/low field (64:1), reward normalization on | 22.95K ± 5.52K [13.81K, 35.03K] | 19.22K ± 5.13K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 16 |
| alternating high/low field (64:1), reward normalization off | 23.72K ± 5.68K [14.33K, 36.16K] | 20.10K ± 5.44K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 16 |
| no bonus | 25.72K ± 6.06K [15.65K, 39.20K] | 21.73K ± 5.67K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^0 | 16 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 27.20K ± 7.31K [15.71K, 42.96K] | 24.52K ± 7.43K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-3 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 132.64K ± 55.64K [22.42K, 242.87K] | 102.22K ± 56.12K | **100.0%** | **100.0%** | <u>93.8%</u> | <u>93.8%</u> | learning rate 0.0001, entropy 0.001, weight 2^-6 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 259.45K ± 141.79K [22.42K, 546.70K] | 145.70K ± 128.84K | **100.0%** | **100.0%** | 12.5% | 12.5% | learning rate 0.0001, entropy 0.01, weight 2^-4 | 16 |

![development, straight, tabular](plots/development_intrinsic-weight_straight_tabular.png)

### straight grid, 32 cells, coordinate-PPO

Best configuration of each condition (1148 of 1148 settings finished):

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -1, reward normalization on | **1,199** ± 204 [849, 1,640] | **1,086** ± 209 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -1, reward normalization off | <u>1,792</u> ± 492 [994, 2,875] | <u>1,441</u> ± 478 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| frozen tabular-RND field, reward normalization off | 1,983 ± 285 [1,389, 2,549] | 1,834 ± 280 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-3 | 16 |
| frozen tabular-RND field, reward normalization on | 2,006 ± 413 [1,209, 2,803] | 1,802 ± 429 | **100.0%** | **100.0%** | <u>87.5%</u> | <u>87.5%</u> | learning rate 0.001, entropy 0.001, weight 2^0 | 16 |
| uniform field, decay exponent -0.5, reward normalization off | 2,122 ± 387 [1,400, 2,927] | 2,007 ± 423 | **100.0%** | **100.0%** | **100.0%** | <u>87.5%</u> | learning rate 0.001, entropy 0.01, weight 2^4 | 16 |
| uniform field, decay exponent -0.5, reward normalization on | 2,240 ± 479 [1,433, 3,312] | 1,929 ± 471 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| uniform field, decay exponent -2, reward normalization on | 2,276 ± 501 [1,359, 3,382] | 2,113 ± 551 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| online tabular-RND, reward normalization off | 3,105 ± 228 [2,646, 3,564] | 3,141 ± 236 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -2, reward normalization off | 3,213 ± 621 [2,005, 4,441] | 3,131 ± 700 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^3 | 16 |
| online tabular-RND, reward normalization on | 3,616 ± 138 [3,330, 3,863] | 3,578 ± 147 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-4 | 16 |
| uniform field, decay exponent -0.25, reward normalization on | 3,723 ± 529 [2,739, 4,783] | 3,676 ± 587 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -0.25, reward normalization off | 3,877 ± 644 [2,651, 5,219] | 3,763 ± 715 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| no bonus | 6,505 ± 2,796 [2,513, 12,913] | 4,022 ± 2,434 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^0 | 16 |
| online coordinate-RND, state normalization on, reward normalization off | 6,870 ± 715 [5,596, 8,298] | 6,679 ± 758 | **100.0%** | **100.0%** | <u>87.5%</u> | <u>87.5%</u> | learning rate 0.001, entropy 0.01, weight 2^-3 | 16 |
| online coordinate-RND, state normalization on, reward normalization on | 7,520 ± 631 [6,274, 8,765] | 7,469 ± 709 | **100.0%** | **100.0%** | 81.2% | 81.2% | learning rate 0.001, entropy 0.01, weight 2^5 | 16 |
| online coordinate-RND, state normalization off, reward normalization off | 9,357 ± 227 [8,968, 9,877] | 9,444 ± 205 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-4 | 16 |
| online coordinate-RND, state normalization off, reward normalization on | 10,512 ± 935 [8,796, 12,278] | 10,439 ± 943 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^4 | 16 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 19,230 ± 2,055 [15,322, 23,237] | 17,114 ± 1,833 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-5 | 16 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 25,327 ± 2,115 [22,170, 29,948] | 25,777 ± 2,311 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-3 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 25,862 ± 14,542 [8,221, 59,222] | 14,113 ± 12,379 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-5 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 63,847 ± 20,397 [25,590, 102,104] | 49,049 ± 21,409 | **100.0%** | **100.0%** | 50.0% | 50.0% | learning rate 0.001, entropy 0.001, weight 2^1 | 16 |
| alternating high/low field (64:1), reward normalization on | 3,461,246 ± 1,045,623 [1,687,387, 5,720,811] | 2,757,284 ± 995,738 | <u>93.8%</u> | <u>97.7%</u> | 43.8% | 37.5% | learning rate 0.001, entropy 0.001, weight 2^0 | 16 |
| alternating high/low field (64:1), reward normalization off | 3,966,218 ± 1,567,540 [1,171,450, 7,224,026] | 3,334,402 ± 1,731,784 | 81.2% | 91.2% | 6.2% | 6.2% | learning rate 0.001, entropy 0.001, weight 2^5 | 16 |

![development, straight, neural](plots/development_intrinsic-weight_straight_neural.png)

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

Best configuration of each condition (1148 of 1148 settings finished):

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| uniform field, decay exponent -2, reward normalization off | **12.40K** ± 1.12K [10.21K, 14.58K] | **12.29K** ± 1.17K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| uniform field, decay exponent -2, reward normalization on | <u>13.21K</u> ± 1.70K [10.07K, 16.73K] | <u>12.72K</u> ± 1.85K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| uniform field, decay exponent -1, reward normalization off | 13.87K ± 1.32K [11.45K, 16.59K] | 13.50K ± 1.36K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| uniform field, decay exponent -1, reward normalization on | 14.56K ± 1.23K [12.21K, 17.00K] | 14.38K ± 1.27K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| online tabular-RND, reward normalization on | 19.33K ± 1.92K [15.49K, 23.17K] | 19.36K ± 2.01K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-3 | 16 |
| online coordinate-RND, state normalization on, reward normalization on | 20.98K ± 0.63K [19.84K, 22.11K] | 20.98K ± 0.69K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^2 | 16 |
| online coordinate-RND, state normalization on, reward normalization off | 21.77K ± 3.12K [16.21K, 27.95K] | 21.27K ± 3.18K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-1 | 16 |
| online tabular-RND, reward normalization off | 22.29K ± 1.73K [18.77K, 25.24K] | 21.71K ± 1.82K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-1 | 16 |
| uniform field, decay exponent -0.5, reward normalization on | 32.67K ± 7.05K [20.38K, 47.45K] | 29.56K ± 7.27K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-5 | 16 |
| uniform field, decay exponent -0.5, reward normalization off | 36.86K ± 8.67K [21.67K, 56.03K] | 33.55K ± 9.54K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| online coordinate-RND, state normalization off, reward normalization off | 39.39K ± 5.23K [29.17K, 49.62K] | 39.18K ± 5.31K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^5 | 16 |
| online coordinate-RND, state normalization off, reward normalization on | 44.14K ± 7.18K [31.26K, 58.14K] | 43.94K ± 7.40K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-5 | 16 |
| frozen coordinate-RND field, state normalization on, reward normalization off | 46.29K ± 5.23K [35.00K, 56.04K] | 46.26K ± 5.79K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-5 | 16 |
| frozen tabular-RND field, reward normalization on | 80.95K ± 9.82K [62.98K, 103.68K] | 79.98K ± 11.61K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-6 | 16 |
| frozen tabular-RND field, reward normalization off | 82.01K ± 18.77K [42.96K, 112.80K] | 80.40K ± 20.85K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^-2 | 16 |
| alternating high/low field (64:1), reward normalization off | 88.26K ± 27.45K [41.01K, 146.20K] | 80.21K ± 30.85K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-6 | 16 |
| uniform field, decay exponent -0.25, reward normalization off | 92.60K ± 18.84K [58.85K, 131.25K] | 81.94K ± 19.19K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.01, weight 2^-6 | 16 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 97.54K ± 24.72K [50.81K, 141.89K] | 94.29K ± 26.51K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.001, weight 2^-4 | 16 |
| alternating high/low field (64:1), reward normalization on | 100.17K ± 24.50K [61.45K, 154.71K] | 83.82K ± 24.46K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 16 |
| no bonus | 145.07K ± 27.50K [94.69K, 198.64K] | 136.68K ± 30.00K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^0 | 16 |
| uniform field, decay exponent -0.25, reward normalization on | 185.18K ± 69.46K [75.29K, 339.65K] | 137.09K ± 67.93K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.001, weight 2^-6 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 832.14K ± 59.58K [706.29K, 928.96K] | 838.95K ± 63.77K | **100.0%** | **100.0%** | <u>85.5%</u> | <u>0.0%</u> | learning rate 0.001, entropy 0.01, weight 2^6 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 1,149.91K ± 213.29K [823.06K, 1,616.71K] | 1,127.68K ± 229.11K | **100.0%** | **100.0%** | **100.0%** | <u>0.0%</u> | learning rate 0.001, entropy 0.01, weight 2^2 | 16 |

![development, snake, tabular](plots/development_intrinsic-weight_snake_tabular.png)

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

Best configuration of each condition (1148 of 1148 settings finished):

| condition | mean capped first-goal steps ↓ ± standard error [95%] | trimmed mean (10 of every 128 seeds removed from each end) ± standard error | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |
|---|---|---|---|---|---|---|---|---|
| frozen coordinate-RND field, state normalization on, reward normalization off | **26.80K** ± 2.27K [22.01K, 30.65K] | **27.11K** ± 2.49K | **100.0%** | **100.0%** | 50.0% | 31.2% | learning rate 0.001, entropy 0.01, weight 2^1 | 16 |
| uniform field, decay exponent -1, reward normalization on | <u>34.96K</u> ± 3.40K [28.31K, 41.51K] | <u>35.30K</u> ± 3.52K | **100.0%** | **100.0%** | 87.5% | 87.5% | learning rate 0.001, entropy 0.01, weight 2^4 | 16 |
| frozen coordinate-RND field, state normalization on, reward normalization on | 35.52K ± 4.20K [27.62K, 43.42K] | 35.48K ± 4.69K | **100.0%** | **100.0%** | 21.1% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^3 | 16 |
| uniform field, decay exponent -0.5, reward normalization off | 39.63K ± 4.32K [30.93K, 47.74K] | 40.01K ± 4.68K | **100.0%** | **100.0%** | 25.0% | 25.0% | learning rate 0.001, entropy 0.001, weight 2^3 | 16 |
| online tabular-RND, reward normalization off | 41.59K ± 7.73K [26.46K, 56.73K] | 41.44K ± 8.55K | **100.0%** | **100.0%** | 18.8% | 18.8% | learning rate 0.001, entropy 0.01, weight 2^4 | 16 |
| uniform field, decay exponent -0.5, reward normalization on | 44.97K ± 6.48K [32.97K, 57.95K] | 44.13K ± 7.13K | **100.0%** | **100.0%** | 25.0% | 25.0% | learning rate 0.001, entropy 0.001, weight 2^-1 | 16 |
| uniform field, decay exponent -0.25, reward normalization off | 48.16K ± 9.55K [30.42K, 67.68K] | 44.48K ± 10.13K | **100.0%** | **100.0%** | 18.8% | 18.8% | learning rate 0.001, entropy 0.001, weight 2^0 | 16 |
| uniform field, decay exponent -0.25, reward normalization on | 57.30K ± 12.56K [34.52K, 84.06K] | 53.21K ± 13.56K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.001, entropy 0.01, weight 2^1 | 16 |
| uniform field, decay exponent -1, reward normalization off | 63.86K ± 14.72K [38.05K, 95.90K] | 58.60K ± 15.94K | **100.0%** | **100.0%** | 43.8% | 37.5% | learning rate 0.001, entropy 0.01, weight 2^1 | 16 |
| online tabular-RND, reward normalization on | 76.35K ± 8.96K [59.56K, 92.83K] | 77.11K ± 9.81K | **100.0%** | **100.0%** | 81.2% | 75.0% | learning rate 0.0001, entropy 0.01, weight 2^-6 | 16 |
| online coordinate-RND, state normalization on, reward normalization off | 90.60K ± 17.72K [68.32K, 131.95K] | 78.37K ± 16.00K | **100.0%** | **100.0%** | <u>93.8%</u> | 87.5% | learning rate 0.001, entropy 0.01, weight 2^4 | 16 |
| online coordinate-RND, state normalization on, reward normalization on | 164.22K ± 11.18K [141.87K, 186.57K] | 165.58K ± 11.97K | **100.0%** | **100.0%** | **100.0%** | <u>93.8%</u> | learning rate 0.0001, entropy 0.01, weight 2^-1 | 16 |
| online coordinate-RND, state normalization off, reward normalization off | 268.83K ± 27.93K [215.84K, 321.82K] | 256.19K ± 29.87K | **100.0%** | **100.0%** | **100.0%** | **100.0%** | learning rate 0.0001, entropy 0.01, weight 2^0 | 16 |
| frozen tabular-RND field, reward normalization off | 380.98K ± 174.98K [97.88K, 770.95K] | 227.33K ± 159.33K | **100.0%** | **100.0%** | 43.8% | 6.2% | learning rate 0.001, entropy 0.01, weight 2^3 | 16 |
| online coordinate-RND, state normalization off, reward normalization on | 419.98K ± 47.69K [342.60K, 532.44K] | 424.81K ± 47.85K | **100.0%** | **100.0%** | 62.5% | 62.5% | learning rate 0.0001, entropy 0.01, weight 2^-3 | 16 |
| frozen tabular-RND field, reward normalization on | 714.75K ± 325.69K [262.69K, 1,439.89K] | 417.96K ± 283.13K | **100.0%** | **100.0%** | 12.5% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^6 | 16 |
| no bonus | 8,466.34K ± 2,080.44K [4,321.51K, 12,619.21K] | 8,476.02K ± 2,377.34K | 50.0% | <u>90.2%</u> | 0.0% | 0.0% | learning rate 0.0001, entropy 0.01, weight 2^0 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization on | 8,756.40K ± 914.23K [6,898.94K, 10,282.77K] | 8,808.07K ± 1,026.24K | <u>56.2%</u> | 86.9% | 12.5% | 12.5% | learning rate 0.001, entropy 0.01, weight 2^-2 | 16 |
| uniform field, decay exponent -2, reward normalization off | 9,733.63K ± 1,908.79K [5,770.07K, 13,344.97K] | 9,925.35K ± 2,179.53K | 50.0% | 84.8% | 0.0% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^-1 | 16 |
| alternating high/low field (64:1), reward normalization on | 9,809.34K ± 1,953.37K [5,982.40K, 13,635.44K] | 10,012.10K ± 2,232.25K | 43.8% | 81.4% | 0.0% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^2 | 16 |
| uniform field, decay exponent -2, reward normalization on | 9,868.77K ± 1,842.70K [6,314.27K, 13,339.29K] | 10,079.43K ± 2,102.37K | 50.0% | 88.6% | 0.0% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^1 | 16 |
| frozen coordinate-RND field, state normalization off, reward normalization off | 10,792.52K ± 1,735.96K [8,186.87K, 14,829.90K] | 11,092.18K ± 1,966.28K | 43.8% | 84.5% | 18.8% | 18.8% | learning rate 0.001, entropy 0.01, weight 2^-4 | 16 |
| alternating high/low field (64:1), reward normalization off | 11,540.57K ± 1,943.14K [7,355.27K, 14,685.04K] | 11,990.65K ± 2,216.16K | 31.2% | 72.8% | 0.0% | 0.0% | learning rate 0.001, entropy 0.01, weight 2^4 | 16 |

![development, snake, neural](plots/development_intrinsic-weight_snake_neural.png)

## Decay: online RND (coordinate- and tabular-RND) against the count rule

Define $e_t(s)$ as the online RND error of cell $s$ after $t$ updates and $N_t(s)$ as its visits. The figures plot $e_t(s)/e_0(s)$ against $N_t(s)$ for the chosen confirmation configurations, beside $N^{-1/2}$, the count rule's decay. Slopes are least-squares fits of $\log(e_t/e_0)$ on $\log N_t$ per cell and seed (cells with at least three checkpoints of 2 or more visits); the count rule's slope is -0.5.

### straight grid, 32 cells, tabular-PPO

![decay, straight, tabular](plots/decay_online-rnd_straight_tabular.png)

| condition | median slope | middle half of slopes | fitted cells |
|---|---|---|---|
| online coordinate-RND, state normalization off, reward normalization off | -2.97 | -3.27 to -2.55 | 4096 |
| online coordinate-RND, state normalization off, reward normalization on | -2.86 | -3.22 to -2.41 | 4096 |
| online coordinate-RND, state normalization on, reward normalization off | -2.64 | -3.01 to -2.31 | 4096 |
| online coordinate-RND, state normalization on, reward normalization on | -2.64 | -3.01 to -2.30 | 4096 |
| online tabular-RND, reward normalization off | -3.37 | -3.71 to -3.10 | 4096 |
| online tabular-RND, reward normalization on | -3.36 | -3.69 to -3.10 | 32768 |

### straight grid, 32 cells, coordinate-PPO

![decay, straight, neural](plots/decay_online-rnd_straight_neural.png)

| condition | median slope | middle half of slopes | fitted cells |
|---|---|---|---|
| online coordinate-RND, state normalization off, reward normalization off | -2.84 | -3.17 to -2.43 | 4080 |
| online coordinate-RND, state normalization off, reward normalization on | -2.75 | -3.12 to -2.24 | 4082 |
| online coordinate-RND, state normalization on, reward normalization off | -2.60 | -3.03 to -2.11 | 4096 |
| online coordinate-RND, state normalization on, reward normalization on | -2.53 | -2.95 to -2.02 | 4069 |
| online tabular-RND, reward normalization off | -3.61 | -3.84 to -3.30 | 4096 |
| online tabular-RND, reward normalization on | -3.52 | -3.71 to -3.28 | 32768 |

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

![decay, snake, tabular](plots/decay_online-rnd_snake_tabular.png)

| condition | median slope | middle half of slopes | fitted cells |
|---|---|---|---|
| online coordinate-RND, state normalization off, reward normalization off | -1.00 | -1.19 to -0.88 | 6400 |
| online coordinate-RND, state normalization off, reward normalization on | -1.01 | -1.19 to -0.88 | 6400 |
| online coordinate-RND, state normalization on, reward normalization off | -1.14 | -1.30 to -1.02 | 6400 |
| online coordinate-RND, state normalization on, reward normalization on | -1.12 | -1.28 to -1.00 | 6400 |
| online tabular-RND, reward normalization off | -2.86 | -3.12 to -2.59 | 6400 |
| online tabular-RND, reward normalization on | -2.87 | -3.15 to -2.60 | 51200 |

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

![decay, snake, neural](plots/decay_online-rnd_snake_neural.png)

| condition | median slope | middle half of slopes | fitted cells |
|---|---|---|---|
| online coordinate-RND, state normalization off, reward normalization off | -0.82 | -0.97 to -0.69 | 6392 |
| online coordinate-RND, state normalization off, reward normalization on | -0.76 | -0.90 to -0.63 | 6374 |
| online coordinate-RND, state normalization on, reward normalization off | -1.05 | -1.18 to -0.90 | 6397 |
| online coordinate-RND, state normalization on, reward normalization on | -0.87 | -1.03 to -0.73 | 6393 |
| online tabular-RND, reward normalization off | -2.39 | -2.90 to -1.76 | 6345 |
| online tabular-RND, reward normalization on | -0.99 | -1.34 to -0.61 | 50856 |

## Records

Unit records: `data/research/fixed-*/summary.json` and `behavior.npz` of this folder and of the extension run listed in `report_extensions.json`. Frozen confirmation choices: `confirmation_frozen.json` of each folder. Report generated 2026-11-11 15:13 PT.

# Online tabular-RND auto research

Generated 2026-11-10 02:35 PT (times are Pacific, converted from machines running Eastern). Finished settings: 3563. Protocol: `program.md`; log: `STATE.md`.

## Contents

| Section | First written | Last changed | Status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-09 21:43 PT | 2026-11-09 22:42 PT | unread |
| [screen 01: results](#screen-01-results) | 2026-11-09 21:43 PT | 2026-11-10 02:35 PT | unread |
| [screen 02: results](#screen-02-results) | 2026-11-09 23:44 PT | 2026-11-10 02:35 PT | unread |
| [screen 03: results](#screen-03-results) | 2026-11-10 02:03 PT | 2026-11-10 02:35 PT | unread |

[Read with section colors](analysis.html). No section marks itself read.

## Measures

Define $b_t(s)$ as the bonus of cell $s$ after $t$ updates (before reward normalization) and $N_t(s)$ as the visits of cell $s$. Start max/min is $\max_s b_0(s) / \min_s b_0(s)$ (median over seeds; target at most 1.5). The decay slope is the least-squares slope of $\log(b_t(s)/b_0(s))$ on $\log N_t(s)$ per cell and seed, over checkpoints with $N_t(s) \ge 2$ (target [-0.6, -0.4]). Late count consistency is the spread over cells of $\log(b_T(s) \sqrt{N_T(s)})$ at the last checkpoint $T$ (zero when every cell follows the same $c/\sqrt{N}$). The screen score is the mean capped first-goal time $M(B)$ at the screen's budget, the best of each idea's settings (policy learning rate 1e-3, and 1e-4 for the control and the main ideas; entropy 0.001 or 0.01; weight 2^-6 to 2^6 in steps of 4×) on seeds 0-31; the standard error and interval resample field seeds (four policies each). Every idea keeps CleanRL's reward normalization on unless its name says it replaces it. The last column counts, over all of an idea's settings in the case, the copies whose bonus became not-a-number or infinite: such an idea is unstable, and its score is not evidence about the two properties.

## screen 01: results

Budget per policy: 4,194,304 transitions, on the full 16,777,216-transition learning-rate schedule.

### straight grid, 32 cells, tabular-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| control | **7.22K** ± 0.28K [6.73K, 7.81K] | 100.0% | 3.65 | -3.78 [-4.18, -3.42] | 4.83 | neither | learning rate 0.001, entropy 0.001, weight 2^-4 | 32 | 0 of 896 |
| wide-input-layer-64 | <u>7.41K</u> ± 0.50K [6.57K, 8.44K] | 100.0% | 2.63 | -3.84 [-4.24, -3.50] | 4.87 | neither | learning rate 0.001, entropy 0.001, weight 2^-4 | 32 | 0 of 448 |
| initial-ratio-readout | 7.95K ± 0.44K [7.14K, 8.84K] | 100.0% | 1.00 | -3.75 [-4.16, -3.41] | 4.83 | even start | learning rate 0.001, entropy 0.01, weight 2^-4 | 32 | 0 of 448 |
| unit-start | 8.57K ± 0.64K [7.33K, 9.86K] | 100.0% | 1.00 | -4.16 [-4.60, -3.66] | 5.52 | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| length-readout | 9.79K ± 0.80K [8.24K, 11.40K] | 100.0% | 1.91 | -1.87 [-2.09, -1.69] | 2.43 | neither | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| unit-start-shrink-target-sgd-2-length-readout | 10.47K ± 0.81K [8.91K, 12.09K] | 100.0% | 1.00 | -0.51 [-0.55, -0.48] | 0.93 | both | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| separate-adam-constant-learning-rate-1e-4 | 11.37K ± 1.33K [8.79K, 13.92K] | 100.0% | 3.65 | -2.54 [-2.90, -2.20] | 3.93 | neither | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| bias-start-0.5 | 11.40K ± 1.08K [9.37K, 13.55K] | 100.0% | 1.04 | -1.93 [-2.31, -1.63] | 4.16 | even start | learning rate 0.001, entropy 0.01, weight 2^-2 | 32 | 0 of 448 |
| unit-start-shrink-target-sgd-3-length-readout | 14.89K ± 2.50K [10.55K, 20.32K] | 100.0% | 1.00 | -0.53 [-0.58, -0.49] | 0.94 | both | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 14 of 896 |
| unit-start-power-4-sgd-0.3-length-readout | 17.15K ± 3.35K [11.61K, 24.30K] | 100.0% | 1.00 | -0.50 [-0.53, -0.47] | 0.89 | both | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 25 of 896 |
| unit-start-adagrad-0.1 | 19.95K ± 3.63K [13.10K, 27.15K] | 100.0% | 1.00 | -0.57 [-1.43, -0.34] | 1.70 | both | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 448 |
| shrink-target-sgd-1-length-readout | 22.31K ± 3.30K [15.93K, 28.65K] | 100.0% | 1.91 | -0.37 [-0.41, -0.34] | 0.60 | neither | learning rate 0.0001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| power-4-sgd-0.1-length-readout | 545.94K ± 367.86K [19.53K, 1,331.62K] | 87.5% | 1.91 | -0.33 [-0.37, -0.29] | 0.53 | neither | learning rate 0.0001, entropy 0.001, weight 2^-6 | 32 | 140 of 896 |

### straight grid, 32 cells, coordinate-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| separate-adam-constant-learning-rate-1e-4 | **3.78K** ± 0.29K [3.22K, 4.35K] | 100.0% | 3.65 | -2.68 [-3.14, -2.30] | 3.96 | neither | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| control | <u>3.90K</u> ± 0.20K [3.51K, 4.29K] | 100.0% | 3.65 | -4.02 [-4.39, -3.60] | 4.87 | neither | learning rate 0.001, entropy 0.001, weight 2^4 | 32 | 0 of 896 |
| wide-input-layer-64 | 4.06K ± 0.11K [3.84K, 4.28K] | 100.0% | 2.63 | -4.13 [-4.47, -3.71] | 4.90 | neither | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 448 |
| initial-ratio-readout | 4.27K ± 0.19K [3.86K, 4.61K] | 100.0% | 1.00 | -4.02 [-4.38, -3.59] | 4.87 | even start | learning rate 0.001, entropy 0.001, weight 2^-2 | 32 | 0 of 448 |
| unit-start | 4.66K ± 0.22K [4.26K, 5.10K] | 100.0% | 1.00 | -3.97 [-4.54, -3.29] | 5.54 | even start | learning rate 0.001, entropy 0.001, weight 2^0 | 32 | 0 of 896 |
| unit-start-adagrad-0.1 | 4.76K ± 0.26K [4.25K, 5.26K] | 100.0% | 1.00 | -0.63 [-1.25, -0.33] | 1.87 | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 448 |
| length-readout | 4.80K ± 0.17K [4.44K, 5.11K] | 100.0% | 1.91 | -2.05 [-2.22, -1.81] | 2.46 | neither | learning rate 0.001, entropy 0.001, weight 2^0 | 32 | 0 of 896 |
| unit-start-power-4-sgd-0.3-length-readout | 5.03K ± 0.52K [4.06K, 6.08K] | 100.0% | 1.00 | -0.50 [-0.52, -0.47] | 0.92 | both | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 7 of 896 |
| unit-start-shrink-target-sgd-2-length-readout | 5.07K ± 0.33K [4.36K, 5.66K] | 100.0% | 1.00 | -0.50 [-0.53, -0.48] | 0.95 | both | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| shrink-target-sgd-1-length-readout | 6.20K ± 0.26K [5.67K, 6.69K] | 100.0% | 1.91 | -0.39 [-0.43, -0.35] | 0.72 | neither | learning rate 0.001, entropy 0.001, weight 2^2 | 32 | 2 of 896 |
| unit-start-shrink-target-sgd-3-length-readout | 6.25K ± 0.29K [5.65K, 6.77K] | 100.0% | 1.00 | -0.51 [-0.55, -0.49] | 0.95 | both | learning rate 0.001, entropy 0.001, weight 2^6 | 32 | 14 of 896 |
| bias-start-0.5 | 7.05K ± 0.66K [5.78K, 8.34K] | 100.0% | 1.04 | -2.08 [-2.52, -1.69] | 4.19 | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 448 |
| power-4-sgd-0.1-length-readout | 530.24K ± 368.18K [6.35K, 1,315.44K] | 87.5% | 1.91 | -0.34 [-0.38, -0.29] | 0.62 | neither | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 118 of 896 |

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| wide-input-layer-64 | **16.37K** ± 0.59K [15.26K, 17.56K] | 100.0% | 3.04 | -3.45 [-3.83, -3.10] | 3.73 | neither | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 448 |
| control | <u>18.74K</u> ± 0.96K [16.96K, 20.71K] | 100.0% | 4.23 | -2.72 [-3.08, -2.38] | 3.48 | neither | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| unit-start | 21.74K ± 1.24K [19.22K, 24.05K] | 100.0% | 1.00 | -4.46 [-4.97, -4.01] | 4.44 | even start | learning rate 0.001, entropy 0.001, weight 2^2 | 32 | 0 of 896 |
| initial-ratio-readout | 24.42K ± 1.69K [21.11K, 27.70K] | 100.0% | 1.00 | -2.66 [-3.00, -2.35] | 3.54 | even start | learning rate 0.001, entropy 0.01, weight 2^-4 | 32 | 0 of 448 |
| length-readout | 26.31K ± 3.48K [20.56K, 33.76K] | 100.0% | 2.06 | -1.30 [-1.48, -1.14] | 1.75 | neither | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| bias-start-0.5 | 28.09K ± 1.10K [25.89K, 30.13K] | 100.0% | 1.04 | -1.48 [-1.65, -1.31] | 1.89 | even start | learning rate 0.001, entropy 0.001, weight 2^0 | 32 | 0 of 448 |
| separate-adam-constant-learning-rate-1e-4 | 50.08K ± 4.21K [41.92K, 58.36K] | 100.0% | 4.23 | -1.41 [-1.65, -1.22] | 2.17 | neither | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-shrink-target-sgd-2-length-readout | 61.28K ± 8.14K [47.03K, 78.72K] | 100.0% | 1.00 | -0.50 [-0.54, -0.48] | 0.69 | both | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 14 of 896 |
| shrink-target-sgd-1-length-readout | 141.51K ± 20.69K [100.89K, 180.49K] | 100.0% | 2.06 | -0.30 [-0.33, -0.27] | 0.39 | neither | learning rate 0.0001, entropy 0.01, weight 2^-4 | 32 | 1 of 896 |
| unit-start-adagrad-0.1 | 231.51K ± 114.37K [69.08K, 483.51K] | 96.9% | 1.00 | -0.37 [-0.51, -0.25] | 0.94 | even start | learning rate 0.001, entropy 0.01, weight 2^-2 | 32 | 0 of 448 |
| unit-start-shrink-target-sgd-3-length-readout | 440.93K ± 173.79K [69.87K, 827.72K] | 90.6% | 1.00 | -0.52 [-0.56, -0.48] | 0.71 | both | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 161 of 896 |

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| separate-adam-constant-learning-rate-1e-4 | **48.86K** ± 1.68K [45.80K, 52.26K] | 100.0% | 4.23 | -1.33 [-1.62, -1.11] | 2.20 | neither | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| shrink-target-sgd-1-length-readout | <u>55.16K</u> ± 8.30K [41.23K, 73.36K] | 100.0% | 2.06 | -0.30 [-0.34, -0.26] | 0.39 | neither | learning rate 0.001, entropy 0.001, weight 2^6 | 32 | 0 of 896 |
| unit-start-shrink-target-sgd-2-length-readout | 56.99K ± 5.87K [44.61K, 67.49K] | 100.0% | 1.00 | -0.50 [-0.53, -0.47] | 0.65 | both | learning rate 0.001, entropy 0.01, weight 2^4 | 32 | 2 of 672 |
| unit-start-adagrad-0.1 | 58.40K ± 4.54K [49.64K, 67.01K] | 100.0% | 1.00 | -0.34 [-0.47, -0.23] | 1.00 | even start | learning rate 0.001, entropy 0.01, weight 2^2 | 32 | 0 of 448 |
| length-readout | 59.81K ± 8.31K [46.04K, 78.03K] | 100.0% | 2.06 | -0.95 [-1.20, -0.73] | 1.65 | neither | learning rate 0.001, entropy 0.001, weight 2^-2 | 32 | 0 of 672 |
| power-4-sgd-0.1-length-readout | 60.47K ± 7.22K [46.21K, 74.80K] | 100.0% | 2.06 | -0.26 [-0.30, -0.23] | 0.35 | neither | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 3 of 896 |
| control | 82.11K ± 12.00K [59.55K, 105.87K] | 100.0% | 4.23 | -0.93 [-1.17, -0.67] | 1.75 | neither | learning rate 0.0001, entropy 0.001, weight 2^-2 | 32 | 0 of 896 |
| unit-start-shrink-target-sgd-3-length-readout | 446.86K ± 249.72K [50.91K, 968.53K] | 90.6% | 1.00 | -0.52 [-0.56, -0.49] | 0.66 | both | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 179 of 896 |
| unit-start-power-4-sgd-0.3-length-readout | 482.36K ± 167.17K [214.41K, 846.20K] | 90.6% | 1.00 | -0.47 [-0.50, -0.44] | 0.64 | both | learning rate 0.0001, entropy 0.001, weight 2^6 | 32 | 200 of 896 |
| bias-start-0.5 | 1,085.52K ± 248.59K [639.55K, 1,598.99K] | 84.4% | 1.04 | -1.09 [-1.38, -0.78] | 3.04 | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 448 |
| wide-input-layer-64 | 1,130.86K ± 273.55K [588.73K, 1,676.47K] | 75.0% | 3.04 | -2.10 [-3.10, -1.25] | 4.42 | neither | learning rate 0.001, entropy 0.001, weight 2^4 | 32 | 0 of 448 |
| initial-ratio-readout | 1,144.45K ± 300.75K [566.51K, 1,705.11K] | 75.0% | 1.00 | -1.37 [-2.11, -0.90] | 4.04 | even start | learning rate 0.001, entropy 0.001, weight 2^2 | 32 | 0 of 224 |
| unit-start | 1,905.10K ± 292.49K [1,392.73K, 2,515.92K] | 56.2% | 1.00 | -1.38 [-2.88, -0.81] | 6.54 | even start | learning rate 0.0001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |

## screen 02: results

Budget per policy: 4,194,304 transitions, on the full 16,777,216-transition learning-rate schedule.

### straight grid, 32 cells, tabular-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-per-rollout-exponent-minus-2 | **7.56K** ± 0.61K [6.35K, 8.71K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1 | <u>7.73K</u> ± 0.47K [6.88K, 8.68K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-step-rate-0.5 | 8.08K ± 0.47K [7.15K, 8.96K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-rule-per-step-rate-1 | 8.42K ± 0.39K [7.65K, 9.16K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-step-rate-0.25 | 8.59K ± 0.60K [7.43K, 9.80K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-rate-0.5 | 9.00K ± 0.59K [7.91K, 10.16K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-rate-0.25 | 9.14K ± 0.82K [7.61K, 10.84K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-shrink-exponent-minus-1-sgd-1-length-readout | 9.45K ± 0.69K [8.14K, 10.84K] | 100.0% | 1.00 | -0.94 [-1.01, -0.88] | 1.48 | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-rate-1 | 9.70K ± 0.72K [8.28K, 11.10K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-rate-4 | 11.24K ± 1.42K [8.82K, 14.29K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| unit-start-exact-step-shrink-target | 15.02K ± 2.92K [10.09K, 21.06K] | 100.0% | 1.00 | -0.39 [-0.41, -0.36] | 0.71 | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-0.25 | 22.21K ± 3.16K [16.14K, 28.26K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.0001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-shrink-exponent-minus-0.25-sgd-4-length-readout | 22.53K ± 3.13K [16.39K, 28.41K] | 100.0% | 1.00 | -0.26 [-0.27, -0.25] | 0.45 | even start | learning rate 0.0001, entropy 0.01, weight 2^-6 | 32 | 3 of 896 |

### straight grid, 32 cells, coordinate-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-rule-per-step-rate-1 | **2.38K** ± 0.22K [1.97K, 2.80K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-step-rate-0.5 | <u>2.64K</u> ± 0.25K [2.11K, 3.10K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1 | 3.09K ± 0.42K [2.30K, 3.96K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-step-rate-0.25 | 3.23K ± 0.24K [2.77K, 3.69K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| count-per-rollout-rate-1 | 3.52K ± 0.32K [2.96K, 4.16K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-rate-0.5 | 3.54K ± 0.23K [3.07K, 3.98K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-rate-0.25 | 3.72K ± 0.23K [3.26K, 4.15K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-4 | 32 | 0 of 896 |
| count-per-rollout-rate-4 | 3.78K ± 0.43K [2.98K, 4.64K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-shrink-exponent-minus-1-sgd-1-length-readout | 4.27K ± 0.40K [3.49K, 5.03K] | 100.0% | 1.00 | -0.91 [-0.96, -0.86] | 1.51 | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-0.25 | 4.57K ± 0.43K [3.73K, 5.46K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-exact-step-shrink-target | 5.35K ± 0.39K [4.58K, 6.11K] | 100.0% | 1.00 | -0.40 [-0.42, -0.37] | 0.73 | even start | learning rate 0.001, entropy 0.01, weight 2^-4 | 32 | 0 of 896 |
| unit-start-shrink-target-sgd-1-length-readout | 5.53K ± 0.41K [4.77K, 6.38K] | 100.0% | 1.00 | -0.50 [-0.52, -0.47] | 0.91 | both | learning rate 0.001, entropy 0.001, weight 2^-4 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2 | 5.77K ± 0.29K [5.21K, 6.34K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| unit-start-shrink-exponent-minus-0.25-sgd-4-length-readout | 6.68K ± 0.66K [5.48K, 8.03K] | 100.0% | 1.00 | -0.26 [-0.28, -0.25] | 0.59 | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-per-rollout-exponent-minus-2 | **14.14K** ± 1.62K [11.06K, 17.44K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1 | <u>16.25K</u> ± 0.88K [14.61K, 18.01K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-shrink-exponent-minus-1-sgd-1-length-readout | 25.42K ± 1.91K [21.78K, 29.42K] | 100.0% | 1.00 | -0.97 [-1.04, -0.90] | 1.17 | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 672 |
| count-per-step-rate-0.25 | 29.11K ± 4.70K [20.40K, 38.56K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-4 | 32 | 0 of 896 |
| count-per-step-rate-0.5 | 29.85K ± 5.26K [21.08K, 40.98K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-rule-per-step-rate-1 | 32.25K ± 4.52K [24.17K, 41.73K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-4 | 32 | 0 of 896 |
| count-per-rollout-rate-0.5 | 37.68K ± 5.16K [27.18K, 47.66K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-rate-0.25 | 40.25K ± 5.54K [29.75K, 51.48K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^0 | 32 | 0 of 896 |
| count-per-rollout-rate-4 | 42.46K ± 7.34K [29.15K, 57.23K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-4 | 32 | 0 of 896 |
| count-per-rollout-rate-1 | 42.76K ± 6.32K [30.66K, 55.70K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-exact-step-shrink-target | 61.32K ± 8.49K [45.16K, 77.84K] | 100.0% | 1.00 | -0.34 [-0.36, -0.31] | 0.44 | even start | learning rate 0.0001, entropy 0.001, weight 2^-6 | 32 | 0 of 448 |
| unit-start-shrink-target-sgd-1-length-readout | 72.89K ± 6.63K [60.22K, 85.75K] | 100.0% | 1.00 | -0.48 [-0.50, -0.45] | 0.64 | both | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-0.25 | 227.16K ± 54.60K [123.95K, 337.95K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.0001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| unit-start-shrink-exponent-minus-0.25-sgd-4-length-readout | 1,056.37K ± 251.72K [562.58K, 1,532.53K] | 81.2% | 1.00 | -0.23 [-0.25, -0.20] | 0.83 | even start | learning rate 0.0001, entropy 0.001, weight 2^4 | 32 | 43 of 224 |

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-per-step-rate-0.25 | **47.08K** ± 4.07K [39.71K, 55.52K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| count-per-rollout-rate-0.5 | <u>51.24K</u> ± 6.55K [40.27K, 65.68K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| count-rule-per-step-rate-1 | 52.66K ± 7.83K [37.53K, 67.77K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^2 | 32 | 0 of 896 |
| unit-start-shrink-target-sgd-1-length-readout | 53.52K ± 6.04K [42.43K, 65.94K] | 100.0% | 1.00 | -0.48 [-0.51, -0.45] | 0.61 | both | learning rate 0.001, entropy 0.01, weight 2^4 | 32 | 0 of 896 |
| count-per-rollout-rate-4 | 53.58K ± 3.35K [47.26K, 60.29K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| count-per-step-rate-0.5 | 54.58K ± 8.40K [41.25K, 72.99K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^4 | 32 | 0 of 896 |
| count-per-rollout-rate-0.25 | 55.24K ± 3.96K [47.22K, 62.65K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-2 | 32 | 0 of 896 |
| count-per-rollout-rate-1 | 56.98K ± 4.40K [47.98K, 65.61K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-0.25 | 65.77K ± 8.00K [50.87K, 81.78K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| unit-start-exact-step-shrink-target | 91.28K ± 12.78K [69.26K, 117.98K] | 100.0% | 1.00 | -0.34 [-0.36, -0.31] | 0.47 | even start | learning rate 0.0001, entropy 0.001, weight 2^-4 | 32 | 0 of 224 |
| unit-start-shrink-exponent-minus-1-sgd-1-length-readout | 91.72K ± 12.36K [69.36K, 117.86K] | 100.0% | 1.00 | -0.90 [-0.96, -0.84] | 1.14 | even start | learning rate 0.0001, entropy 0.001, weight 2^0 | 32 | 0 of 224 |
| count-per-rollout-exponent-minus-1 | 116.86K ± 41.64K [57.87K, 208.98K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| unit-start-shrink-exponent-minus-0.25-sgd-4-length-readout | 490.33K ± 168.02K [204.30K, 851.33K] | 90.6% | 1.00 | -0.24 [-0.26, -0.21] | 0.43 | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 149 of 672 |
| count-per-rollout-exponent-minus-2 | 3,288.33K ± 281.54K [2,763.18K, 3,809.12K] | 25.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-4 | 32 | 0 of 896 |

## screen 03: results

Budget per policy: 4,194,304 transitions, on the full 16,777,216-transition learning-rate schedule.

### straight grid, 32 cells, tabular-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-per-rollout-exponent-minus-1-rate-0.25 | **7.08K** ± 0.37K [6.43K, 7.81K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-rate-4 | <u>7.18K</u> ± 0.42K [6.42K, 8.01K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1.5 | 7.21K ± 0.44K [6.38K, 8.07K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-0.75 | 8.67K ± 0.70K [7.33K, 10.04K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-plus-0.1-of-exponent-minus-0.5 | 8.84K ± 0.66K [7.61K, 10.16K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.03-of-exponent-minus-0.5 | 9.28K ± 0.79K [7.84K, 10.91K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-plus-0.3-of-exponent-minus-0.5 | 9.76K ± 1.17K [7.64K, 12.24K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.1-of-exponent-minus-0.5 | 21.04K ± 3.63K [14.81K, 28.82K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.0001, entropy 0.001, weight 2^-6 | 32 | 0 of 224 |

### straight grid, 32 cells, coordinate-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-per-step-exponent-minus-2 | **3.96K** ± 0.45K [3.13K, 4.90K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^2 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1.5 | <u>15.86K</u> ± 0.42K [15.00K, 16.66K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.0001, entropy 0.001, weight 2^-2 | 32 | 0 of 224 |

### snake grid, width 16, 2 turns (50 cells), tabular-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-per-rollout-exponent-minus-2-rate-0.25 | **12.75K** ± 0.77K [11.37K, 14.35K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1.5 | <u>12.94K</u> ± 1.13K [10.96K, 15.38K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-rate-4 | 15.21K ± 0.67K [13.94K, 16.55K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-rate-0.25 | 15.92K ± 0.79K [14.44K, 17.42K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-3 | 16.12K ± 1.38K [13.56K, 18.85K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 672 |
| count-per-rollout-exponent-minus-2-rate-4 | 16.62K ± 1.77K [13.36K, 20.27K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-2 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-0.75 | 18.46K ± 1.68K [15.29K, 21.73K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.03-of-exponent-minus-0.5 | 19.24K ± 1.72K [15.78K, 22.46K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-plus-0.1-of-exponent-minus-0.5 | 22.38K ± 1.42K [19.57K, 25.00K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-plus-0.3-of-exponent-minus-0.5 | 26.15K ± 2.15K [22.11K, 30.40K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.1-of-exponent-minus-0.5 | 27.87K ± 2.26K [23.36K, 32.27K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-2 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.3-of-exponent-minus-0.5 | 37.27K ± 4.14K [29.17K, 45.33K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.001, weight 2^-4 | 32 | 0 of 896 |

### snake grid, width 16, 2 turns (50 cells), coordinate-PPO

| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | copies with a non-finite bonus, all settings |
|---|---|---|---|---|---|---|---|---|---|
| count-per-step-exponent-minus-1 | **41.28K** ± 2.75K [35.65K, 46.63K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-plus-0.3-of-exponent-minus-0.5 | <u>56.78K</u> ± 3.81K [49.47K, 64.38K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^4 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-0.75 | 59.58K ± 7.18K [45.73K, 73.65K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^4 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-plus-0.1-of-exponent-minus-0.5 | 59.72K ± 6.96K [47.97K, 74.19K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.3-of-exponent-minus-0.5 | 61.95K ± 6.00K [50.60K, 74.50K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-rate-0.25 | 63.82K ± 9.67K [46.22K, 83.66K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^4 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.1-of-exponent-minus-0.5 | 73.25K ± 8.06K [59.40K, 90.55K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^4 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-plus-0.03-of-exponent-minus-0.5 | 109.01K ± 15.10K [78.31K, 137.86K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1-rate-4 | 319.64K ± 51.62K [218.40K, 421.69K] | 100.0% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^2 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-1.5 | 2,428.62K ± 324.54K [1,781.67K, 3,035.42K] | 56.2% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-rate-0.25 | 2,612.95K ± 249.39K [2,092.67K, 3,029.46K] | 46.9% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^0 | 32 | 0 of 896 |
| count-per-step-exponent-minus-2 | 2,839.29K ± 438.92K [1,988.88K, 3,676.73K] | 34.4% | 1.00 | n/a | n/a | even start | learning rate 0.0001, entropy 0.01, weight 2^-2 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-2-rate-4 | 3,212.16K ± 310.88K [2,576.87K, 3,815.48K] | 25.0% | 1.00 | n/a | n/a | even start | learning rate 0.0001, entropy 0.001, weight 2^-6 | 32 | 0 of 896 |
| count-per-rollout-exponent-minus-3 | 3,283.33K ± 248.84K [2,786.68K, 3,762.75K] | 28.1% | 1.00 | n/a | n/a | even start | learning rate 0.001, entropy 0.01, weight 2^-2 | 32 | 0 of 896 |


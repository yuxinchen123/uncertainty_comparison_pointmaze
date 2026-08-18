# Curve shape, all 66 configurations of this sweep

**Does every configuration rise then fall? No.**

| shape | configurations | share |
|---|---:|---:|
| `rise_then_fall` | 20 | 30% |
| `no_learning` | 16 | 24% |
| `still_rising` | 14 | 21% |
| `declining` | 13 | 20% |
| `rise_then_plateau` | 3 | 5% |

Label unchanged under all 9 threshold variants: 53 of 66.

## Best configuration that does NOT rise then fall, one per algorithm

| algorithm | learning rate | intrinsic weight | shape | score | rank overall |
|---|---:|---:|---|---:|---:|
| gt_position_velocity_sqrt | 1e-04 | 1e+02 | `rise_then_plateau` | 10.3914 | 11 |
| rnd_next_state | 1e-05 | 1e+01 | `still_rising` | 27.7777 | 6 |

The rank column is over every configuration of the sweep. A rank of 1 means this arm's best configuration overall is already the one named here, so the results table and the curve figure carry it and no separate table or plot repeats it.

## Thresholds

| setting | value |
|---|---:|
| `smooth_fraction` | 0.1 |
| `edge_fraction` | 0.05 |
| `noise_multiple` | 3.0 |
| `fall_fraction` | 0.05 |
| `trend_correlation` | 0.5 |
| `min_peak` | 0.001 |

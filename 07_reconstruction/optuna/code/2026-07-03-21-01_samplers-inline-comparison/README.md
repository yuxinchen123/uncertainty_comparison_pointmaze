# Inline four-sampler comparison for tutorial section 6.2

One script: RandomSampler / TPESampler / GPSampler / GridSampler at an equal 60-trial budget on a
noisy two-parameter landscape (peak at log10 ridge = -6, log10 beta = -2; noise SD 10). Written
2026-07-03 so the tutorial can show one fully self-contained comparison block with its observed
output.

Rerun: conda run -n exploration python compare_samplers.py

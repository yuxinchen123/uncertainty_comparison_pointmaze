| idea | copies (settings x copies) | seconds per iteration | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | total relative to baseline | setup seconds | setting 0's copies after 10 iterations, against the baseline |
|---|---|---|---|---|---|---|---|---|
| baseline: 1 setting x 128 copies, 1 program per card | 128 (1 x 128) | 0.0091 | 7.22 | 56.4 | 0.0049 | 1.00 | 55 | 3 of 128 identical; largest difference 2.8e-04 |
| 01 four settings in one program | 512 (4 x 128) | 0.0269 | 9.75 | 19.0 | 0.0146 | 1.35 | 68 | 0 of 128 identical; largest difference 1.7e-02 |
| 02 eight settings in one program | 1024 (8 x 128) | 0.0505 | 10.39 | 10.1 | 0.0274 | 1.44 | 84 | 0 of 128 identical; largest difference 1.7e-02 |
| 03 two programs per card | 256 (2 x 128) | 0.0192 | 6.83 | 26.7 | 0.0104 | 0.95 | 66 | 91 of 128 identical; largest difference 2.8e-04 |
| 04 four programs per card | 512 (4 x 128) | 0.0381 | 6.88 | 13.4 | 0.0207 | 0.95 | 85 | 1 of 128 identical; largest difference 3.3e-03 |
| 05 XLA command buffers off | 128 (1 x 128) | 0.0128 | 5.17 | 40.4 | 0.0069 | 0.72 | 60 | 91 of 128 identical; largest difference 2.8e-04 |
| 06 XLA command buffers incl. loops | 128 (1 x 128) | 0.0101 | 6.49 | 50.7 | 0.0055 | 0.90 | 58 | 26 of 128 identical; largest difference 2.8e-04 |
| 07 latency-hiding scheduler | 128 (1 x 128) | 0.0090 | 7.25 | 56.6 | 0.0049 | 1.00 | 59 | 0 of 128 identical; largest difference 2.9e-03 |
| 08 compilation cache, cold | 128 (1 x 128) | 0.0091 | 7.24 | 56.6 | 0.0049 | 1.00 | 57 | 5 of 128 identical; largest difference 2.3e-04 |
| 08 compilation cache, warm | 128 (1 x 128) | 0.0091 | 7.22 | 56.4 | 0.0049 | 1.00 | 7 | 5 of 128 identical; largest difference 2.3e-04 |
| 09 two processors instead of 8 | 128 (1 x 128) | 0.0091 | 7.22 | 56.4 | 0.0049 | 1.00 | 79 | 91 of 128 identical; largest difference 2.8e-04 |
| 10 study diagnostics on | 128 (1 x 128) | 0.0105 | 6.23 | 48.7 | 0.0057 | 0.86 | 50 | 0 of 128 identical; largest difference 2.9e-03 |

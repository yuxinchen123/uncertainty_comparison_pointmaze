# Bias initialization ablation on the untrained RND bonus field

One row per bias scheme. All values are means across seeds [111, 0, 1, 2, 3] except **ratio v0**, which shows mean ± sd. Metrics are computed over the free (non-wall) points of a 100×100 grid at velocity 0 unless noted.

Definitions (bonus $B=\tfrac12\lVert \text{predictor}-\text{target}\rVert^2$, over free points): ratio v0 $=B_{\max}/B_{\min}$; center bonus $=B$ at the free point with the smallest whitened input norm $\lVert\bar x\rVert$; max bonus v0 $=B_{\max}$; Spearman r = rank correlation of $B$ with $\lVert\bar x\rVert$; coeff. of variation $=\operatorname{std}(B)/\operatorname{mean}(B)$; velocity ratio $=B(v_x{=}5)/B(v_x{=}0)$ at the start cell; global ratio $=\max_{v\in\{0,2.5,5\}}B \,/\, \min_{v=0}B$.

Smaller ratio v0 = flatter field. **Bold** = smallest ratio v0, <u>underline</u> = second smallest.

| bias scheme | ratio v0<br>(mean ± sd) | center<br>bonus | max bonus<br>v0 | Spearman r<br>(B vs radius) | coeff. of<br>variation | velocity<br>ratio | global<br>ratio |
|---|---|---|---|---|---|---|---|
| zero | 2,864 ± 1,554 | 0.020 | 44.06 | 0.957 | 0.652 | 1.93 | 6,132 |
| pytorch_default | 5.21 ± 0.552 | 10.65 | 54.11 | 0.950 | 0.363 | 1.72 | 9.63 |
| normal_0.1 | 17.87 ± 2.83 | 2.62 | 45.76 | 0.957 | 0.547 | 1.84 | 37.19 |
| normal_0.25 | 3.75 ± 0.371 | 16.30 | 60.09 | 0.935 | 0.304 | 1.56 | 6.89 |
| normal_0.5 | 1.74 ± 0.123 | 65.13 | 111 | 0.861 | 0.124 | 1.25 | 2.57 |
| normal_1.0 | <u>1.22 ± 0.058</u> | 260 | 313 | 0.734 | 0.043 | 1.07 | 1.42 |
| normal_2.0 | **1.07 ± 0.024** | 1,041 | 1,106 | 0.525 | 0.015 | 1.02 | 1.12 |

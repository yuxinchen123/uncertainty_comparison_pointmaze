"""Numerical check of the Bergstra et al. (2011) TPE result: maximizing l(x)/g(x) maximizes
expected improvement (EI).

We fix a good set and a rest set, build l = KDE(good) and g = KDE(rest), then on a dense x-grid
compute (a) the ratio l(x)/g(x) and (b) EI(x) by integrating numerically over y under the
two-density model p(x|y<y*) = l, p(x|y>=y*) = g, P(y<y*) = gamma. We report the Spearman rank
correlation between the two, which the theorem predicts is exactly 1.

Minimization convention: 'good' = low y; improvement over the threshold y* is (y* - y).
"""

import numpy as np
from scipy import stats

GAMMA = 0.25  # P(y < y_star): the good fraction


def kde_pdf(x, data, bw):
    """Return a normalized 1-D Gaussian-KDE density at points x, equal-weight kernels at data."""
    # Average of Gaussian kernels centered at each data point; integrates to 1.
    z = (x[:, None] - data[None, :]) / bw
    kernels = np.exp(-0.5 * z**2) / (bw * np.sqrt(2 * np.pi))
    return kernels.mean(axis=1)


def main():
    """Build l and g, compute ratio and numerical EI on a grid, print the Spearman correlation."""
    # Fixed good set (clustered low-x) and rest set (spread), the two conditioning samples.
    good = np.array([1.6, 2.0, 2.1, 2.4, 3.0])
    rest = np.array([3.5, 4.5, 5.0, 6.0, 6.5, 7.5, 8.0])
    bw_good, bw_rest = 0.6, 1.0

    # Dense x-grid over the domain; l(x), g(x), and their ratio.
    xs = np.linspace(-1.0, 10.0, 400)
    l = kde_pdf(xs, good, bw_good)
    g = kde_pdf(xs, rest, bw_rest)
    ratio = l / g

    # Model p(y) as standard normal; y_star is its gamma-quantile so P(y<y_star)=gamma.
    y_star = stats.norm.ppf(GAMMA)
    ys = np.linspace(-6.0, 6.0, 2000)
    dy = ys[1] - ys[0]
    p_y = stats.norm.pdf(ys)
    below = ys < y_star  # region of y that counts as an improvement

    # EI(x) = integral over y<y_star of (y_star - y) * p(y|x) dy, computed numerically per x.
    ei = np.empty_like(xs)
    for i, x in enumerate(xs):
        # p(x|y) is l(x) for y<y_star and g(x) otherwise; p(x) is the mixture marginal.
        p_x_given_y = np.where(below, l[i], g[i])
        p_x = GAMMA * l[i] + (1 - GAMMA) * g[i]
        p_y_given_x = p_x_given_y * p_y / p_x
        # Numerically integrate the improvement (y_star - y) over the below-threshold region.
        ei[i] = np.sum((y_star - ys[below]) * p_y_given_x[below]) * dy

    # The theorem: EI(x) and l(x)/g(x) induce the same ordering, so Spearman = 1.
    rho, _ = stats.spearmanr(ei, ratio)
    pearson_log = np.corrcoef(np.log(ei), np.log(ratio))[0, 1]
    print("optuna-independent numerical check (no optuna import needed here)")
    print(f"gamma = P(y<y*) = {GAMMA}, y_star = {y_star:.4f}")
    print(f"grid points: {len(xs)}  (x in [-1,10]),  y-integration points: {len(ys)}")
    print(f"\nSpearman rank correlation  EI(x) vs l(x)/g(x): {rho:.10f}")
    print(f"Pearson corr of log EI vs log ratio         : {pearson_log:.6f}")

    # Closed form for cross-check: EI(x) proportional to 1 / (gamma + (1-gamma) g/l).
    closed = 1.0 / (GAMMA + (1 - GAMMA) * (g / l))
    rho_closed, _ = stats.spearmanr(ei, closed)
    print(f"Spearman EI(x) vs closed form 1/(g+(1-g)g/l): {rho_closed:.10f}")

    # Show a few grid rows sorted by ratio so the monotone agreement is visible.
    order = np.argsort(ratio)
    idx = order[:: len(order) // 8][:8]
    print("\n  x        l/g          EI(x)        (both rise together)")
    for i in sorted(idx, key=lambda k: ratio[k]):
        print(f"  {xs[i]:6.2f}  {ratio[i]:11.5e}  {ei[i]:11.5e}")

    # The argmax location must coincide for the two criteria.
    print(f"\nargmax x by l/g  : {xs[np.argmax(ratio)]:.4f}")
    print(f"argmax x by EI(x): {xs[np.argmax(ei)]:.4f}")


if __name__ == "__main__":
    main()

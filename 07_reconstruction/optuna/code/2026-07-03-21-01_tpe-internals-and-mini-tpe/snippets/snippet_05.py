# Numerical check: maximizing l(x)/g(x) maximizes expected improvement (Bergstra et al. 2011)
import numpy as np
from scipy import stats

GAMMA = 0.25                                    # P(y < y_star): the "good" fraction

def kde(x, data, bw):                            # normalized 1-D Gaussian KDE density
    z = (x[:, None] - data[None, :]) / bw
    return (np.exp(-0.5 * z**2) / (bw * np.sqrt(2 * np.pi))).mean(1)

good = np.array([1.6, 2.0, 2.1, 2.4, 3.0])       # fixed good set (low y) and rest set
rest = np.array([3.5, 4.5, 5.0, 6.0, 6.5, 7.5, 8.0])
xs = np.linspace(-1, 10, 400)
l, g = kde(xs, good, 0.6), kde(xs, rest, 1.0)
ratio = l / g

# minimization convention: model p(y)=N(0,1); y_star is its gamma-quantile so P(y<y*)=gamma
y_star = stats.norm.ppf(GAMMA)
ys = np.linspace(-6, 6, 2000); dy = ys[1] - ys[0]
p_y = stats.norm.pdf(ys); below = ys < y_star

ei = np.empty_like(xs)                            # EI(x) integrated numerically over y
for i in range(len(xs)):
    p_x_given_y = np.where(below, l[i], g[i])     # p(x|y): l below threshold, g above
    p_x = GAMMA * l[i] + (1 - GAMMA) * g[i]
    p_y_given_x = p_x_given_y * p_y / p_x
    ei[i] = np.sum((y_star - ys[below]) * p_y_given_x[below]) * dy

rho, _ = stats.spearmanr(ei, ratio)
print(f"Spearman rank correlation  EI(x) vs l(x)/g(x): {rho:.10f}")
print(f"argmax x by l/g : {xs[np.argmax(ratio)]:.4f}   argmax x by EI: {xs[np.argmax(ei)]:.4f}")

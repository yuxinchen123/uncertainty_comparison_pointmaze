import numpy as np, sys
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in sorted(a.files):
    x, y = a[k].astype(float), b[k].astype(float)
    d = np.abs(x - y)
    print(f"{k:55s} shape {str(x.shape):16s} max|x| {np.abs(x).max():.3e} maxdiff {d.max():.3e} median rel {np.median(d/(np.abs(x)+1e-12)):.1e}")

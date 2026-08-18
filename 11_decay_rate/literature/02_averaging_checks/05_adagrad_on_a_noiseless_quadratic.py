import numpy as np
rng=np.random.default_rng(0)
d=30
lam=np.logspace(0,-2,d)           # curvature spread 100x
e0=np.ones(d)                     # equal initial error in every eigendirection
def run(kind, T=200000, eta=0.1, eps=1e-12, noise=0.0):
    e=e0.copy(); acc=np.zeros(d); out=[]
    for t in range(1,T+1):
        g=lam*e + (noise*rng.normal(size=d) if noise else 0.0)
        if kind=="fullmatrix_or_diag_in_eigenbasis":
            acc+=g*g; e=e-eta*g/(np.sqrt(acc)+eps)
        elif kind=="gd":
            e=e-eta/lam.max()*g*lam.max()/lam.max()  # placeholder
        out.append(np.abs(e).copy())
    return np.array(out)
T=200000
tr=run("fullmatrix_or_diag_in_eigenbasis",T=T)
ns=np.array([10,100,1000,10000,100000,200000])
print("AdaGrad (diagonal in the eigenbasis = full-matrix AdaGrad on a quadratic), NO gradient noise")
print("  |e_k(n)| for the largest / median / smallest curvature direction:")
for n in ns:
    v=tr[n-1]; print("   n=%7d  lam_max dir %.3e   median %.3e   lam_min dir %.3e" % (n, v[0], v[d//2], v[-1]))
lo=T//2
sl=np.array([np.polyfit(np.log(np.arange(lo,T)+1), np.log(np.maximum(tr[lo:,k],1e-300)),1)[0] for k in range(d)])
print("  log-log slope over the last half: min %+.2f max %+.2f  (a straight power law would be flat)" % (sl.min(), sl.max()))
print("  accumulator sqrt(S_k) at the end / at n=1000, per direction: %s" % np.round(
    np.sqrt((( (lam*tr)**2).cumsum(0))[-1]/(( (lam*tr)**2).cumsum(0))[999]),3)[:5])

trn=run("fullmatrix_or_diag_in_eigenbasis",T=20000,noise=0.05)
print("\nsame AdaGrad WITH gradient noise 0.05:")
for n in [100,1000,10000,20000]:
    v=trn[n-1]; print("   n=%6d  lam_max dir %.3e median %.3e lam_min dir %.3e" % (n,v[0],v[d//2],v[-1]))

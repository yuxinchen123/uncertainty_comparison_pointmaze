import numpy as np, torch
torch.set_default_dtype(torch.float64)

def slopes(B, lo, hi):
    n = np.arange(1, B.shape[0]+1); m=(n>=lo)&(n<=hi); ln=np.log(n[m])
    return np.array([np.polyfit(ln, np.log(np.maximum(B[m,i],1e-300)),1)[0] for i in range(B.shape[1])])

M, DIN, H, DOUT = 40, 4, 64, 8
g = torch.Generator().manual_seed(1)
X = torch.zeros(M, DIN); X[:, :2] = torch.rand(M,2,generator=g)*2-1

def init(seed):
    gg = torch.Generator().manual_seed(seed)
    W1 = torch.randn(H,DIN,generator=gg)/np.sqrt(DIN); b1 = torch.zeros(H)
    W2 = torch.randn(DOUT,H,generator=gg)/np.sqrt(H);  b2 = torch.zeros(DOUT)
    return [W1,b1,W2,b2]

def fwd(p, X):
    W1,b1,W2,b2 = p
    Z = X@W1.t()+b1; Hh = torch.relu(Z); S = (Z>0).to(X.dtype)
    return Hh@W2.t()+b2, Hh, S

Wt = init(321); Y,_,_ = fwd(Wt, X)
P = sum(q.numel() for q in init(9))
print(f"full-network GN test: {M} positions x {DOUT} outputs = {M*DOUT} constraints, {P} parameters")

def gn_run(lam, steps=1500, seed=9, eta_fn=lambda t: 1.0/(2*t)):
    p = init(seed); W1,b1,W2,b2 = p
    Kxx = X@X.t()+1.0
    B=[]
    for t in range(1, steps+1):
        O, Hh, S = fwd([W1,b1,W2,b2], X)
        R = O - Y
        B.append(R.norm(dim=1).numpy())
        Khh = Hh@Hh.t()+1.0                                   # M x M
        T = torch.einsum('kj,lj,ij,mj->imkl', W2, W2, S, S)    # M,M,DOUT,DOUT
        G = (Kxx[:,:,None,None]*T)
        G = G + Khh[:,:,None,None]*torch.eye(DOUT)[None,None,:,:]
        G = G.permute(0,2,1,3).reshape(M*DOUT, M*DOUT)
        v = torch.linalg.solve(G + lam*torch.eye(M*DOUT), R.reshape(-1)).reshape(M, DOUT)
        eta = eta_fn(t)
        U = (v@W2)*S
        W2 = W2 - eta*(v.t()@Hh); b2 = b2 - eta*v.sum(0)
        W1 = W1 - eta*(U.t()@X);  b1 = b1 - eta*U.sum(0)
    return np.array(B)

for lam, tag in [(1e-12,"lam=1e-12"), (1e-6,"lam=1e-6"), (1e-2,"lam=1e-2"), (1.0,"lam=1")]:
    B = gn_run(lam)
    s = slopes(B, 200, 1500)
    print(f"  {tag:10s}: per-position slope mean {s.mean():+.4f} sd {s.std():.4f} range [{s.min():+.4f},{s.max():+.4f}]  final bonus range [{B[-1].min():.2e},{B[-1].max():.2e}]")

# plain gradient descent (lam = infinity limit -> eta*R): compare
def gd_run(lr_fn, steps=1500, seed=9):
    W1,b1,W2,b2 = init(seed); B=[]
    for t in range(1, steps+1):
        O,Hh,S = fwd([W1,b1,W2,b2], X); R = O-Y
        B.append(R.norm(dim=1).numpy())
        lr = lr_fn(t)
        U = (R@W2)*S
        W2 = W2 - lr*(R.t()@Hh); b2 = b2 - lr*R.sum(0)
        W1 = W1 - lr*(U.t()@X);  b1 = b1 - lr*U.sum(0)
    return np.array(B)
B = gd_run(lambda t: min(0.5/t, 0.02)); s = slopes(B,200,1500)
print(f"  plain GD 0.5/t : per-position slope mean {s.mean():+.4f} sd {s.std():.4f} range [{s.min():+.4f},{s.max():+.4f}]")

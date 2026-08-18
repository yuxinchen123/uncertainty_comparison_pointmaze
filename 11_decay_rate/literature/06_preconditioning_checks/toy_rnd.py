import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

M, DIN, H, DOUT = 100, 4, 256, 128
X = torch.zeros(M, DIN)
X[:, :2] = torch.rand(M, 2) * 2 - 1          # 2-D maze positions in [-1,1]^2

def mk(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(DIN, H), nn.ReLU(), nn.Linear(H, DOUT))

target = mk(123)
for p in target.parameters(): p.requires_grad_(False)
Y = target(X).detach()                        # M x DOUT

def slopes(B, n_lo, n_hi):
    """per-position log-log slope of bonus over [n_lo, n_hi]; B is (steps, M)."""
    n = np.arange(1, B.shape[0]+1)
    m = (n >= n_lo) & (n <= n_hi)
    ln = np.log(n[m]); out = []
    for i in range(B.shape[1]):
        y = np.log(np.maximum(B[m, i], 1e-30))
        out.append(np.polyfit(ln, y, 1)[0])
    return np.array(out)

STEPS = 3000

# ---------- (a) plain Adam, constant lr, full batch ----------
pred = mk(7)
opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
Ba = []
for t in range(1, STEPS+1):
    R = pred(X) - Y
    loss = (R**2).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    Ba.append(R.detach().norm(dim=1).numpy())
Ba = np.array(Ba)
sa = slopes(Ba, 300, 3000)
print(f"(a) Adam const lr   : aggregate slope {np.polyfit(np.log(np.arange(300,3001)), np.log(Ba[299:,:].mean(1)),1)[0]:+.3f} "
      f"| per-position slope mean {sa.mean():+.3f} sd {sa.std():.3f} min {sa.min():+.3f} max {sa.max():+.3f}")

# ---------- (b) plain SGD with 1/t learning rate ----------
pred = mk(7)
Bb = []
params = list(pred.parameters())
for t in range(1, STEPS+1):
    R = pred(X) - Y
    loss = 0.5*(R**2).sum()
    for p in params:
        if p.grad is not None: p.grad = None
    loss.backward()
    lr = 2.0/t
    with torch.no_grad():
        for p in params: p -= lr * p.grad
    Bb.append(R.detach().norm(dim=1).numpy())
Bb = np.array(Bb)
sb = slopes(Bb, 300, 3000)
print(f"(b) SGD 1/t lr      : aggregate slope {np.polyfit(np.log(np.arange(300,3001)), np.log(Bb[299:,:].mean(1)),1)[0]:+.3f} "
      f"| per-position slope mean {sb.mean():+.3f} sd {sb.std():.3f} min {sb.min():+.3f} max {sb.max():+.3f}")

# ---------- (c) last-layer exact prescribed decay ----------
# g(x) = W2 h(x) + b2 with h from a FROZEN random first layer.
pred = mk(7)
with torch.no_grad():
    Hfeat = torch.relu(pred[0](X))                     # M x H
    Hb = torch.cat([Hfeat, torch.ones(M,1)], 1)        # M x (H+1), absorb bias
    W = torch.cat([pred[2].weight.t(), pred[2].bias[None,:]], 0)   # (H+1) x DOUT
    Hpinv = torch.linalg.pinv(Hb)                      # (H+1) x M
    # step 0: exact solve so every residual has norm exactly 1
    R = Hb @ W - Y
    Rt = R / R.norm(dim=1, keepdim=True)               # target residual, unit norm each row
    W = W + Hpinv @ (Rt - R)
    Bc = []
    for t in range(1, STEPS+1):
        R = Hb @ W - Y
        Bc.append(R.norm(dim=1).numpy())
        eta = 1.0/(2*t)
        W = W + Hpinv @ (-eta * R)                     # prescribe dR = -eta R exactly
Bc = np.array(Bc)
sc = slopes(Bc, 300, 3000)
print(f"(c) last-layer exact: aggregate slope {np.polyfit(np.log(np.arange(300,3001)), np.log(Bc[299:,:].mean(1)),1)[0]:+.3f} "
      f"| per-position slope mean {sc.mean():+.3f} sd {sc.std():.6f} min {sc.min():+.3f} max {sc.max():+.3f}")
print(f"    bonus at n=1: min {Bc[0].min():.6f} max {Bc[0].max():.6f}; at n=1000: min {Bc[999].min():.6e} max {Bc[999].max():.6e}")
print(f"    theory (pi n)^-1/2 at n=1000: {1/np.sqrt(np.pi*1000):.6e}")

# ---------- (d) non-uniform visitation: per-position count schedule ----------
rng = np.random.default_rng(0)
p = rng.dirichlet(np.ones(M)*0.5)                      # very non-uniform visit probabilities
pred = mk(7)
with torch.no_grad():
    Hfeat = torch.relu(pred[0](X)); Hb = torch.cat([Hfeat, torch.ones(M,1)],1)
    W = torch.cat([pred[2].weight.t(), pred[2].bias[None,:]],0)
    Hpinv = torch.linalg.pinv(Hb)
    R = Hb @ W - Y; Rt = R / R.norm(dim=1, keepdim=True)
    W = W + Hpinv @ (Rt - R)
    counts = np.zeros(M)
    Bd = np.zeros((STEPS, M)); Cd = np.zeros((STEPS, M))
    for t in range(1, STEPS+1):
        R = Hb @ W - Y
        Bd[t-1] = R.norm(dim=1).numpy(); Cd[t-1] = counts
        # one "visit" per step: sample a batch of 10 positions with replacement
        idx = rng.choice(M, size=10, replace=True, p=p)
        eta = np.zeros(M)
        for i in idx:
            counts[i] += 1
            eta[i] = 1.0/(2*counts[i])      # per-position 1/n_i step (last visit wins for duplicates)
        E = torch.tensor(eta, dtype=torch.float32)[:, None]
        W = W + Hpinv @ (-E * R)
# per-position slope of bonus against its OWN count
ok = Cd[-1] >= 30
sl = []
for i in np.where(ok)[0]:
    m = Cd[:, i] >= 5
    sl.append(np.polyfit(np.log(Cd[m, i]), np.log(np.maximum(Bd[m, i],1e-30)), 1)[0])
sl = np.array(sl)
print(f"(d) non-uniform, per-position 1/n_i: {ok.sum()} positions with >=30 visits; "
      f"slope vs OWN count: mean {sl.mean():+.4f} sd {sl.std():.4f} min {sl.min():+.4f} max {sl.max():+.4f}")

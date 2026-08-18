import torch, numpy as np
torch.manual_seed(0); torch.set_num_threads(4)
def net(): return torch.nn.Sequential(torch.nn.Linear(4,256), torch.nn.ReLU(), torch.nn.Linear(256,128))
gx,gy=torch.meshgrid(torch.linspace(-1,1,10),torch.linspace(-1,1,10),indexing='ij')
X=torch.stack([gx.reshape(-1),gy.reshape(-1),torch.zeros(100),torch.zeros(100)],1)
f=net(); [p.requires_grad_(False) for p in f.parameters()]
with torch.no_grad(): F=f(X)
def bonus(g):
    with torch.no_grad(): return (g(X)-F).norm(dim=1)
g=net(); print("initial bonus: mean %.4f min %.4f max %.4f" % (bonus(g).mean(), bonus(g).min(), bonus(g).max()))
for lr in [0.05,0.2,1.0]:
    g2=net(); avg=[p.detach().clone() for p in g2.parameters()]; ga=net()
    out=[]
    for n in range(1,5001):
        loss=((g2(X)-F)**2).sum(1).mean()
        gr=torch.autograd.grad(loss,list(g2.parameters()))
        with torch.no_grad():
            for p,gg in zip(g2.parameters(),gr): p-=lr*gg
            for a,p in zip(avg,g2.parameters()): a+=(p-a)/n
        if n in (10,100,1000,5000):
            with torch.no_grad():
                for q,a in zip(ga.parameters(),avg): q.copy_(a)
            out.append((n, float(bonus(g2).mean()), float(bonus(ga).mean())))
    print("lr=%.2f " % lr, " ".join("n=%d last %.3e avg %.3e"%o for o in out))

"""Measures torch.backends.cudnn.deterministic (the script's default is True) and the
uint8-vs-float32 observation storage path."""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ppo_rnd import Agent, RNDModel, NUM_ENVS, NUM_STEPS, BATCH, MINIBATCH, N_ACTIONS, timeit, wallclock
from bench_isolate import build, make_update

dev = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
out = {"gpu": p.name, "cap": f"sm_{p.major}{p.minor}", "host": os.uname().nodename, "res": {}}
print(json.dumps({k: v for k, v in out.items() if k != "res"}), flush=True)

def rec(k, v):
    out["res"][k] = round(v, 3); print(f"  {k:62s} {v:9.3f} ms", flush=True)

# ---- cudnn.deterministic (script sets it True by default via --torch_deterministic) -----------
for det, bm, mtf in [(True, False, False), (False, False, False), (True, True, False),
                     (False, True, False), (True, True, True), (False, True, True)]:
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = bm
    torch.backends.cuda.matmul.allow_tf32 = mtf
    torch.backends.cudnn.allow_tf32 = True          # PyTorch default, what the script actually runs with
    d = build(dev)
    opt = optim.Adam(d["combined"], lr=1e-4, eps=1e-5)
    rec(f"update deterministic={det} benchmark={bm} matmul_tf32={mtf}",
        timeit(make_update(d, opt, dev=dev), 8, 25))
    del d, opt; torch.cuda.empty_cache()

# ---- uint8 vs float32 observation storage ----------------------------------------------------
torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True
agent = Agent().to(dev)
obs_f32 = torch.zeros((BATCH, 4, 84, 84), device=dev)
out["obs_f32_GB"] = round(obs_f32.numel() * 4 / 1e9, 3)
idx = torch.from_numpy(np.random.permutation(BATCH)[:MINIBATCH]).to(dev)

def gather_f32():
    """Advanced-index gather of one minibatch out of the float32 rollout buffer, then the trunk."""
    with torch.no_grad():
        return agent.network(obs_f32[idx] / 255.0)

rec("minibatch gather+trunk, float32 obs buffer", timeit(gather_f32, 8, 30))
peak_f32 = torch.cuda.max_memory_allocated() / 1e9
del obs_f32; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

obs_u8 = torch.zeros((BATCH, 4, 84, 84), dtype=torch.uint8, device=dev)
out["obs_u8_GB"] = round(obs_u8.numel() / 1e9, 3)

def gather_u8():
    """Same gather from a uint8 buffer; uint8/255.0 gives bit-identical float32 values."""
    with torch.no_grad():
        return agent.network(obs_u8[idx] / 255.0)

rec("minibatch gather+trunk, uint8 obs buffer", timeit(gather_u8, 8, 30))
out["peak_GB_f32_path"] = round(peak_f32, 2)
out["peak_GB_u8_path"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

# bit-identical check on real data
a = torch.randint(0, 256, (256, 4, 84, 84), dtype=torch.uint8, device=dev)
lhs = agent.network(a.float() / 255.0)
rhs = agent.network(a / 255.0)
out["uint8_path_bit_identical"] = bool(torch.equal(lhs, rhs))
print("  uint8/255 == float32/255 exactly:", out["uint8_path_bit_identical"], flush=True)

# ---- merged rollout policy: bit-identical check ----------------------------------------------
x = torch.rand(NUM_ENVS, 4, 84, 84, device=dev) * 255
with torch.no_grad():
    torch.manual_seed(7); ve1, vi1 = agent.get_value(x); a1, lp1, _, _, _ = agent.get_action_and_value(x)
    torch.manual_seed(7); a2, lp2, ve2, vi2 = agent.get_action_and_value_fused(x)
out["merged_policy_bit_identical"] = dict(
    value_ext=bool(torch.equal(ve1, ve2)), value_int=bool(torch.equal(vi1, vi2)),
    action=bool(torch.equal(a1, a2)), logprob=bool(torch.equal(lp1, lp2)))
print("  merged rollout identical:", out["merged_policy_bit_identical"], flush=True)

print("=== RESULT JSON ==="); print(json.dumps(out, indent=2), flush=True)
with open(sys.argv[1], "w") as f: json.dump(out, f, indent=2)

"""Batched multi-copy PPO+RND in PyTorch — implements ../research/ppo_rnd_algorithm_spec.md.

C independent training copies (own networks, own envs, own running statistics, own optimizer
moments) advance in lockstep as one batched computation. Every parameter carries a leading copy
axis and every forward is a batched GEMM (baddbmm). All reductions keep the copy axis; the
scalar loss is the SUM over copies. The two update styles are two separate functions with no
data-dependent branch inside (spec section 12.1).
"""
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "pointmaze" / "common"))
sys.path.insert(0, str(BASE / "pointmaze" / "torch_env"))
from pm_common import EnvConfig  # noqa: E402
from torch_pointmaze import TorchPointMaze  # noqa: E402

LOG2PI = 1.8378770664093453


@dataclass(frozen=True)
class PPOConfig:
    """All trainer knobs (spec section 15). Style is resolved at build time, never branched on."""
    n_copies: int = 8
    n_envs: int = 4                  # N; T*N = 512 rows per copy per iteration
    num_steps: int = 128             # T
    update_style: str = "epoch_minibatch"  # or "full_batch"
    update_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    anneal_lr: bool = True
    gamma_ext: float = 0.999
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    int_coef: float = 1.0
    ext_coef: float = 2.0
    rnd_feature_dim: int = 128
    rnd_hidden: int = 256
    obs_norm_init_iters: int = 10
    bootstrap_on_truncation: bool = True
    base_seed: int = 0
    rollout_mode: str = "eager"      # "eager" | "compile-step" | "capture"
                                     # capture = whole-rollout CUDA-graph replay of the
                                     # compiled per-step function (fused kernels, no python)
    fused_adam: bool = False         # fused CUDA Adam kernel (GPU runs)
    capture_update: bool = False     # CUDA-graph the whole update phase (implies capturable Adam)
    one_graph: bool = False          # capture rollout+post+update as ONE graph per iteration
                                     # (requires rollout_mode="capture" semantics and
                                     # capture_update=True; overrides both)
    tf32: bool = False               # TF32 tensor-core matmuls (faster, ~1e-3 relative rounding)
    env_backend: str = "torch"       # "torch" (env fused into the compiled step) or "cuda"
                                     # (the single fused kernel from pointmaze/cuda_env)


def _substream_seed(*parts) -> int:
    """Keyed 32-bit seed per rng-seeding rule: sha256 of '::'-joined identifiers."""
    key = "::".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def _orthogonal(rows, cols, gain, gen):
    """Orthogonal matrix [rows, cols] like nn.init.orthogonal_ (QR with sign fix)."""
    flat = torch.randn(max(rows, cols), min(rows, cols), generator=gen)
    q, r = torch.linalg.qr(flat)
    q = q * torch.sign(torch.diagonal(r))
    if rows < cols:
        q = q.T
    return (gain * q)[:rows, :cols].contiguous()


class BatchedRMS:
    """Per-copy running mean/var (gymnasium parallel-variance formula), float64 (spec 4.2).

    Shapes: mean/var [C, dim] (dim may be 1 for the scalar intrinsic statistic), count [C, 1].
    """

    def __init__(self, n_copies, dim, device):
        self.mean = torch.zeros(n_copies, dim, dtype=torch.float64, device=device)
        self.var = torch.ones(n_copies, dim, dtype=torch.float64, device=device)
        self.count = torch.full((n_copies, 1), 1e-4, dtype=torch.float64, device=device)

    def update(self, batch):
        """batch [C, B, dim] float; population (ddof=0) batch variance per copy.

        Writes are IN-PLACE so consumers inside a captured CUDA graph keep reading the same
        storages across iterations.
        """
        b = batch.double()
        batch_mean = b.mean(dim=1)
        batch_var = b.var(dim=1, unbiased=False)
        batch_count = float(b.shape[1])
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / tot
        self.mean.add_(delta * batch_count / tot)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)


class PPORND:
    """The batched trainer. Parameters live in dicts of [C, ...] tensors."""

    def __init__(self, cfg: PPOConfig, env_cfg: EnvConfig = None, device="cuda"):
        # one_graph captures the update phase inside the iteration graph, so it REQUIRES
        # the capturable-Adam machinery; force the flag so the pairing cannot be missed
        # (a frozen python step counter inside a captured graph corrupts bias correction)
        if cfg.one_graph and not cfg.capture_update:
            object.__setattr__(cfg, "capture_update", True)
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.tf32:
            torch.set_float32_matmul_precision("high")
        if cfg.env_backend == "cuda":
            sys.path.insert(0, str(BASE / "pointmaze" / "cuda_env"))
            from cuda_pointmaze import CudaPointMaze
            self.env = CudaPointMaze(env_cfg or EnvConfig(), cfg.n_copies, cfg.n_envs,
                                     device=device, base_seed=cfg.base_seed)
        else:
            self.env = TorchPointMaze(env_cfg or EnvConfig(), cfg.n_copies, cfg.n_envs,
                                      device=device, base_seed=cfg.base_seed)
        C = cfg.n_copies
        F, Hh = cfg.rnd_feature_dim, cfg.rnd_hidden

        def stack(net, layers):
            # per-copy keyed init: copy i's weights identical whether C=8 or C=128
            out = {}
            for li, (rows, cols, gain) in enumerate(layers):
                ws, bs = [], []
                for c in range(C):
                    g = torch.Generator().manual_seed(_substream_seed(cfg.base_seed, c, net, li))
                    # stored [in, out] for x @ W; init orthogonal on [out, in] like nn.Linear
                    ws.append(_orthogonal(rows, cols, gain, g).T)
                    bs.append(torch.zeros(rows))
                out[f"W{li}"] = torch.stack(ws).to(self.device)
                out[f"b{li}"] = torch.stack(bs).to(self.device)
            return out

        # actor 4-64-64-(2 mean); critic 4-64-64-(1 ext + 1 int); rnd target 4-H-F frozen,
        # predictor 4-H-F-F trainable (spec section 3). layers are (out_dim, in_dim, gain)
        self.actor = stack("actor", [(64, 4, 2 ** 0.5), (64, 64, 2 ** 0.5), (2, 64, 0.01)])
        self.actor["logstd"] = torch.zeros(C, 2, device=self.device)
        self.critic = stack("critic", [(64, 4, 2 ** 0.5), (64, 64, 2 ** 0.5)])
        heads = stack("critic_heads", [(1, 64, 1.0), (1, 64, 1.0)])
        self.critic["Wext"], self.critic["bext"] = heads["W0"], heads["b0"]
        self.critic["Wint"], self.critic["bint"] = heads["W1"], heads["b1"]
        self.target = stack("rnd_target", [(Hh, 4, 2 ** 0.5), (F, Hh, 2 ** 0.5)])
        self.predictor = stack("rnd_predictor",
                               [(Hh, 4, 2 ** 0.5), (F, Hh, 2 ** 0.5), (F, F, 2 ** 0.5)])

        self.trainable = ([self.actor[k] for k in ["W0", "b0", "W1", "b1", "W2", "b2", "logstd"]]
                          + [self.critic[k] for k in ["W0", "b0", "W1", "b1",
                                                      "Wext", "bext", "Wint", "bint"]]
                          + [self.predictor[k] for k in ["W0", "b0", "W1", "b1", "W2", "b2"]])
        for p in self.trainable:
            p.requires_grad_(True)
        self._lr_t = torch.tensor(cfg.learning_rate, device=self.device)
        if cfg.capture_update:
            # capturable: Adam's step count and lr live on the GPU so the captured graph
            # replays correct bias correction and annealed lr
            self.opt = torch.optim.Adam(self.trainable, lr=self._lr_t, eps=cfg.adam_eps,
                                        fused=cfg.fused_adam, capturable=True)
        else:
            self.opt = torch.optim.Adam(self.trainable, lr=cfg.learning_rate,
                                        eps=cfg.adam_eps, fused=cfg.fused_adam)

        # per-copy running statistics (spec 4.2, 5.2) and the intrinsic forward filter
        self.obs_rms = BatchedRMS(C, 4, self.device)
        self.int_rms = BatchedRMS(C, 1, self.device)
        self.int_filter = torch.zeros(C, cfg.n_envs, device=self.device)

        self.env.reset()
        self.global_step = 0

        # persistent rollout storage: entry-state buffers, the pre-drawn action noise, and
        # the per-step buffers. One replay of the captured graph (or one eager pass of
        # _rollout_body) overwrites all of them.
        T, N = cfg.num_steps, cfg.n_envs
        dev = self.device
        self._S_pos = self.env.pos.clone()
        self._S_vel = self.env.vel.clone()
        self._S_goal = self.env.goal.clone()
        self._S_sc = self.env.step_count.clone()
        self._S_rc = self.env.reset_count.clone()
        self._S_obs = torch.cat([self.env.pos, self.env.vel], -1)
        self._Z = torch.zeros(T, C, N, 2, device=dev)
        self._bufs = {k: torch.zeros(T, C, N, d, device=dev).squeeze(-1) if d == 0
                      else torch.zeros(T, C, N, d, device=dev)
                      for k, d in [("obs", 4), ("nobs", 4), ("act", 2)]}
        for k in ["logp", "rext", "rint", "term", "done", "vext", "vint"]:
            self._bufs[k] = torch.zeros(T, C, N, device=dev)
        self._graph = None
        # static flattened-batch buffers, filled IN PLACE by _post_body each iteration and
        # consumed by every update path (the update graph reads them directly, no copies)
        B = T * N
        self._U = {
            "obs": torch.zeros(C, B, 4, device=dev), "actions": torch.zeros(C, B, 2, device=dev),
            "old_logprob": torch.zeros(C, B, device=dev), "adv": torch.zeros(C, B, device=dev),
            "ret_ext": torch.zeros(C, B, device=dev), "ret_int": torch.zeros(C, B, device=dev),
            "vext_old": torch.zeros(C, B, device=dev), "rnd_input": torch.zeros(C, B, 4, device=dev),
            "rnd_tf": torch.zeros(C, B, cfg.rnd_feature_dim, device=dev),
        }
        self._log_rext_sum = torch.zeros(C, device=dev)
        self._log_rint_mean = torch.zeros(C, device=dev)
        # cumulative visited-cell map per copy (exploration diagnostic; open cells only count)
        rows, cols = self.env.rows, self.env.cols
        self._visited = torch.zeros(C, rows * cols, dtype=torch.bool, device=dev)
        self._open_cells = (self.env.nb_mask >= 0)  # placeholder replaced two lines down
        import numpy as _np
        from pm_common import MAPS as _MAPS
        wall = _np.array(_MAPS[self.env.cfg.map_name]) == 1
        self._open_cells = torch.as_tensor(~wall.reshape(-1), device=dev)
        # the per-step function used by _rollout_body: compiled for the GPU fast paths
        if cfg.rollout_mode in ("compile-step", "capture"):
            self._one_step = torch.compile(self._one_step_pure, fullgraph=True, dynamic=False)
            self._policy_fn = torch.compile(self._policy_part, fullgraph=True, dynamic=False)
            self._rnd_fn = torch.compile(self._rnd_part, fullgraph=True, dynamic=False)
        else:
            self._one_step = self._one_step_pure
            self._policy_fn = self._policy_part
            self._rnd_fn = self._rnd_part
        # the loss forward, compiled for the GPU fast paths (backward compiles with it)
        if cfg.rollout_mode in ("compile-step", "capture") or cfg.capture_update:
            self._loss_fn = torch.compile(self._losses, fullgraph=True, dynamic=False)
        else:
            self._loss_fn = self._losses
        self._update_graph = None

    # ---- batched forwards (x always [C, M, in]) ----

    def _mlp2(self, p, x, act=torch.tanh):
        """Two hidden layers: act(x W0 + b0) W1 ... -> hidden features [C, M, 64]."""
        h = act(torch.baddbmm(p["b0"].unsqueeze(1), x, p["W0"]))
        return act(torch.baddbmm(p["b1"].unsqueeze(1), h, p["W1"]))

    def actor_mean(self, x):
        """Action mean [C, M, 2]."""
        h = self._mlp2(self.actor, x)
        return torch.baddbmm(self.actor["b2"].unsqueeze(1), h, self.actor["W2"])

    def critic_values(self, x):
        """(Vext, Vint) each [C, M]."""
        h = self._mlp2(self.critic, x)
        vext = torch.baddbmm(self.critic["bext"].unsqueeze(1), h, self.critic["Wext"])
        vint = torch.baddbmm(self.critic["bint"].unsqueeze(1), h, self.critic["Wint"])
        return vext.squeeze(-1), vint.squeeze(-1)

    def rnd_features(self, x):
        """(target_features detached, predictor_features), each [C, M, F] — layer-1 packed:
        target and predictor read the same input, so their first layers run as ONE GEMM."""
        W0 = torch.cat([self.target["W0"], self.predictor["W0"]], -1)
        b0 = torch.cat([self.target["b0"], self.predictor["b0"]], -1)
        h = torch.relu(torch.baddbmm(b0.unsqueeze(1), x, W0))
        Hh = self.target["W0"].shape[-1]
        th, ph = h[..., :Hh], h[..., Hh:]
        with torch.no_grad():
            tf = torch.baddbmm(self.target["b1"].unsqueeze(1), th, self.target["W1"])
        ph = torch.relu(torch.baddbmm(self.predictor["b1"].unsqueeze(1), ph, self.predictor["W1"]))
        pf = torch.baddbmm(self.predictor["b2"].unsqueeze(1), ph, self.predictor["W2"])
        return tf, pf

    def actor_critic(self, x):
        """Packed actor+critic forward on the same input [C, M, 4]: one layer-1 GEMM for
        both trunks, packed critic heads. Returns (mean [C,M,2], vext [C,M], vint [C,M])."""
        W0 = torch.cat([self.actor["W0"], self.critic["W0"]], -1)
        b0 = torch.cat([self.actor["b0"], self.critic["b0"]], -1)
        h = torch.tanh(torch.baddbmm(b0.unsqueeze(1), x, W0))
        ha, hc = h[..., :64], h[..., 64:]
        ha = torch.tanh(torch.baddbmm(self.actor["b1"].unsqueeze(1), ha, self.actor["W1"]))
        hc = torch.tanh(torch.baddbmm(self.critic["b1"].unsqueeze(1), hc, self.critic["W1"]))
        mean = torch.baddbmm(self.actor["b2"].unsqueeze(1), ha, self.actor["W2"])
        Wh = torch.cat([self.critic["Wext"], self.critic["Wint"]], -1)
        bh = torch.cat([self.critic["bext"], self.critic["bint"]], -1)
        v = torch.baddbmm(bh.unsqueeze(1), hc, Wh)
        return mean, v[..., 0], v[..., 1]

    def predictor_features(self, x):
        """Predictor features only [C, M, F] (the frozen target is hoisted per iteration)."""
        ph = torch.relu(torch.baddbmm(self.predictor["b0"].unsqueeze(1), x, self.predictor["W0"]))
        ph = torch.relu(torch.baddbmm(self.predictor["b1"].unsqueeze(1), ph, self.predictor["W1"]))
        return torch.baddbmm(self.predictor["b2"].unsqueeze(1), ph, self.predictor["W2"])

    def target_features(self, x):
        """Frozen target features [C, M, F], never with gradient."""
        with torch.no_grad():
            th = torch.relu(torch.baddbmm(self.target["b0"].unsqueeze(1), x, self.target["W0"]))
            return torch.baddbmm(self.target["b1"].unsqueeze(1), th, self.target["W1"])

    def whiten(self, obs):
        """RND input whitening with the CURRENT per-copy statistics, clip +-5 (spec 4.2).

        before: obs [C, M, 4] raw; after: ((obs - mean)/sqrt(var+1e-8)) clipped, float32
        """
        mean = self.obs_rms.mean.to(torch.float32).unsqueeze(1)
        std = (self.obs_rms.var + 1e-8).sqrt().to(torch.float32).unsqueeze(1)
        return ((obs - mean) / std).clamp(-5.0, 5.0)

    def _logprob(self, mean, logstd, action):
        """Diagonal-Gaussian log-density summed over action dims -> [C, M] (spec 9.1)."""
        z = (action - mean) / logstd.exp().unsqueeze(1)
        return (-0.5 * z * z - logstd.unsqueeze(1) - 0.5 * LOG2PI).sum(-1)

    # ---- rollout ----

    def _sync_state_from_env(self):
        """Copy the env's current state into the static entry buffers (in-place)."""
        self._S_pos.copy_(self.env.pos)
        self._S_vel.copy_(self.env.vel)
        self._S_goal.copy_(self.env.goal)
        self._S_sc.copy_(self.env.step_count)
        self._S_rc.copy_(self.env.reset_count)
        self._S_obs.copy_(torch.cat([self.env.pos, self.env.vel], -1))

    def _policy_part(self, obs, z):
        """Policy + value + sample + logprob only (compiled for the cuda-env backend)."""
        mean, vext, vint = self.actor_critic(obs)
        std = self.actor["logstd"].exp().unsqueeze(1)
        action = mean + std * z
        logp = (-0.5 * z * z - self.actor["logstd"].unsqueeze(1) - 0.5 * LOG2PI).sum(-1)
        return action, logp, vext, vint

    def _rnd_part(self, final_obs):
        """Whiten + RND bonus only (compiled for the cuda-env backend)."""
        tf, pf = self.rnd_features(self.whiten(final_obs))
        return 0.5 * (pf - tf).square().sum(-1)

    def _one_step_pure(self, pos, vel, goal, sc, rc, obs, z):
        """One PURE rollout step (no attribute writes): policy + value + sample + env
        step_core + RND bonus. Compiled into one fused graph for the capture/compile modes.

        Returns (pos', vel', goal', sc', rc', obs', act, logp, vext, vint, r_ext, term_f,
        done_f, final_obs, r_int)."""
        mean, vext, vint = self.actor_critic(obs)
        std = self.actor["logstd"].exp().unsqueeze(1)
        action = mean + std * z
        logp = (-0.5 * z * z - self.actor["logstd"].unsqueeze(1) - 0.5 * LOG2PI).sum(-1)
        (pos, vel, goal, sc, rc, nobs, r_ext, terminated, truncated,
         final_obs) = self.env.step_core(pos, vel, goal, sc, rc, action)
        tf, pf = self.rnd_features(self.whiten(final_obs))
        r_int = 0.5 * (pf - tf).square().sum(-1)
        return (pos, vel, goal, sc, rc, nobs, action, logp, vext, vint, r_ext,
                terminated.to(self.env.dtype), (terminated | truncated).to(self.env.dtype),
                final_obs, r_int)

    @torch.no_grad()
    def _rollout_body(self):
        """The T-step interaction loop over the static buffers (capturable as one CUDA graph).

        Reads the entry state (_S_*) and the pre-drawn noise _Z, fills every _bufs[k][t],
        and writes the exit state back into _S_* for the next pass.
        """
        pos, vel, goal, sc, rc = self._S_pos, self._S_vel, self._S_goal, self._S_sc, self._S_rc
        obs = self._S_obs
        b = self._bufs
        step = self._one_step
        for t in range(self.cfg.num_steps):
            b["obs"][t].copy_(obs)
            (pos, vel, goal, sc, rc, obs, act, logp, vext, vint, r_ext, term_f, done_f,
             final_obs, r_int) = step(pos, vel, goal, sc, rc, obs, self._Z[t])
            b["act"][t].copy_(act)
            b["logp"][t].copy_(logp)
            b["vext"][t].copy_(vext)
            b["vint"][t].copy_(vint)
            b["nobs"][t].copy_(final_obs)
            b["rext"][t].copy_(r_ext)
            b["term"][t].copy_(term_f)
            b["done"][t].copy_(done_f)
            b["rint"][t].copy_(r_int)
        self._S_pos.copy_(pos)
        self._S_vel.copy_(vel)
        self._S_goal.copy_(goal)
        self._S_sc.copy_(sc)
        self._S_rc.copy_(rc)
        self._S_obs.copy_(obs)

    @torch.no_grad()
    def _rollout_body_cuda(self):
        """The T-step loop for the cuda env backend: compiled policy/RND subgraphs around
        the fused env kernel; env state is internal to the env and updated in place."""
        env = self.env
        b = self._bufs
        obs = env.state
        for t in range(self.cfg.num_steps):
            b["obs"][t].copy_(obs)
            action, logp, vext, vint = self._policy_fn(obs, self._Z[t])
            b["act"][t].copy_(action)
            b["logp"][t].copy_(logp)
            b["vext"][t].copy_(vext)
            b["vint"][t].copy_(vint)
            obs, r_ext, terminated, truncated, final_obs = env.step(action)
            b["nobs"][t].copy_(final_obs)
            b["rext"][t].copy_(r_ext)
            b["term"][t].copy_(terminated.to(env.dtype))
            b["done"][t].copy_((terminated | truncated).to(env.dtype))
            b["rint"][t].copy_(self._rnd_fn(final_obs))

    def _build_rollout_graph(self):
        """Warm up and capture _rollout_body as one CUDA graph (128 steps -> one replay).

        The two warmup passes advance the entry-state buffers; they are snapshotted and
        restored so the first replay continues exactly where the eager path would have.
        """
        body = self._rollout_body_cuda if self.cfg.env_backend == "cuda" \
            else self._rollout_body
        if self.cfg.env_backend == "cuda":
            state_tensors = [self.env.state, self.env.goal, self.env.step_count,
                             self.env.reset_count]
        else:
            state_tensors = [self._S_pos, self._S_vel, self._S_goal, self._S_sc,
                             self._S_rc, self._S_obs]
        snap = [t.clone() for t in state_tensors]
        self._Z.normal_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)
        for dst, src in zip(state_tensors, snap):
            dst.copy_(src)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        self._graph = g

    @torch.no_grad()
    def rollout(self):
        """Collect T steps for every copy/env. Returns the flattened per-copy batch (spec 6)."""
        cfg = self.cfg
        C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
        dev = self.device
        self._Z.normal_()
        if self._graph is not None:
            self._graph.replay()
        elif self.cfg.env_backend == "cuda":
            self._rollout_body_cuda()
        else:
            self._rollout_body()
        self.global_step += T * C * N
        self._post_body()
        out = dict(self._U)
        out["reward_ext_sum"] = self._log_rext_sum
        out["rint_mean"] = self._log_rint_mean
        return out

    @torch.no_grad()
    def _post_body(self):
        """Post-rollout processing over the static buffers (capturable): bootstrap values,
        intrinsic filter + normalization, two-stream GAE, running-statistics updates, and
        the flattened batch written IN PLACE into _U (spec sections 4-7)."""
        cfg = self.cfg
        C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
        dev = self.device
        b = self._bufs
        obs_buf, nobs_buf, act_buf = b["obs"], b["nobs"], b["act"]
        logp_buf, rext_buf, rint_buf = b["logp"], b["rext"], b["rint"]
        term_buf, done_buf, vext_buf, vint_buf = b["term"], b["done"], b["vext"], b["vint"]

        # bootstrap values for every stored next obs in one batched call (spec 6)
        vext_next, vint_next = self.critic_values(nobs_buf.permute(1, 0, 2, 3).reshape(C, T * N, 4))
        vext_next = vext_next.view(C, T, N).permute(1, 0, 2)
        vint_next = vint_next.view(C, T, N).permute(1, 0, 2)

        # intrinsic filter forward in time + per-copy normalization (spec 5.2)
        filt = torch.empty(T, C, N, device=dev)
        f = self.int_filter
        for t in range(T):
            f = cfg.gamma_int * f + rint_buf[t]
            filt[t] = f
        self.int_filter.copy_(f)
        self.int_rms.update(filt.permute(1, 0, 2).reshape(C, T * N, 1))
        int_std = (self.int_rms.var + 1e-8).sqrt().to(torch.float32).view(C, 1, 1)
        rint_hat = rint_buf / int_std.permute(2, 0, 1)

        # two-stream GAE backwards (spec 7)
        aext = torch.zeros(C, N, device=dev)
        aint = torch.zeros(C, N, device=dev)
        aext_buf = torch.empty(T, C, N, device=dev)
        aint_buf = torch.empty(T, C, N, device=dev)
        boot_mask = term_buf if cfg.bootstrap_on_truncation else done_buf
        for t in range(T - 1, -1, -1):
            d_ext = rext_buf[t] + cfg.gamma_ext * vext_next[t] * (1 - boot_mask[t]) - vext_buf[t]
            aext = d_ext + cfg.gamma_ext * cfg.gae_lambda * (1 - done_buf[t]) * aext
            d_int = rint_hat[t] + cfg.gamma_int * vint_next[t] - vint_buf[t]
            aint = d_int + cfg.gamma_int * cfg.gae_lambda * aint
            aext_buf[t] = aext
            aint_buf[t] = aint

        # flatten (C, T*N, ...) IN PLACE into the static batch buffers; RND obs statistics
        # update precedes the whiten so the update phase sees the NEW statistics (spec 4.2)
        flat = lambda x: x.permute(1, 0, 2, *range(3, x.dim())).reshape(
            C, T * N, *x.shape[3:])
        nobs_flat = flat(nobs_buf)
        self.obs_rms.update(nobs_flat)
        U = self._U
        U["obs"].copy_(flat(obs_buf))
        U["actions"].copy_(flat(act_buf))
        U["old_logprob"].copy_(flat(logp_buf))
        U["adv"].copy_(flat(cfg.int_coef * aint_buf + cfg.ext_coef * aext_buf))
        U["ret_ext"].copy_(flat(aext_buf + vext_buf))
        U["ret_int"].copy_(flat(aint_buf + vint_buf))
        U["vext_old"].copy_(flat(vext_buf))
        U["rnd_input"].copy_(self.whiten(nobs_flat))
        # hoist: the target net is frozen, so its features are constant within the
        # iteration — computed once here instead of once per minibatch step
        U["rnd_tf"].copy_(self.target_features(U["rnd_input"]))
        self._log_rext_sum.copy_(rext_buf.sum(dim=(0, 2)))
        self._log_rint_mean.copy_(rint_buf.mean(dim=(0, 2)))
        # mark visited cells: world (x, y) -> flat cell index, per copy
        rows, cols = self.env.rows, self.env.cols
        jj = (nobs_flat[..., 0] + cols / 2.0).long().clamp(0, cols - 1)
        ii = (rows / 2.0 - nobs_flat[..., 1]).long().clamp(0, rows - 1)
        self._visited.scatter_(1, ii * cols + jj, torch.ones_like(jj, dtype=torch.bool))

    # ---- update phase ----

    def _clip_per_copy_and_step(self):
        """Per-copy gradient-norm clip (spec 10), then one Adam step over all copies."""
        C = self.cfg.n_copies
        g2 = torch.zeros(C, device=self.device)
        for p in self.trainable:
            g2 = g2 + p.grad.reshape(C, -1).square().sum(1)
        scale = (self.cfg.max_grad_norm / (g2.sqrt() + 1e-6)).clamp(max=1.0)
        for p in self.trainable:
            p.grad.mul_(scale.view(C, *([1] * (p.dim() - 1))))
        self.opt.step()
        # under graph capture the grad buffers must keep their storage across replays
        self.opt.zero_grad(set_to_none=not self.cfg.capture_update)

    def _losses(self, mb, style_a):
        """Per-copy losses on one (mini)batch dict; returns the scalar sum over copies.

        style_a=True drops the CLIP machinery only (gradient-exact at ratio == 1, where the
        clip branches coincide in value AND derivative). The ratio itself is KEPT: it carries
        the policy gradient -A * grad(log pi); replacing the surrogate by mean(-A) would make
        the policy loss a constant. (This corrects spec section 11 — noted there.)
        """
        cfg = self.cfg
        a = mb["adv"]
        a_n = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, unbiased=True, keepdim=True) + 1e-8)

        mean, vext, vint = self.actor_critic(mb["obs"])
        logstd = self.actor["logstd"]
        newlogp = self._logprob(mean, logstd, mb["actions"])
        ratio = (newlogp - mb["old_logprob"]).exp()

        if style_a:
            pg = (-a_n * ratio).mean(dim=1)
            v_ext = 0.5 * (vext - mb["ret_ext"]).square().mean(dim=1)
        else:
            pg = torch.maximum(-a_n * ratio,
                               -a_n * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean(dim=1)
            vc = mb["vext_old"] + (vext - mb["vext_old"]).clamp(-cfg.clip_coef, cfg.clip_coef)
            v_ext = 0.5 * torch.maximum((vext - mb["ret_ext"]).square(),
                                        (vc - mb["ret_ext"]).square()).mean(dim=1)
        v_int = 0.5 * (vint - mb["ret_int"]).square().mean(dim=1)

        pf = self.predictor_features(mb["rnd_input"])
        fwd = (pf - mb["rnd_tf"]).square().mean(dim=2).mean(dim=1)

        ent = (0.5 + 0.5 * LOG2PI + logstd).sum(dim=1)
        loss_c = pg - cfg.ent_coef * ent + cfg.vf_coef * (v_ext + v_int) + fwd
        return loss_c.sum()

    def update_full_batch(self, batch):
        """Style A: one gradient step on all T*N rows per copy, then discard (spec 11)."""
        loss = self._loss_fn(batch, style_a=True)
        loss.backward()
        self._clip_per_copy_and_step()
        return loss.detach()

    def update_epoch_minibatch(self, batch):
        """Style B: update_epochs x num_minibatches shuffled steps of B/num_minibatches rows
        per copy (spec 12). Permutations drawn up front, independent per copy AND per epoch."""
        cfg = self.cfg
        C = cfg.n_copies
        Brows = cfg.num_steps * cfg.n_envs
        mb_size = Brows // cfg.num_minibatches
        perm = torch.rand(cfg.update_epochs, C, Brows, device=self.device).argsort(dim=-1)
        last = 0.0
        for e in range(cfg.update_epochs):
            for k in range(cfg.num_minibatches):
                idx = perm[e][:, k * mb_size:(k + 1) * mb_size]
                mb = {}
                for key in self._U_KEYS:
                    t = batch[key]
                    ix = idx.unsqueeze(-1).expand(C, mb_size, t.shape[-1]) if t.dim() == 3 else idx
                    mb[key] = t.gather(1, ix)
                loss = self._loss_fn(mb, style_a=False)
                loss.backward()
                self._clip_per_copy_and_step()
                last = loss.detach()
        return last

    _U_KEYS = ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int",
               "vext_old", "rnd_input", "rnd_tf"]

    def _update_body_captured(self):
        """The update loop over the static _U buffers (capturable as one CUDA graph).

        Style resolved at build time; gathers use the static permutation buffer _perm,
        refilled outside the graph each iteration.
        """
        cfg = self.cfg
        C = cfg.n_copies
        if cfg.update_style == "full_batch":
            loss = self._loss_fn(self._U, style_a=True)
            loss.backward()
            self._clip_per_copy_and_step()
        else:
            Brows = cfg.num_steps * cfg.n_envs
            mb_size = Brows // cfg.num_minibatches
            for e in range(cfg.update_epochs):
                for k in range(cfg.num_minibatches):
                    idx = self._perm[e][:, k * mb_size:(k + 1) * mb_size]
                    mb = {}
                    for key in self._U_KEYS:
                        t = self._U[key]
                        ix = idx.unsqueeze(-1).expand(C, mb_size, t.shape[-1])                             if t.dim() == 3 else idx
                        mb[key] = t.gather(1, ix)
                    loss = self._loss_fn(mb, style_a=False)
                    loss.backward()
                    self._clip_per_copy_and_step()
        self._loss_out.copy_(loss.detach())

    def _build_update_graph(self, example_batch=None):
        """Warm up (with parameter/optimizer state restored afterwards) and capture the
        whole update phase as one CUDA graph."""
        cfg = self.cfg
        C = cfg.n_copies
        Brows = cfg.num_steps * cfg.n_envs
        # identity permutations for warmup/capture: building must not consume the global
        # RNG (real permutations are drawn in update_captured, in step with the eager path)
        self._perm = torch.arange(Brows, device=self.device).expand(
            cfg.update_epochs, C, Brows).contiguous()
        self._loss_out = torch.zeros((), device=self.device)

        # warmup passes allocate grads / Adam state / compiled artifacts, but also step the
        # parameters — snapshot and restore so training starts from the virgin state
        snap = [t.detach().clone() for t in self.trainable]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._update_body_captured()
        torch.cuda.current_stream().wait_stream(side)
        with torch.no_grad():
            for t, sv in zip(self.trainable, snap):
                t.copy_(sv)
        for group_state in self.opt.state.values():
            for k, v in group_state.items():
                if torch.is_tensor(v):
                    v.zero_()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._update_body_captured()
        self._update_graph = g

    def update_captured(self, batch):
        """The batch already lives in the static _U buffers (filled by _post_body); redraw
        permutations and replay. Builds the graph on first use (warmup restores parameters,
        so iteration 1 is real)."""
        if self._update_graph is None:
            self._build_update_graph(batch)
        if self.cfg.update_style != "full_batch":
            cfg = self.cfg
            self._perm.copy_(torch.rand(cfg.update_epochs, cfg.n_copies,
                                        cfg.num_steps * cfg.n_envs,
                                        device=self.device).argsort(dim=-1))
        self._update_graph.replay()
        return self._loss_out

    def _iteration_body(self):
        """One full training iteration over the static buffers: rollout, post, update."""
        if self.cfg.env_backend == "cuda":
            self._rollout_body_cuda()
        else:
            self._rollout_body()
        self._post_body()
        self._update_body_captured()

    def _build_iteration_graph(self):
        """Warm up and capture the WHOLE iteration as one CUDA graph (Module 3, torch).

        Warmup advances env state, running statistics, parameters, and optimizer state;
        everything is snapshotted and restored so training starts from the virgin state.
        """
        cfg = self.cfg
        C = cfg.n_copies
        Brows = cfg.num_steps * cfg.n_envs
        self._perm = torch.arange(Brows, device=self.device).expand(
            cfg.update_epochs, C, Brows).contiguous()
        self._loss_out = torch.zeros((), device=self.device)
        self._Z.normal_()

        if self.cfg.env_backend == "cuda":
            env_state = [self.env.state, self.env.goal, self.env.step_count,
                         self.env.reset_count]
        else:
            env_state = [self._S_pos, self._S_vel, self._S_goal, self._S_sc, self._S_rc,
                         self._S_obs]
        stat_tensors = env_state + [self.int_filter,
                        self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count,
                        self.int_rms.mean, self.int_rms.var, self.int_rms.count]
        snap_state = [t.clone() for t in stat_tensors]
        snap_params = [t.detach().clone() for t in self.trainable]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._iteration_body()
        torch.cuda.current_stream().wait_stream(side)
        with torch.no_grad():
            for t, sv in zip(stat_tensors, snap_state):
                t.copy_(sv)
            for t, sv in zip(self.trainable, snap_params):
                t.copy_(sv)
        for group_state in self.opt.state.values():
            for k, v in group_state.items():
                if torch.is_tensor(v):
                    v.zero_()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._iteration_body()
        self._iteration_graph = g

    def iteration_captured(self):
        """One training iteration: refill noise and permutations, replay the graph."""
        cfg = self.cfg
        self._Z.normal_()
        if cfg.update_style != "full_batch":
            self._perm.copy_(torch.rand(cfg.update_epochs, cfg.n_copies,
                                        cfg.num_steps * cfg.n_envs,
                                        device=self.device).argsort(dim=-1))
        self._iteration_graph.replay()
        self.global_step += cfg.num_steps * cfg.n_copies * cfg.n_envs
        return self._loss_out

    # ---- driver ----

    def prime_obs_rms(self):
        """Spec 4.3: random-action rollouts updating only the RND observation statistics."""
        cfg = self.cfg
        C, N = cfg.n_copies, cfg.n_envs
        with torch.no_grad():
            for _ in range(cfg.obs_norm_init_iters):
                batch = []
                for _ in range(cfg.num_steps):
                    a = torch.rand(C, N, 2, device=self.device) * 2 - 1
                    _, _, _, _, final_obs = self.env.step(a)
                    batch.append(final_obs)
                self.obs_rms.update(torch.stack(batch).permute(1, 0, 2, 3).reshape(C, -1, 4))
        self.env.reset()
        self._sync_state_from_env()

    def train(self, num_iterations, log_every_seconds=1200, log_fn=print,
              history_every=0):
        """Run the full loop. Sparse logging (default every 20 minutes); returns final stats.

        history_every > 0 samples per-copy metrics every that many iterations (one small
        host sync each sample) and returns them under stats["history"]."""
        cfg = self.cfg
        if cfg.capture_update:
            update = self.update_captured
        elif cfg.update_style == "full_batch":
            update = self.update_full_batch
        else:
            update = self.update_epoch_minibatch
        history = []
        self.prime_obs_rms()
        if cfg.one_graph:
            self._build_iteration_graph()
        elif self.cfg.rollout_mode == "capture":
            self._build_rollout_graph()
        t0 = time.time()
        last_log = t0
        for it in range(1, num_iterations + 1):
            if cfg.anneal_lr:
                lr = cfg.learning_rate * (1.0 - (it - 1.0) / num_iterations)
                if cfg.capture_update:
                    self._lr_t.fill_(lr)
                else:
                    for gme in self.opt.param_groups:
                        gme["lr"] = lr
            if cfg.one_graph:
                loss = self.iteration_captured()
            else:
                batch = self.rollout()
                loss = update(batch)
            now = time.time()
            if history_every and (it % history_every == 0 or it == num_iterations):
                cov = self._visited[:, self._open_cells].float().mean(dim=1)
                history.append({
                    "iteration": it, "global_step": self.global_step,
                    "seconds": now - t0,
                    "reward_ext_sum_per_copy": self._log_rext_sum.tolist(),
                    "rint_mean_per_copy": self._log_rint_mean.tolist(),
                    "coverage_per_copy": cov.tolist(),
                })
            if now - last_log >= log_every_seconds or it == num_iterations:
                last_log = now
                r = self._log_rext_sum
                cov = self._visited[:, self._open_cells].float().mean(dim=1)
                log_fn(f"iter {it}/{num_iterations} step {self.global_step} "
                       f"loss {float(loss):.4f} ext-reward/copy mean {r.mean():.3f} "
                       f"min {r.min():.3f} max {r.max():.3f} "
                       f"rint mean {self._log_rint_mean.mean():.4f} "
                       f"coverage mean {cov.mean():.3f} max {cov.max():.3f} "
                       f"elapsed {now - t0:.0f}s")
        return {"iterations": num_iterations, "global_step": self.global_step,
                "seconds": time.time() - t0, "history": history}


def main():
    """Smoke entry: tiny run on CPU or GPU."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-copies", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--style", default="epoch_minibatch")
    args = ap.parse_args()
    trainer = PPORND(PPOConfig(n_copies=args.n_copies, update_style=args.style),
                     device=args.device)
    stats = trainer.train(args.iters, log_every_seconds=0)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

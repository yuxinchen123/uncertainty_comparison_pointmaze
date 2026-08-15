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
        self.cfg = cfg
        self.device = torch.device(device)
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
        # the per-step function used by _rollout_body: compiled for the GPU fast paths
        if cfg.rollout_mode in ("compile-step", "capture"):
            self._one_step = torch.compile(self._one_step_pure, fullgraph=True, dynamic=False)
        else:
            self._one_step = self._one_step_pure
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
        """(target_features detached, predictor_features), each [C, M, F]."""
        with torch.no_grad():
            th = torch.relu(torch.baddbmm(self.target["b0"].unsqueeze(1), x, self.target["W0"]))
            tf = torch.baddbmm(self.target["b1"].unsqueeze(1), th, self.target["W1"])
        ph = torch.relu(torch.baddbmm(self.predictor["b0"].unsqueeze(1), x, self.predictor["W0"]))
        ph = torch.relu(torch.baddbmm(self.predictor["b1"].unsqueeze(1), ph, self.predictor["W1"]))
        pf = torch.baddbmm(self.predictor["b2"].unsqueeze(1), ph, self.predictor["W2"])
        return tf, pf

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

    def _one_step_pure(self, pos, vel, goal, sc, rc, obs, z):
        """One PURE rollout step (no attribute writes): policy + value + sample + env
        step_core + RND bonus. Compiled into one fused graph for the capture/compile modes.

        Returns (pos', vel', goal', sc', rc', obs', act, logp, vext, vint, r_ext, term_f,
        done_f, final_obs, r_int)."""
        vext, vint = self.critic_values(obs)
        mean = self.actor_mean(obs)
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

    def _build_rollout_graph(self):
        """Warm up and capture _rollout_body as one CUDA graph (128 steps -> one replay).

        The two warmup passes advance the entry-state buffers; they are snapshotted and
        restored so the first replay continues exactly where the eager path would have.
        """
        snap = [t.clone() for t in (self._S_pos, self._S_vel, self._S_goal,
                                    self._S_sc, self._S_rc, self._S_obs)]
        self._Z.normal_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._rollout_body()
        torch.cuda.current_stream().wait_stream(side)
        for dst, src in zip((self._S_pos, self._S_vel, self._S_goal,
                             self._S_sc, self._S_rc, self._S_obs), snap):
            dst.copy_(src)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._rollout_body()
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
        else:
            self._rollout_body()
        self.global_step += T * C * N
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
        self.int_filter = f
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

        adv = cfg.int_coef * aint_buf + cfg.ext_coef * aext_buf
        ret_ext = aext_buf + vext_buf
        ret_int = aint_buf + vint_buf

        # update RND observation statistics with this rollout's next observations, then
        # rebuild the update-phase RND input with the NEW statistics (spec 4.2 ordering)
        flat = lambda x: x.permute(1, 0, 2, *range(3, x.dim())).reshape(
            C, T * N, *x.shape[3:])
        nobs_flat = flat(nobs_buf)
        self.obs_rms.update(nobs_flat)
        return {
            "obs": flat(obs_buf), "actions": flat(act_buf), "old_logprob": flat(logp_buf),
            "adv": flat(adv), "ret_ext": flat(ret_ext), "ret_int": flat(ret_int),
            "vext_old": flat(vext_buf), "rnd_input": self.whiten(nobs_flat),
            "reward_ext_sum": rext_buf.sum(dim=(0, 2)), "rint_mean": rint_buf.mean(dim=(0, 2)),
        }

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

        mean = self.actor_mean(mb["obs"])
        logstd = self.actor["logstd"]
        newlogp = self._logprob(mean, logstd, mb["actions"])
        vext, vint = self.critic_values(mb["obs"])
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

        tf, pf = self.rnd_features(mb["rnd_input"])
        fwd = (pf - tf).square().mean(dim=2).mean(dim=1)

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
                for key in ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int",
                            "vext_old", "rnd_input"]:
                    t = batch[key]
                    ix = idx.unsqueeze(-1).expand(C, mb_size, t.shape[-1]) if t.dim() == 3 else idx
                    mb[key] = t.gather(1, ix)
                loss = self._loss_fn(mb, style_a=False)
                loss.backward()
                self._clip_per_copy_and_step()
                last = loss.detach()
        return last

    _U_KEYS = ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int",
               "vext_old", "rnd_input"]

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

    def _build_update_graph(self, example_batch):
        """Warm up (with parameter/optimizer state restored afterwards) and capture the
        whole update phase as one CUDA graph."""
        cfg = self.cfg
        C = cfg.n_copies
        Brows = cfg.num_steps * cfg.n_envs
        self._U = {k: example_batch[k].clone() for k in self._U_KEYS}
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
        """Copy the fresh batch into the static buffers, redraw permutations, replay.
        Builds the graph on first use (warmup restores parameters, so iteration 1 is real)."""
        if self._update_graph is None:
            self._build_update_graph(batch)
        for k in self._U_KEYS:
            self._U[k].copy_(batch[k])
        if self.cfg.update_style != "full_batch":
            cfg = self.cfg
            self._perm.copy_(torch.rand(cfg.update_epochs, cfg.n_copies,
                                        cfg.num_steps * cfg.n_envs,
                                        device=self.device).argsort(dim=-1))
        self._update_graph.replay()
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

    def train(self, num_iterations, log_every_seconds=1200, log_fn=print):
        """Run the full loop. Sparse logging (default every 20 minutes); returns final stats."""
        cfg = self.cfg
        if cfg.capture_update:
            update = self.update_captured
        elif cfg.update_style == "full_batch":
            update = self.update_full_batch
        else:
            update = self.update_epoch_minibatch
        self.prime_obs_rms()
        if self.cfg.rollout_mode == "capture":
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
            batch = self.rollout()
            loss = update(batch)
            now = time.time()
            if now - last_log >= log_every_seconds or it == num_iterations:
                last_log = now
                r = batch["reward_ext_sum"]
                log_fn(f"iter {it}/{num_iterations} step {self.global_step} "
                       f"loss {float(loss):.4f} ext-reward/copy mean {r.mean():.3f} "
                       f"min {r.min():.3f} max {r.max():.3f} "
                       f"rint mean {batch['rint_mean'].mean():.4f} "
                       f"elapsed {now - t0:.0f}s")
        return {"iterations": num_iterations, "global_step": self.global_step,
                "seconds": time.time() - t0}


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

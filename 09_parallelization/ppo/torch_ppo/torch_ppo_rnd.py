"""Batched multi-copy PPO+RND in PyTorch — implements ../research/ppo_rnd_algorithm_spec.md.

C independent training copies (own networks, own envs, own running statistics, own optimizer
moments) advance in lockstep as one batched computation. Every parameter carries a leading copy
axis and every forward is a batched matrix multiplication. All reductions keep the copy axis; the
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
    fused_adam: bool = False         # accepted for compatibility and no longer used: the
                                     # optimizer is two streaming passes over the flat
                                     # parameter buffer, which is faster here than torch's
                                     # multi-tensor kernels
    compile_opt: bool = False        # compile the gradient limit and the Adam step over the
                                     # flat buffer (as two separate programs)
    capture_update: bool = False     # CUDA-graph the whole update phase (implies capturable Adam)
    one_graph: bool = False          # capture rollout+post+update as ONE graph per iteration
                                     # (requires rollout_mode="capture" semantics and
                                     # capture_update=True; overrides both)
    tf32: bool = False               # TF32 tensor-core matmuls (faster, ~1e-3 relative rounding)
    learning_rates: tuple = ()       # sweep: one learning rate per GROUP of copies; empty
                                     # means every copy uses `learning_rate`
    copies_per_rate: tuple = ()      # sweep: how many copies each rate gets (sums to n_copies)
    sweep_seed_mode: str = "paired"  # "paired": copy k of every group shares one seed stream,
                                     # so the groups differ ONLY by the swept value;
                                     # "distinct": every copy is its own seed
    compile_post: bool = False       # compile the post-rollout body (its filter and GAE are
                                     # python loops over T, so eager they are ~256 tiny kernels)
    gradient_buffer: bool = False    # True holds every gradient in ONE [C, P] buffer, so the gradient
                                     # limit and the Adam step are one program each. The
                                     # automatic-differentiation system writes its gradients into
                                     # fresh tensors, so this costs a copy of the whole buffer in
                                     # every update step — 981 megabytes read and 981 written at
                                     # 4,096 copies. False reads the gradients where they were
                                     # written: twenty-one programs instead of one for each of the
                                     # two optimizer passes, no copy, and no buffer to hold.
                                     # Which is better depends on the copy count, so
                                     # production_config chooses it from there.
    fuse_copy_and_limit: bool = False  # with the buffer on: sum the squared gradients in the
                                     # SAME program that copies them into it, instead of reading
                                     # the whole buffer again afterwards. One pass over 981
                                     # megabytes less per update step at 4,096 copies. The sum
                                     # then runs tensor by tensor, so it lands on exactly the
                                     # value the no-buffer form computes and a few last bits from
                                     # the one-row reduction's.
    parameter_layout: str = "parameter_major"
    # "copy_major": one buffer row per copy, [C, P]. A parameter's window is a column slice, so
    # it is strided across copies, but the buffer is one contiguous array and a single program
    # can walk it applying a per-copy rate. "parameter_major": one contiguous block per
    # parameter instead, so every window is contiguous and every pass over it reaches the card's
    # full bandwidth — at the price of twenty-one programs per optimizer pass rather than one,
    # because a copy no longer owns a row. Requires gradient_buffer=False.
    generated_rollout_matmul: bool = False
    # let the compiler generate the rollout step's matrix multiplications from its own templates
    # instead of calling the library's. The rollout's are the least efficient programs in the
    # trainer: four rows per copy against a whole copy's weights, re-read on every one of the 128
    # steps, and the library's kernels reach about 850 gigabytes per second on them where the
    # update stage's reach 2,500 to 3,400.
    env_backend: str = "torch"       # "torch" (env fused into the compiled step) or "cuda"
                                     # (the single fused kernel from pointmaze/cuda_env)


def sweep_config(learning_rates, copies_per_rate, style="epoch_minibatch", **overrides):
    """Build the configuration for a learning-rate sweep across copy groups.

    learning_rates: the rates to try, e.g. [1e-4, 3e-4, 1e-3, 3e-3].
    copies_per_rate: how many independent copies each rate gets — one number for all rates,
    or one per rate. The total copy count is their sum.

    before: sweep_config([1e-4, 1e-3], 128) ; after: 256 copies, the first 128 training at
    1e-4 and the second 128 at 1e-3, with copy k of both groups sharing initial weights and
    environments (paired), so the two groups differ only by the rate.
    """
    rates = tuple(float(x) for x in learning_rates)
    counts = ((int(copies_per_rate),) * len(rates) if isinstance(copies_per_rate, int)
              else tuple(int(x) for x in copies_per_rate))
    assert len(counts) == len(rates), "give one copy count per learning rate, or a single number"
    return production_config(sum(counts), style=style, learning_rates=rates,
                             copies_per_rate=counts, **overrides)


def production_config(n_copies, style="epoch_minibatch", **overrides) -> "PPOConfig":
    """The measured-best PyTorch configuration for this copy count. One definition, so the
    benchmarks, the training driver and the tests cannot drift apart: whole-iteration CUDA-graph
    capture, compiled post-rollout body, TF32 matrix units, and a compiled gradient limit and
    Adam step.

    Where the gradients live is chosen from the copy count, because the measurement reverses.
    Reading them where the backward pass wrote them removes a copy of the whole parameter buffer
    from every update step and costs forty-two device programs where there were two: worth -2.5%
    at 512 copies and -4.4% at 4,096, and +2.8% to +6.7% AGAINST it at 128 copies and below,
    where an iteration's cost is the number of programs it issues rather than the bytes they
    move. The buffer's layout follows: one block per parameter where the optimizer is twenty-one
    programs, one row per copy where it is one.
    """
    base = dict(rollout_mode="capture", capture_update=True, one_graph=True,
                fused_adam=True, tf32=True, compile_post=True, compile_opt=True)
    # before: n_copies=128  -> gradient_buffer True,  parameter_layout "copy_major"
    # after:  n_copies=4096 -> gradient_buffer False, parameter_layout "parameter_major"
    memory_bound = n_copies >= 512
    base.update(gradient_buffer=not memory_bound,
                parameter_layout="parameter_major" if memory_bound else "copy_major")
    base.update(overrides)                      # an explicit override always wins
    return PPOConfig(n_copies=n_copies, update_style=style, **base)


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
        if cfg.one_graph and cfg.rollout_mode != "capture":
            # the iteration graph captures the rollout, which is only correct (and only fast)
            # with the compiled per-step function the capture path builds
            object.__setattr__(cfg, "rollout_mode", "capture")
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.tf32:
            torch.set_float32_matmul_precision("high")
        C = cfg.n_copies
        # per-copy learning rate and per-copy seed stream (a sweep gives each GROUP of copies
        # its own rate; paired seeding makes copy k of every group start from the same weights
        # and see the same environments, so a difference between groups is the rate's doing)
        # before: rates (1e-4, 1e-3), counts (2, 2); after: lr [1e-4,1e-4,1e-3,1e-3],
        #         seed index [0,1,0,1], group index [0,0,1,1]
        self.is_sweep = bool(cfg.learning_rates)
        if self.is_sweep:
            assert sum(cfg.copies_per_rate) == C, "copies_per_rate must sum to n_copies"
            lr_list, seed_list, group_list = [], [], []
            for g, (rate, count) in enumerate(zip(cfg.learning_rates, cfg.copies_per_rate)):
                lr_list += [rate] * count
                seed_list += list(range(count)) if cfg.sweep_seed_mode == "paired" \
                    else list(range(len(seed_list), len(seed_list) + count))
                group_list += [g] * count
            self.copy_seed_index = seed_list
            self.copy_group = torch.tensor(group_list, device=self.device)
            self.lr_per_copy = torch.tensor(lr_list, dtype=torch.float32, device=self.device)
        else:
            self.copy_seed_index = list(range(C))
            self.copy_group = torch.zeros(C, dtype=torch.long, device=self.device)
            self.lr_per_copy = torch.full((C,), cfg.learning_rate, device=self.device)

        if cfg.env_backend == "cuda":
            sys.path.insert(0, str(BASE / "pointmaze" / "cuda_env"))
            from cuda_pointmaze import CudaPointMaze
            self.env = CudaPointMaze(env_cfg or EnvConfig(), cfg.n_copies, cfg.n_envs,
                                     device=device, base_seed=cfg.base_seed)
            assert not self.is_sweep, "the cuda env backend does not take a seed index yet"
        else:
            self.env = TorchPointMaze(env_cfg or EnvConfig(), cfg.n_copies, cfg.n_envs,
                                      device=device, base_seed=cfg.base_seed,
                                      copy_seed_index=self.copy_seed_index)
        F, Hh = cfg.rnd_feature_dim, cfg.rnd_hidden


        def stack(net, layers):
            # per-copy keyed init: copy i's weights identical whether C=8 or C=128
            out = {}
            for li, (rows, cols, gain) in enumerate(layers):
                ws, bs = [], []
                for c in range(C):
                    g = torch.Generator().manual_seed(
                        _substream_seed(cfg.base_seed, self.copy_seed_index[c], net, li))
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

        # ---- every trainable tensor is a window onto ONE flat buffer ----
        # The parameters stay twenty-one separate leaf tensors, because that is what the forward
        # pass and autograd want, but their storage is twenty-one windows onto a single [C, P]
        # buffer, and their gradients are windows onto a single [C, P] gradient buffer. The
        # optimizer side then touches one tensor instead of twenty-one: the per-copy gradient
        # norm is one reduction, the Adam step is one kernel, and zeroing is one kernel.
        # Measured before this change: the clip alone was 31% of a minibatch step, because it
        # walked all twenty-one tensors twice.
        # before: 19 separately allocated tensors, each with its own gradient
        # after:  19 windows onto _flat [C, P] and _flat_grad [C, P]
        spec = ([("actor", k) for k in ["W0", "b0", "W1", "b1", "W2", "b2", "logstd"]]
                + [("critic", k) for k in ["W0", "b0", "W1", "b1", "Wext", "bext",
                                           "Wint", "bint"]]
                + [("predictor", k) for k in ["W0", "b0", "W1", "b1", "W2", "b2"]])
        groups = {"actor": self.actor, "critic": self.critic, "predictor": self.predictor}
        shapes = [(g, k, tuple(groups[g][k].shape)) for g, k in spec]
        # Each window starts at a multiple of ALIGN numbers, and so does the per-copy row, so
        # every parameter's address is a multiple of 16 bytes for EVERY copy. Without this the
        # windows pack tightly, the row length is 59,910, and a copy's parameters sit at
        # addresses that are not multiples of 16; the matrix-multiply library then selects its
        # scalar-load kernels (the "align1" variants visible in a kernel profile) instead of the
        # four-at-a-time ones, and every multiplication in the trainer pays for it. The padding
        # costs ten numbers per copy and is never read: nothing writes a gradient into it, so
        # its Adam step is exactly zero and it adds exactly zero to the gradient norm.
        # before: widths 256, 64, 4096, ..., 2, 2, ...; row length 59,910 (not a multiple of 4)
        # after:  strides 256, 64, 4096, ..., 4, 4, ...; row length 59,920 (a multiple of 4)
        ALIGN = 4
        widths, strides = [], []
        for _, _, sh in shapes:
            n = 1
            for d in sh[1:]:
                n *= d
            widths.append(n)
            strides.append(-(-n // ALIGN) * ALIGN)

        # Two layouts of the same buffer, both keeping every copy's start on a sixteen-byte
        # boundary. Which is better is a measurement, and it is the OPPOSITE of what round four
        # assumed: the layout that lets the optimizer be one program is not the layout that lets
        # it be fast when it is twenty-one.
        #   "copy_major"       one row per copy, [C, P]. A parameter's window is a column slice,
        #                      so it is strided across copies, but the buffer is one contiguous
        #                      array and a single program can walk it applying a per-copy rate.
        #   "parameter_major"  one contiguous block per parameter, C rows of n numbers each.
        #                      Every window is contiguous, so every pass over it reaches the
        #                      card's full bandwidth; the price is that a copy no longer owns a
        #                      row of the buffer, so the gradient limit and the Adam step have to
        #                      be twenty-one programs.
        # before (copy_major, C=4): [[copy 0's 59,920 numbers], [copy 1's], [copy 2's], ...]
        # after  (parameter_major): [W0 for copies 0..3][b0 for copies 0..3][W1 for copies 0..3]
        assert cfg.parameter_layout in ("copy_major", "parameter_major"), \
            f"unknown parameter layout {cfg.parameter_layout!r}"
        copy_major = cfg.parameter_layout == "copy_major"
        assert copy_major or not cfg.gradient_buffer, \
            "a parameter-major buffer has no per-copy row for one program to walk, so the " \
            "gradient buffer that exists to allow one buys nothing"
        offsets, length = [], 0
        for stride in strides:
            offsets.append(length)
            length += stride if copy_major else C * stride
        shape = (C, length) if copy_major else (length,)
        self._flat = torch.zeros(*shape, device=self.device)
        self._m = torch.zeros_like(self._flat)
        self._v = torch.zeros_like(self._flat)
        # the gradient buffer exists only for the form that copies into it; the other form has
        # nothing to hold, which is 981 megabytes less on the card at 4,096 copies
        self._flat_grad = torch.zeros(*shape, device=self.device) if cfg.gradient_buffer else None

        def window(buf, i):
            """Parameter i's view of one of the buffers, in whichever layout is in use."""
            if copy_major:
                return buf[:, offsets[i]:offsets[i] + widths[i]].view(C, *shapes[i][2][1:])
            block = buf[offsets[i]:offsets[i] + C * strides[i]].view(C, strides[i])
            return block[:, :widths[i]].view(C, *shapes[i][2][1:])

        self.trainable, self.grad_windows = [], []
        self.param_windows, self.m_windows, self.v_windows = [], [], []
        for i, (g, k, sh) in enumerate(shapes):
            window(self._flat, i).copy_(groups[g][k])
            # detach makes the window a LEAF that shares storage. Its gradient is NOT attached
            # here: see _backward for why an attached gradient costs four extra passes over a
            # buffer that is a gigabyte at four thousand copies.
            w = window(self._flat, i).detach().requires_grad_(True)
            groups[g][k] = w
            self.trainable.append(w)
            # the same storage seen WITHOUT a gradient, for the optimizer to write through:
            # writing into a leaf that requires a gradient from inside a compiled region does
            # not reliably reach the parameter (found in round 3 and fixed there the same way)
            self.param_windows.append(window(self._flat, i))
            self.m_windows.append(window(self._m, i))
            self.v_windows.append(window(self._v, i))
            if cfg.gradient_buffer:
                self.grad_windows.append(window(self._flat_grad, i))
        assert length % ALIGN == 0, "the buffer length must keep every copy aligned"
        base = self._flat.untyped_storage().data_ptr()
        end = base + self._flat.numel() * self._flat.element_size()
        for i, w in enumerate(self.trainable):
            for copy in range(min(2, C)):
                assert w[copy].data_ptr() % (ALIGN * 4) == 0, \
                    "a parameter window lost its alignment"
            assert base <= w.data_ptr() < end, "a parameter is not a window onto the buffer"
            assert self.param_windows[i].data_ptr() == w.data_ptr(), "a write window drifted"
            if cfg.gradient_buffer:
                gw = self.grad_windows[i]
                assert gw.data_ptr() % (ALIGN * 4) == 0, "a gradient window lost its alignment"
                assert (gw.untyped_storage().data_ptr()
                        == self._flat_grad.untyped_storage().data_ptr()), "a gradient escaped"

        self._lr_t = torch.tensor(cfg.learning_rate, device=self.device)
        # lr_scale is the annealing factor; the per-copy rates are multiplied by it, so one
        # device tensor drives every group and the captured graph reads the current value
        self._lr_scale = torch.ones((), device=self.device)
        self._adam_t = torch.zeros((), device=self.device)
        # the decay constants live on the device: building them per step would be a
        # host-to-device copy, which graph capture forbids
        self._b1 = torch.tensor(0.9, device=self.device)
        self._b2 = torch.tensor(0.999, device=self.device)
        self._lr_col = (self.lr_per_copy.view(C, 1) if self.is_sweep
                        else self._lr_t.reshape(1, 1).expand(C, 1))
        # the gradient-limiting factor lives in its own buffer, so the reduction that produces
        # it and the Adam step that consumes it are two separate programs (see _grad_scale)
        self._scale = torch.ones(C, 1, device=self.device)
        # the two per-copy columns reshaped once for each parameter's rank, so the per-tensor
        # optimizer's inner loop does no shape work — and so no view of an expanded tensor is
        # attempted inside a compiled region, which `view` refuses
        # before: scale [C, 1] and a parameter [C, 256, 128]
        # after:  scale [C, 1, 1], broadcasting over both of the parameter's own axes
        self._scale_views = [self._scale.view(-1, *((1,) * (p.dim() - 1)))
                             for p in self.param_windows]
        self._lr_views = [self._lr_col.reshape(-1, *((1,) * (p.dim() - 1)))
                          for p in self.param_windows]
        self.opt = None

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
        # the same batch in shuffled order: filled once per epoch so that each minibatch is a
        # slice of it rather than its own gather (see the update body)
        self._UP = {k: torch.empty_like(v) for k, v in self._U.items()}
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
        # the per-step function used by _rollout_body: compiled for the GPU fast paths.
        # The settings go to the compiler as options rather than around it, so that two trainers
        # in one process — which is how this project measures a change — compile separately.
        rollout_options = dict(max_autotune_gemm=True,
                               max_autotune_gemm_backends="TRITON") \
            if cfg.generated_rollout_matmul else None
        if cfg.rollout_mode in ("compile-step", "capture"):
            self._one_step = torch.compile(self._one_step_pure, fullgraph=True, dynamic=False,
                                           options=rollout_options)
            self._policy_fn = torch.compile(self._policy_part, fullgraph=True, dynamic=False,
                                            options=rollout_options)
        else:
            self._one_step = self._one_step_pure
            self._policy_fn = self._policy_part
        # the gradient limit and the Adam step are compiled SEPARATELY on purpose: each is a
        # streaming pass over the flat buffer and each then runs at the card's full bandwidth,
        # whereas compiling them together produces one reduce-and-update program that runs at
        # two thirds of it (measured, `benchmarks/profile_kernels.py`)
        assert cfg.gradient_buffer or not cfg.fuse_copy_and_limit, \
            "there is no copy for the gradient limit to ride along with when there is no buffer"
        # how the gradients get from the backward pass to the optimizer, chosen once here so the
        # update loop itself has no branch in it
        self._place_fn = (self._place_where_written if not cfg.gradient_buffer
                          else self._place_and_measure if cfg.fuse_copy_and_limit
                          else self._place_by_copying)
        adam_impl = self._adam_flat if cfg.gradient_buffer else self._adam_per_tensor
        self._adam_fn = torch.compile(adam_impl, dynamic=False) \
            if cfg.compile_opt else adam_impl
        # the gradient limit is its own program EXCEPT when it rides along with the copy: with
        # the buffer it reads one contiguous row, without it the twenty-one gradients
        if cfg.fuse_copy_and_limit:
            copy_impl = self._copy_and_measure
            self._copy_measure_fn = torch.compile(copy_impl, dynamic=False) \
                if cfg.compile_opt else copy_impl
            self._scale_fn = None
        else:
            scale_impl = self._grad_scale if cfg.gradient_buffer else self._grad_scale_per_tensor
            self._scale_fn = torch.compile(scale_impl, dynamic=False) \
                if cfg.compile_opt else scale_impl
        # the post-rollout body, compiled when asked (fuses the T-step scans)
        self._post_fn = torch.compile(self._post_body_impl, fullgraph=True, dynamic=False) \
            if cfg.compile_post else self._post_body_impl
        # the loss forward, compiled for the GPU fast paths (backward compiles with it)
        if cfg.rollout_mode in ("compile-step", "capture") or cfg.capture_update:
            self._loss_fn = torch.compile(self._losses, fullgraph=True, dynamic=False)
        else:
            self._loss_fn = self._losses
        self._update_graph = None

    # ---- batched forwards (x always [C, M, in]) ----
    #
    # Every layer is written as a plain batched matrix multiplication followed by "add the bias"
    # in the SAME expression as the activation function, rather than as one `baddbmm` call.
    # `baddbmm(b, x, W)` looks like the tighter form but is the more expensive one here: the
    # library has no batched multiply that broadcasts a bias, so it first writes the expanded
    # bias into the output tensor, then asks the multiplication to accumulate on top of it,
    # and the activation then reads and writes that tensor again — five passes over the output.
    # Written as `act(bmm(x, W) + b)` the compiler fuses the bias and the activation into one
    # pass, so the output is touched three times. At a thousand copies and more these tensors
    # are hundreds of megabytes and the passes over them are what the stage costs.
    # before: baddbmm writes bias, multiply reads+writes, activation reads+writes  (5 passes)
    # after:  multiply writes, one fused kernel reads and writes                    (3 passes)

    def _mlp2(self, p, x, act=torch.tanh):
        """Two hidden layers: act(x W0 + b0) W1 ... -> hidden features [C, M, 64]."""
        h = act(torch.bmm(x, p["W0"]) + p["b0"].unsqueeze(1))
        return act(torch.bmm(h, p["W1"]) + p["b1"].unsqueeze(1))

    def actor_mean(self, x):
        """Action mean [C, M, 2]."""
        h = self._mlp2(self.actor, x)
        return torch.bmm(h, self.actor["W2"]) + self.actor["b2"].unsqueeze(1)

    def critic_values(self, x):
        """(Vext, Vint) each [C, M]."""
        h = self._mlp2(self.critic, x)
        vext = torch.bmm(h, self.critic["Wext"]) + self.critic["bext"].unsqueeze(1)
        vint = torch.bmm(h, self.critic["Wint"]) + self.critic["bint"].unsqueeze(1)
        return vext.squeeze(-1), vint.squeeze(-1)

    def rnd_features(self, x):
        """(target_features detached, predictor_features), each [C, M, F] — layer-1 packed:
        target and predictor read the same input, so their first layers run as ONE GEMM."""
        W0 = torch.cat([self.target["W0"], self.predictor["W0"]], -1)
        b0 = torch.cat([self.target["b0"], self.predictor["b0"]], -1)
        h = torch.relu(torch.bmm(x, W0) + b0.unsqueeze(1))
        Hh = self.target["W0"].shape[-1]
        th, ph = h[..., :Hh], h[..., Hh:]
        with torch.no_grad():
            tf = torch.bmm(th, self.target["W1"]) + self.target["b1"].unsqueeze(1)
        ph = torch.relu(torch.bmm(ph, self.predictor["W1"]) + self.predictor["b1"].unsqueeze(1))
        pf = torch.bmm(ph, self.predictor["W2"]) + self.predictor["b2"].unsqueeze(1)
        return tf, pf

    def actor_critic(self, x):
        """Packed actor+critic forward on the same input [C, M, 4]: one layer-1 GEMM for
        both trunks, packed critic heads. Returns (mean [C,M,2], vext [C,M], vint [C,M])."""
        W0 = torch.cat([self.actor["W0"], self.critic["W0"]], -1)
        b0 = torch.cat([self.actor["b0"], self.critic["b0"]], -1)
        h = torch.tanh(torch.bmm(x, W0) + b0.unsqueeze(1))
        ha, hc = h[..., :64], h[..., 64:]
        ha = torch.tanh(torch.bmm(ha, self.actor["W1"]) + self.actor["b1"].unsqueeze(1))
        hc = torch.tanh(torch.bmm(hc, self.critic["W1"]) + self.critic["b1"].unsqueeze(1))
        mean = torch.bmm(ha, self.actor["W2"]) + self.actor["b2"].unsqueeze(1)
        Wh = torch.cat([self.critic["Wext"], self.critic["Wint"]], -1)
        bh = torch.cat([self.critic["bext"], self.critic["bint"]], -1)
        v = torch.bmm(hc, Wh) + bh.unsqueeze(1)
        return mean, v[..., 0], v[..., 1]

    def predictor_features(self, x):
        """Predictor features only [C, M, F] (the frozen target is hoisted per iteration)."""
        ph = torch.relu(torch.bmm(x, self.predictor["W0"]) + self.predictor["b0"].unsqueeze(1))
        ph = torch.relu(torch.bmm(ph, self.predictor["W1"]) + self.predictor["b1"].unsqueeze(1))
        return torch.bmm(ph, self.predictor["W2"]) + self.predictor["b2"].unsqueeze(1)

    def target_features(self, x):
        """Frozen target features [C, M, F], never with gradient."""
        with torch.no_grad():
            th = torch.relu(torch.bmm(x, self.target["W0"]) + self.target["b0"].unsqueeze(1))
            return torch.bmm(th, self.target["W1"]) + self.target["b1"].unsqueeze(1)

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
        """Action sampling only (compiled for the cuda-env backend); see _one_step_pure."""
        mean = self.actor_mean(obs)
        std = self.actor["logstd"].exp().unsqueeze(1)
        return mean + std * z

    def _one_step_pure(self, pos, vel, goal, sc, rc, obs, z,
                       out_obs, out_act, out_nobs, out_rext, out_term, out_done):
        """One rollout step: sample an action, step the env, write this step's buffer slices.

        The critic values, the log-probability and the RND bonus are deliberately NOT computed
        here. Each is a pure function of quantities the loop already stores (the observations,
        the action noise, the true next observations) and of parameters that do not change
        during a rollout, so computing them once after the loop over the whole [T, C, N] batch
        gives exactly the same numbers with a fraction of the kernels. Only the action (which
        the environment consumes) and the environment step are genuinely sequential.

        The per-step buffer writes are done INSIDE this function, into slices passed as
        arguments, so the compiler can fuse each write into the kernel that produces the value
        instead of leaving a separate copy kernel per buffer per step.

        Returns the carried environment state (pos', vel', goal', sc', rc', obs')."""
        out_obs.copy_(obs)
        mean = self.actor_mean(obs)
        std = self.actor["logstd"].exp().unsqueeze(1)
        action = mean + std * z
        (pos, vel, goal, sc, rc, nobs, r_ext, terminated, truncated,
         final_obs) = self.env.step_core(pos, vel, goal, sc, rc, action)
        out_act.copy_(action)
        out_nobs.copy_(final_obs)
        out_rext.copy_(r_ext)
        out_term.copy_(terminated.to(self.env.dtype))
        out_done.copy_((terminated | truncated).to(self.env.dtype))
        return pos, vel, goal, sc, rc, nobs

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
            pos, vel, goal, sc, rc, obs = step(
                pos, vel, goal, sc, rc, obs, self._Z[t],
                b["obs"][t], b["act"][t], b["nobs"][t], b["rext"][t], b["term"][t], b["done"][t])
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
            action = self._policy_fn(obs, self._Z[t])
            b["act"][t].copy_(action)
            obs, r_ext, terminated, truncated, final_obs = env.step(action)
            b["nobs"][t].copy_(final_obs)
            b["rext"][t].copy_(r_ext)
            b["term"][t].copy_(terminated.to(env.dtype))
            b["done"][t].copy_((terminated | truncated).to(env.dtype))

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
        """Run the post-rollout processing (compiled when the knob is set)."""
        self._post_fn()

    @torch.no_grad()
    def _post_body_impl(self):
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
        # before: buffers are [T, C, N, k]; after: [C, T*N, k], row index t*N + n
        flat = lambda x: x.permute(1, 0, 2, *range(3, x.dim())).reshape(C, T * N, *x.shape[3:])
        unflat = lambda x: x.view(C, T, N).permute(1, 0, 2)
        flat_obs, flat_nobs = flat(obs_buf), flat(nobs_buf)

        # --- quantities hoisted out of the T-step rollout loop (exact: pure functions of the
        # stored data and of parameters that did not change during the rollout) ---
        # log-probability of the sampled actions depends only on the noise and on logstd
        logstd = self.actor["logstd"]
        logp_buf.copy_((-0.5 * self._Z * self._Z - logstd.view(1, C, 1, 2)
                        - 0.5 * LOG2PI).sum(-1))
        # one critic pass covers both the on-step values and the bootstrap values
        vall_e, vall_i = self.critic_values(torch.cat([flat_obs, flat_nobs], dim=1))
        vext_buf.copy_(unflat(vall_e[:, :T * N]))
        vint_buf.copy_(unflat(vall_i[:, :T * N]))
        vext_next = unflat(vall_e[:, T * N:])
        vint_next = unflat(vall_i[:, T * N:])
        # intrinsic bonus on the true next observations, with the statistics as they stood at
        # the START of this iteration (the update below advances them) — spec section 4.2
        tf_old, pf_old = self.rnd_features(self.whiten(flat_nobs))
        rint_buf.copy_(unflat(0.5 * (pf_old - tf_old).square().sum(-1)))

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

        # flatten (C, T*N, ...) IN PLACE into the static batch buffers; the RND observation
        # statistics update precedes the whiten so the update phase sees the NEW statistics
        # (spec section 4.2), unlike the bonus above which used the old ones
        nobs_flat = flat_nobs
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

    def _backward(self, loss):
        """Produce every parameter gradient and return it in the form the optimizer reads.

        `loss.backward()` with a gradient already attached to each parameter ADDS the new
        gradient into it — reading the freshly produced gradient, reading the buffer and
        writing it back, three passes — and the buffer then has to be zeroed before the next
        step, a fourth. That buffer holds 245 megabytes at 1,024 copies and a gigabyte at
        4,096. Asking autograd for the gradients instead returns twenty-one freshly written
        tensors that nothing has to be added to, and never zeroed.

        What happens to them next is one of three things, chosen at build time (see the
        `gradient_buffer` and `fuse_copy_and_limit` fields).
        """
        return self._place_fn(list(torch.autograd.grad(loss, self.trainable)))

    def _place_where_written(self, grads):
        """No buffer: the optimizer reads the gradients where the backward pass wrote them."""
        return grads

    def _place_by_copying(self, grads):
        """Copy every gradient into the flat buffer; the gradient limit reads it afterwards."""
        torch._foreach_copy_(self.grad_windows, grads)
        return self._flat_grad

    def _place_and_measure(self, grads):
        """Copy every gradient into the flat buffer and sum its squares in the same pass."""
        self._copy_measure_fn(grads)
        return self._flat_grad

    def _copy_and_measure(self, grads):
        """Write the gradients into their windows and set the limiting factor, reading each once.

        Copying and then measuring touches every gradient three times per update step: the copy
        reads the fresh tensor and writes the window, and the limit reads the whole buffer again.
        Doing both jobs in one program removes that third pass — 981 megabytes per step at 4,096
        copies. The sum is the same one _grad_scale_per_tensor computes.
        """
        C = self.cfg.n_copies
        total = grads[0].reshape(C, 1, -1).square().sum(2)
        self.grad_windows[0].copy_(grads[0])
        for window, g in zip(self.grad_windows[1:], grads[1:]):
            total = total + g.reshape(C, 1, -1).square().sum(2)
            window.copy_(g)
        self._scale.copy_(
            (self.cfg.max_grad_norm / (total.sqrt() + 1e-6)).clamp(max=1.0))

    def _clip_per_copy_and_step(self, grads):
        """Per-copy gradient-norm clip (spec 10), then one Adam step over all copies.

        The limit is already set when it rode along with the copy, so there is nothing to run.
        """
        if self._scale_fn is not None:
            self._scale_fn(grads)
        self._adam_fn(grads)

    def _reset_optimizer_state(self):
        """Zero the optimizer moments and step count (after a graph build's warmup passes)."""
        self._m.zero_()
        self._v.zero_()
        self._adam_t.zero_()

    def _grad_scale(self, g):
        """The per-copy factor that limits the gradient norm: one reduction over the buffer.

        Kept as its own compiled function, separate from the Adam step below. Written together
        the compiler fuses them into a single program that both reduces and updates, and that
        program reaches only 2.4 of the card's 3.5 terabytes per second; as two programs, the
        reduction runs at 3.2 and the update at 3.5. The arithmetic is identical either way.
        """
        self._scale.copy_(
            (self.cfg.max_grad_norm / (g.square().sum(1, keepdim=True).sqrt() + 1e-6)
             ).clamp(max=1.0))

    def _grad_scale_per_tensor(self, grads):
        """The same factor summed over the twenty-one separate gradients instead of one buffer.

        The sum runs over the same numbers in a different order, so the result differs in the
        last bits of float32; tests/test_gradient_form_gpu.py repeats both in double precision,
        where they agree, and measures the single-precision distance.
        """
        C = self.cfg.n_copies
        total = grads[0].reshape(C, 1, -1).square().sum(2)
        for g in grads[1:]:
            total = total + g.reshape(C, 1, -1).square().sum(2)
        self._scale.copy_(
            (self.cfg.max_grad_norm / (total.sqrt() + 1e-6)).clamp(max=1.0))

    def _adam_flat(self, g):
        """One Adam step over the whole flat buffer, with the gradients already limited.

        Adam is elementwise, so the C rows of the buffer are C independent Adams. The learning
        rate is a [C, 1] column, which covers both a uniform run (every entry the same) and a
        sweep (one value per copy group) with the same arithmetic. The bias correction reads a
        device-side step count so the update stays correct inside a captured graph.
        """
        cfg = self.cfg
        b1, b2 = 0.9, 0.999
        # add_ rather than += : the latter rebinds the attribute to a NEW tensor, which a
        # captured graph would not see (it holds the original storage)
        self._adam_t.add_(1)
        bc1 = 1.0 - torch.pow(self._b1, self._adam_t)
        bc2 = 1.0 - torch.pow(self._b2, self._adam_t)
        with torch.no_grad():
            # written as one expression chain rather than a sequence of in-place operations:
            # the compiler then fuses the whole update into a single pass that reads the
            # gradient, the two moments and the parameters once and writes three of them back.
            # At large copy counts this buffer is hundreds of megabytes, so the number of
            # passes over it — not the number of kernels — is what sets the cost. The gradient
            # is NOT zeroed here: the next backward pass overwrites it (_backward).
            gs = g * self._scale
            m_new = self._m * b1 + gs * (1.0 - b1)
            v_new = self._v * b2 + gs * gs * (1.0 - b2)
            step = (m_new / bc1) * (self._lr_col * self._lr_scale) / \
                   ((v_new / bc2).sqrt() + cfg.adam_eps)
            self._m.copy_(m_new)
            self._v.copy_(v_new)
            self._flat.sub_(step)

    def _adam_per_tensor(self, grads):
        """The same Adam step, one program per parameter, reading the gradients where they lie.

        Identical arithmetic to _adam_flat: the moments and the parameters are the same windows
        onto the same two buffers, the same expression chain runs over each one, and the
        padding between windows is simply never visited instead of being multiplied by a zero
        gradient. Only the gradient's home differs.
        """
        cfg = self.cfg
        b1, b2 = 0.9, 0.999
        self._adam_t.add_(1)
        bc1 = 1.0 - torch.pow(self._b1, self._adam_t)
        bc2 = 1.0 - torch.pow(self._b2, self._adam_t)
        with torch.no_grad():
            for p, g, m, v, scale, lr in zip(self.param_windows, grads, self.m_windows,
                                             self.v_windows, self._scale_views, self._lr_views):
                gs = g * scale
                m_new = m * b1 + gs * (1.0 - b1)
                v_new = v * b2 + gs * gs * (1.0 - b2)
                step = (m_new / bc1) * (lr * self._lr_scale) / \
                       ((v_new / bc2).sqrt() + cfg.adam_eps)
                m.copy_(m_new)
                v.copy_(v_new)
                p.sub_(step)

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
        grads = self._backward(loss)
        self._clip_per_copy_and_step(grads)
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
                grads = self._backward(loss)
                self._clip_per_copy_and_step(grads)
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
            grads = self._backward(loss)
            self._clip_per_copy_and_step(grads)
        else:
            Brows = cfg.num_steps * cfg.n_envs
            mb_size = Brows // cfg.num_minibatches
            for e in range(cfg.update_epochs):
                # shuffle the whole batch ONCE per epoch into a static buffer; the minibatches
                # are then contiguous slices of it, which cost nothing. Before: nine gather
                # kernels in every one of the sixteen steps. After: nine per epoch. The rows
                # and their order are identical either way, so this is exact.
                # The gather writes STRAIGHT INTO the static buffer. Written as
                # `buffer.copy_(t.gather(...))` it allocates a whole second copy of the shuffled
                # batch and then copies it across: 1.2 gigabytes written and read again in every
                # epoch at 4,096 copies, which a kernel profile shows as 2.9 milliseconds of
                # device-to-device copying per iteration.
                idx = self._perm[e]
                for key in self._U_KEYS:
                    t = self._U[key]
                    ix = idx.unsqueeze(-1).expand(C, Brows, t.shape[-1]) if t.dim() == 3 else idx
                    torch.gather(t, 1, ix, out=self._UP[key])
                for k in range(cfg.num_minibatches):
                    mb = {key: self._UP[key][:, k * mb_size:(k + 1) * mb_size]
                          for key in self._U_KEYS}
                    loss = self._loss_fn(mb, style_a=False)
                    grads = self._backward(loss)
                    self._clip_per_copy_and_step(grads)
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
        self._reset_optimizer_state()

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
        self._reset_optimizer_state()

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
                    # clone: the fused-CUDA env returns a preallocated buffer that the next
                    # step overwrites, so appending it directly would stack N copies of the
                    # last step and prime the statistics with zero variance along time
                    batch.append(final_obs.clone())
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
                # one device tensor drives the schedule; the optimizer reads it every step,
                # whether the rate is uniform or one value per copy group
                self._lr_scale.fill_(1.0 - (it - 1.0) / num_iterations)
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

"""Batched Montezuma's Revenge in JAX — the platform's first discrete-action environment.

The game logic is JAXAtari's pure-JAX reimplementation of the hardest exploration game the
original random-network-distillation paper solved, wrapped in JAXAtari's own `AtariWrapper`
for the ALE-v5 conventions (sticky actions 0.25; no no-op starts — sticky actions are the
Machado et al. protocol that replaced them), and adapted here to the platform's environment
contract: `[C, N]`-batched pure reset/step with explicit auto-reset, frame skip 4, a 4-frame
stack of the flattened object-centric observation, and the room index / inventory / lives
appended so that different rooms do not alias (spec.md, deviations).

Unlike PointMaze and AntMaze the resets and steps are stochastic (sticky actions), so the
environment carries one PRNG key per environment inside the wrapper state, seeded from
(base_seed, copy seed index, environment index) so paired sweep groups meet the same episodes.

The platform runs with `jax_enable_x64=True` globally; JAXAtari's game logic assumes the jax
default (its integer constants then disagree across `lax.cond` branches), so — exactly as the
MJX AntMaze does — every call into the game is wrapped in `jax.enable_x64(False)`, read at
trace time.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import flatten_util

import jaxatari
from jaxatari.games.montezuma_revenge.core import get_room_idx
from jaxatari.wrappers import AtariWrapper

F32 = jnp.float32
N_ROOMS = 24          # distinct playable rooms (get_room_idx maps room ids onto 0..23)
FRAME_FEATURES = 192  # the flattened object-centric observation of one frame
EXTRA_FEATURES = 6    # room index, the 4 inventory slots, lives


@dataclass(frozen=True)
class MontezumaConfig:
    """All env-side knobs. The defaults are the specification `montezuma_oc4500_sticky@1`."""
    sticky_actions: float = 0.25     # ALE-v5 repeat_action_probability
    frame_skip: int = 4              # one environment step = 4 game frames, action held
    frame_stack: int = 4             # frames stacked into the observation
    max_episode_steps: int = 4500    # truncation cap in env steps (18,000 frames, the RND cap)
    clip_reward: bool = True         # reward = sign(score delta), the RND convention
    reward_shift: float = 0.0        # added to every step's reward


class EnvState(NamedTuple):
    """All mutable env state, every leaf batched over B = C*N environments."""
    atari: object              # AtariWrapper state (game state + per-env PRNG key)
    stack: jnp.ndarray         # [B, frame_stack, FRAME_FEATURES] float32
    step_count: jnp.ndarray    # [C, N] int32, env steps since this environment's last reset
    reset_count: jnp.ndarray   # [C, N] int32, completed episodes


class JaxMontezuma:
    """Holds the wrapped game and configuration and exposes pure reset/step functions."""

    action_kind = "discrete"
    n_actions = 18

    def __init__(self, cfg: MontezumaConfig, n_copies: int, n_envs: int, base_seed: int = 0,
                 dtype=jnp.float32, copy_seed_index=None):
        """Build the wrapped game and the per-environment key grid for one (C, N) shape."""
        if dtype is not jnp.float32:
            raise ValueError("JaxMontezuma produces float32 observations only")
        self.cfg, self.C, self.N, self.dtype = cfg, n_copies, n_envs, dtype
        self.B = n_copies * n_envs
        self.obs_dim = cfg.frame_stack * FRAME_FEATURES + EXTRA_FEATURES
        # the trainer sizes its action-noise draws by this; for a discrete actor it is the
        # number of actions (one Gumbel per action per step)
        self.act_dim = self.n_actions

        # the wrapper: v5 sticky actions, no no-op starts, no first-fire, the raw game's own
        # done (lives exhausted); the frame cap backstop equals the platform cap in frames
        self._atari = AtariWrapper(
            jaxatari.make("montezumarevenge"), sticky_actions=cfg.sticky_actions,
            episodic_life=False, first_fire=False, noop_max=0,
            max_frames_per_episode=cfg.max_episode_steps * cfg.frame_skip)

        # one key per environment from (base seed, copy seed index, env index): paired sweep
        # groups pass the same index list, so their sticky-action draws and episodes match
        seed_idx = (jnp.arange(n_copies, dtype=jnp.uint32) if copy_seed_index is None
                    else jnp.asarray(copy_seed_index, dtype=jnp.uint32))
        base = jax.random.PRNGKey(base_seed)
        keys = jax.vmap(lambda c: jax.vmap(lambda e: jax.random.fold_in(
            jax.random.fold_in(base, c), e))(jnp.arange(n_envs, dtype=jnp.uint32)))(seed_idx)
        self._init_keys = keys.reshape(self.B, -1)

        # coverage: one cell per playable room, all of them counting as open
        self.rows, self.cols = 1, N_ROOMS
        self.n_cells = N_ROOMS
        self.open_cells = jnp.ones((N_ROOMS,), bool)

        # the flattener for one frame's structured observation, fixed by the first reset
        with jax.enable_x64(False):
            obs0, _ = self._atari.reset(jax.random.PRNGKey(0))
            flat0, self._unravel = flatten_util.ravel_pytree(obs0)
        if flat0.shape[0] != FRAME_FEATURES:
            raise ValueError(f"the flattened frame has {flat0.shape[0]} features, not "
                             f"{FRAME_FEATURES}; update FRAME_FEATURES and the spec")

    def _single_reset(self, key):
        """One environment's reset -> (atari state, [frame_stack, F] stack)."""
        obs, ast = self._atari.reset(key)
        flat = flatten_util.ravel_pytree(obs)[0].astype(F32)
        return ast, jnp.tile(flat[None], (self.cfg.frame_stack, 1))

    def _obs_vec(self, atari, stack):
        """The platform observation [B, obs_dim]: the stack flattened, then room index,
        inventory and lives — the state features the object list does not carry.

        before: stack [B, 4, 192], room id 4, inventory [1,0,0,0], lives 5
        after:  [B, 774] = 768 stacked features + (1.0, 1, 0, 0, 0, 5)
        """
        room = get_room_idx(atari.env_state.room_id).astype(F32)[:, None]
        inv = atari.env_state.inventory.astype(F32)
        lives = atari.env_state.lives.astype(F32)[:, None]
        return jnp.concatenate([stack.reshape(self.B, -1), room, inv, lives], -1)

    def reset(self) -> EnvState:
        """Initial state: every environment reset under its own key, counters at zero.

        The copy at the end breaks buffer aliasing: one compiled reset can give two all-zero
        state fields the same device buffer, and the donated training state must not hold any
        buffer twice.
        """
        with jax.enable_x64(False):
            atari, stack = jax.vmap(self._single_reset)(self._init_keys)
            atari = jax.tree.map(jnp.copy, atari)
        return EnvState(atari, stack, jnp.zeros((self.C, self.N), jnp.int32),
                        jnp.zeros((self.C, self.N), jnp.int32))

    def respawn(self, state: EnvState):
        """Every environment freshly reset (keys advanced past the warm-up episodes), step
        counts zeroed, episode counter kept."""
        with jax.enable_x64(False):
            atari, stack = jax.vmap(self._single_reset)(state.atari.key)
            atari = jax.tree.map(jnp.copy, atari)   # break aliasing; see reset()
            obs = self._obs_vec(atari, stack).reshape(self.C, self.N, self.obs_dim)
        state = EnvState(atari, stack, jnp.zeros_like(state.step_count), state.reset_count)
        return state, obs

    def step(self, state: EnvState, act):
        """One env step (frame_skip frames at the held action) with auto-reset. Returns
        (state', obs, reward, terminated, truncated, final_obs) — the platform contract.

        `act` is int32 [C, N] (an index into the game's 18-action set)."""
        with jax.enable_x64(False):
            return self._step_x32(state, act)

    def _step_x32(self, state: EnvState, act):
        """The step body, traced with x64 off so the game's integer arithmetic stays int32."""
        action_b = act.reshape(self.B).astype(jnp.int32)

        def single_step(ast, stack, action):
            """One environment forward frame_skip frames; no reset decision yet."""
            def body(carry, _):
                a_state = carry
                obs, a_state, r, term, trunc, info = self._atari.step(a_state, action)
                return a_state, (flatten_util.ravel_pytree(obs)[0].astype(F32), r,
                                 term, trunc)
            ast, (frames, rs, terms, truncs) = jax.lax.scan(
                body, ast, None, length=self.cfg.frame_skip)
            stack = jnp.concatenate([stack[1:], frames[-1][None]], axis=0)
            return ast, stack, rs.sum(), terms.any(), truncs.any()

        atari, stack, reward_raw, term_b, trunc_wrapper = jax.vmap(single_step)(
            state.atari, state.stack, action_b)

        step_count = state.step_count + 1
        reward = jnp.sign(reward_raw) if self.cfg.clip_reward else reward_raw
        reward = reward.astype(F32).reshape(self.C, self.N) + self.cfg.reward_shift
        terminated = term_b.reshape(self.C, self.N)
        truncated = (trunc_wrapper.reshape(self.C, self.N)
                     | (step_count >= self.cfg.max_episode_steps)) & ~terminated
        done = terminated | truncated

        final_obs = self._obs_vec(atari, stack).reshape(self.C, self.N, self.obs_dim)

        # auto-reset: finished environments start a new episode under their advanced key, so
        # sticky-action draws never repeat across episodes
        fresh_atari, fresh_stack = jax.vmap(self._single_reset)(atari.key)
        done_b = done.reshape(self.B)
        pick = lambda new, old: jax.tree.map(
            lambda n, o: jnp.where(done_b.reshape((-1,) + (1,) * (o.ndim - 1)), n, o),
            new, old)
        atari = pick(fresh_atari, atari)
        stack = pick(fresh_stack, stack)
        step_count = jnp.where(done, 0, step_count)
        reset_count = state.reset_count + done.astype(jnp.int32)

        obs = self._obs_vec(atari, stack).reshape(self.C, self.N, self.obs_dim)
        return (EnvState(atari, stack, step_count, reset_count),
                obs, reward, terminated, truncated, final_obs)

    def cell_index(self, obs_flat):
        """Room index of [C, M, obs] observations -> [C, M] int32, for rooms-visited coverage.

        before: obs_flat[..., 768] = 4.0 (the room-index feature)
        after:  4
        """
        room = obs_flat[..., self.cfg.frame_stack * FRAME_FEATURES]
        return jnp.clip(room.astype(jnp.int32), 0, N_ROOMS - 1)

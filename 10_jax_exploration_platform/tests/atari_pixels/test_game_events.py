"""The game events of pixel Montezuma: the rules, the environment's report and the trainer's sums.

Run on a compute node. The rule tests edit states of the pinned game (corrected sprite pack); the
trainer tests use a stand-in environment that reports scripted events.
"""

import os
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple

from exploration_platform.atari_pixels.environment import CORRECTED_ASSET_ROOT

os.environ.setdefault("XDG_DATA_HOME", CORRECTED_ASSET_ROOT)
os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.game_events import (GAME_EVENT_NAMES, ITEM_PICKUP_EVENTS, game_events,
                                                           game_has_key, game_left_start_room)
from exploration_platform.atari_pixels.game_event_names import START_ROOM
from exploration_platform.atari_pixels.training import PixelPPORND, first_step_index

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/atari_pixel_campaign"
LEFT = 4
ARCHIVED_METRIC_NAMES = {"approximate_kl", "entropy", "episode_count", "episode_return_sum", "gradient_norm",
                         "intrinsic_reward_mean", "learning_rate", "loss", "policy_loss", "predictor_loss",
                         "reward_ext_sum", "rooms_visited", "steps_per_policy", "value_loss"}
EVENT_METRIC_NAMES = {"event_counts", "event_first_step", "room_first_step", "games_with_key", "games_left_start_room"}


def named(counts):
    """The non-zero counts of one event array by event name."""
    # before: [1, 0, 0, ..., 0] ; after: {"key_pickup": 1}
    return {name: int(count) for name, count in zip(GAME_EVENT_NAMES, np.asarray(counts)) if count}


@pytest.fixture(scope="module")
def start_state():
    """A fresh game of the pinned Montezuma, loaded from the corrected pack."""
    import jaxtari

    with jax.enable_x64(False):
        game = jaxtari.make("montezumarevenge")
        assert int(game.consts.INITIAL_ROOM_ID) == START_ROOM
        return game.reset(jax.random.PRNGKey(0))[1]


def test_event_names_agree_between_the_trainer_and_the_record_tools():
    """The standard-library fold reads the same names, in the same order, as the event rules use."""
    sys.path.insert(0, str(SCRIPTS))
    import game_records

    names, start_room = game_records.load_event_names()
    assert names == GAME_EVENT_NAMES and start_room == START_ROOM
    assert len(set(GAME_EVENT_NAMES)) == len(GAME_EVENT_NAMES) == 13


def test_an_unchanged_state_has_no_event(start_state):
    """Comparing a state with itself counts nothing; the result has one int32 count per name."""
    with jax.enable_x64(False):
        counts = game_events(start_state, start_state, START_ROOM)
    assert counts.shape == (len(GAME_EVENT_NAMES),) and counts.dtype == jnp.int32
    assert named(counts) == {}
    assert not bool(game_has_key(start_state, START_ROOM))
    assert not bool(game_left_start_room(start_state, START_ROOM))


def test_each_item_type_gives_its_own_pickup_event(start_state):
    """An item that was in the room and is gone counts once, under the name of its type."""
    with jax.enable_x64(False):
        # The start room's item 0 is the key (type 0); other types are written into the same place.
        assert int(start_state.items_active[0]) == 1 and int(start_state.items_type[0]) == 0
        for code, name in enumerate(ITEM_PICKUP_EVENTS):
            before = start_state.replace(items_type=start_state.items_type.at[0].set(code))
            after = before.replace(items_active=before.items_active.at[0].set(0))
            assert named(game_events(before, after, START_ROOM)) == {name: 1}
        # Two items picked up in one agent step count separately.
        before = start_state.replace(items_active=jnp.asarray([1, 1, 0], jnp.int32),
                                     items_type=jnp.asarray([1, 1, 0], jnp.int32))
        after = before.replace(items_active=jnp.zeros(3, jnp.int32))
        assert named(game_events(before, after, START_ROOM)) == {"gem_pickup": 2}
        # The key of the start room is the game's key fact once it is gone.
        taken = start_state.replace(items_active=start_state.items_active.at[0].set(0))
        assert bool(game_has_key(taken, START_ROOM))


def test_door_and_kill_follow_the_key_and_sword_counts(start_state):
    """A key used on a door and a sword used on an enemy each count once; a pickup alone counts neither."""
    with jax.enable_x64(False):
        holding = start_state.replace(inventory=jnp.asarray([1, 1, 0, 0], jnp.int32))
        opened = holding.replace(inventory=jnp.asarray([0, 1, 0, 0], jnp.int32))
        killed = holding.replace(inventory=jnp.asarray([1, 0, 0, 0], jnp.int32))
        assert named(game_events(holding, opened, START_ROOM)) == {"door_opened": 1}
        assert named(game_events(holding, killed, START_ROOM)) == {"enemy_killed": 1}
        assert named(game_events(start_state, holding, START_ROOM)) == {}


def test_leaving_the_start_room_and_the_state_of_its_doors(start_state):
    """An exit counts once; it is an exit with both doors locked only when the stored doors say so."""
    with jax.enable_x64(False):
        assert np.asarray(start_state.global_doors_active[START_ROOM]).tolist() == [1, 1]
        # The item arrays of the new room differ from the old room's: no pickup may be counted.
        outside = start_state.replace(room_id=jnp.int32(START_ROOM + 1), items_active=jnp.zeros(3, jnp.int32))
        assert named(game_events(start_state, outside, START_ROOM)) == {
            "left_start_room": 1, "left_start_room_with_both_doors_locked": 1}
        through_door = outside.replace(
            global_doors_active=outside.global_doors_active.at[START_ROOM, 1].set(0))
        assert named(game_events(start_state, through_door, START_ROOM)) == {"left_start_room": 1}
        # Moving between two other rooms, or coming back, is not an exit from the start room.
        further = outside.replace(room_id=jnp.int32(START_ROOM + 2))
        assert named(game_events(outside, further, START_ROOM)) == {}
        assert named(game_events(outside, start_state, START_ROOM)) == {}
        # The game facts: outside now, or back inside with a door that was open when it was left.
        assert bool(game_left_start_room(outside, START_ROOM))
        back = through_door.replace(room_id=jnp.int32(START_ROOM))
        assert bool(game_left_start_room(back, START_ROOM))
        # Elsewhere the start room's key is read from the per-room store.
        assert not bool(game_has_key(outside, START_ROOM))
        stored = outside.replace(global_items_active=outside.global_items_active.at[START_ROOM, 0].set(0))
        assert bool(game_has_key(stored, START_ROOM))


def test_a_lost_life_is_counted_under_the_games_cause(start_state):
    """The cause codes 1, 2 and 3 are fall, enemy and laser; any other value is the other cause."""
    with jax.enable_x64(False):
        for code, name in ((1, "life_lost_by_fall"), (2, "life_lost_by_enemy"), (3, "life_lost_by_laser"),
                           (0, "life_lost_by_other_cause"), (7, "life_lost_by_other_cause")):
            after = start_state.replace(lives=start_state.lives - 1, death_type=jnp.int32(code))
            assert named(game_events(start_state, after, START_ROOM)) == {name: 1}
        # A cause code without a lost life (the death animation still running) counts nothing.
        animation = start_state.replace(death_type=jnp.int32(1))
        assert named(game_events(animation, animation, START_ROOM)) == {}


def test_walking_off_the_start_ledge_counts_one_fall():
    """The environment reports one lost life by fall, at the agent step that loses it, and nothing else."""
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    environment = AtariPixelEnvironment(num_envs=2, noop_max=0, sticky_action_probability=0.0,
                                        cache_room_backgrounds=True, render_final_two_only=True,
                                        report_game_events=True)
    state, _ = jax.jit(environment.reset)(jnp.asarray([3], jnp.int32))
    step = jax.jit(environment.step)
    actions = jnp.full((1, 2), LEFT, jnp.int32)
    totals, lost_at = np.zeros((2, len(GAME_EVENT_NAMES)), np.int64), None
    for index in range(120):
        state, _, _, _, info = step(state, actions)
        assert info["game_events"].shape == (1, 2, len(GAME_EVENT_NAMES))
        assert info["room_after"].shape == (1, 2) and info["game_has_key"].shape == (1, 2)
        events = np.asarray(info["game_events"][0])
        totals += events
        lost = np.asarray(info["life_lost"][0])
        # The event and the environment's own life boundary name the same agent step.
        np.testing.assert_array_equal(events[:, -4:].sum(axis=1) > 0, lost)
        if lost.any():
            lost_at = index
            break
    assert lost_at is not None, "holding LEFT from the start never lost a life in 120 agent steps"
    for lane in range(2):
        assert named(totals[lane]) == {"life_lost_by_fall": 1}
    np.testing.assert_array_equal(info["room_after"], [[START_ROOM, START_ROOM]])
    assert not bool(np.asarray(info["game_has_key"]).any())
    assert not bool(np.asarray(info["game_left_start_room"]).any())


def test_environment_without_the_option_reports_no_event_fields():
    """With report_game_events off the step's information has exactly the fields it had before."""
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    environment = AtariPixelEnvironment(num_envs=2, noop_max=0, cache_room_backgrounds=True,
                                        render_final_two_only=True)
    state, _ = jax.jit(environment.reset)(jnp.asarray([3], jnp.int32))
    _, _, _, _, info = jax.jit(environment.step)(state, jnp.zeros((1, 2), jnp.int32))
    assert set(info) == {"episode_return", "episode_done", "raw_reward", "game_over", "life_lost", "truncated",
                         "final_observation", "full_game_reset"}


class WaitingFrameGameState(NamedTuple):
    """The fields of the game state that the adapter and the event rules read."""

    position: jax.Array
    lives: jax.Array
    room_id: jax.Array
    items_active: jax.Array
    items_type: jax.Array
    inventory: jax.Array
    global_doors_active: jax.Array
    global_items_active: jax.Array
    death_type: jax.Array


class WaitingFrameGame:
    """A scripted game in which things happen only inside the waiting frame of a life-loss reset."""

    consts = SimpleNamespace(INITIAL_ROOM_ID=START_ROOM)

    def reset(self, key):
        """Start at position zero with three lives, the start room's key present and both doors locked."""
        del key
        state = WaitingFrameGameState(
            jnp.int32(0), jnp.int32(3), jnp.int32(START_ROOM), jnp.asarray([1, 0, 0], jnp.int32),
            jnp.zeros(3, jnp.int32), jnp.zeros(4, jnp.int32), jnp.ones((START_ROOM + 1, 2), jnp.int32),
            jnp.ones((START_ROOM + 1, 3), jnp.int32), jnp.int32(0))
        return state.position, state

    def step(self, state, action):
        """Move by the action value; lose a life by a fall when crossing position 4 or 20.

        The reset's FIRE frame (action 1) picks up the key when two lives are left and moves to the
        next room when one life is left. A held action of 2 never does either.
        """
        position = state.position + action
        crossed = ((state.position < 4) & (position >= 4)) | ((state.position < 20) & (position >= 20))
        fire = action == 1
        picks_key, changes_room = fire & (state.lives == 2), fire & (state.lives == 1)
        state = state._replace(
            position=position, lives=state.lives - crossed.astype(jnp.int32),
            death_type=jnp.where(crossed, 1, state.death_type),
            items_active=jnp.where(picks_key, state.items_active.at[0].set(0), state.items_active),
            inventory=jnp.where(picks_key, state.inventory.at[0].add(1), state.inventory),
            room_id=jnp.where(changes_room, START_ROOM + 1, state.room_id))
        return position, state, action.astype(jnp.float32), position >= 100, {}


def waiting_frame_environment():
    """The pixel adapter around the scripted game: no NOOP frames, no sticky actions, events reported."""
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    class ScriptedPixelEnvironment(AtariPixelEnvironment):
        """Exercise the adapter's step and reset without the real game or its renderer."""

        def __init__(self):
            """Set the adapter's settings directly; the real constructor would load the game."""
            self.num_envs, self.frame_skip, self.max_episode_steps, self.noop_max = 2, 4, 108000, 0
            self.episodic_life, self.sticky_action_probability = True, 0.0
            self.render_final_two_only = self.render_final_two_together = False
            self.frame_skip_unroll, self.batch_layout, self.stack_update = 1, "flat", "concatenate"
            self.reset_only_when_needed = True
            self.fuse_pixel_preprocessing = self.combine_palette_and_sprite_drawing = False
            self.report_game_events = True
            self.game = WaitingFrameGame()

        def _frame(self, game):
            """Represent the game's position as a single grayscale pixel."""
            return jnp.asarray([[game.position]], dtype=jnp.uint8)

        def _resize(self, gray):
            """Expand the single pixel into the learner's 84 by 84 shape."""
            return jnp.broadcast_to(gray, (84, 84))

    return ScriptedPixelEnvironment()


def test_events_inside_the_waiting_frames_of_a_life_loss_reset_are_counted():
    """A pickup or an exit during the reset after a lost life belongs to the agent step that lost it."""
    environment = waiting_frame_environment()
    state, _ = jax.jit(environment.reset)(jnp.asarray([7], jnp.int32))
    step = jax.jit(environment.step)
    held = jnp.full((1, 2), 2, jnp.int32)
    # Step 1: positions 1 -> 9 cross 4 (a fall); the reset's FIRE frame then picks up the key.
    state, _, _, done, info = step(state, held)
    assert bool(done.all()) and bool(info["life_lost"].all()) and not bool(info["full_game_reset"].any())
    for lane in range(2):
        assert named(info["game_events"][0, lane]) == {"key_pickup": 1, "life_lost_by_fall": 1}
    np.testing.assert_array_equal(info["room_after"], [[START_ROOM, START_ROOM]])
    np.testing.assert_array_equal(state.game.inventory[:, 0], [1, 1])
    # Step 2: positions 10 -> 18, nothing happens and no reset runs.
    state, _, _, done, info = step(state, held)
    assert not bool(done.any())
    assert named(info["game_events"][0, 0]) == {}
    # Step 3: positions 18 -> 26 cross 20 (a fall); the reset's FIRE frame then leaves the start room.
    state, _, _, done, info = step(state, held)
    assert bool(done.all())
    for lane in range(2):
        assert named(info["game_events"][0, lane]) == {
            "left_start_room": 1, "left_start_room_with_both_doors_locked": 1, "life_lost_by_fall": 1}
    np.testing.assert_array_equal(info["room_after"], [[START_ROOM + 1, START_ROOM + 1]])


def test_a_new_game_after_game_over_adds_no_event():
    """The new game is another game: comparing it with the ended one must not count anything."""
    environment = waiting_frame_environment()
    state, _ = jax.jit(environment.reset)(jnp.asarray([7], jnp.int32))
    # One frame from game over, holding a key: the new game holds none, which is not a door opening.
    game = state.game._replace(position=jnp.full_like(state.game.position, 98),
                               lives=jnp.zeros_like(state.game.lives),
                               room_id=jnp.full_like(state.game.room_id, START_ROOM + 1),
                               inventory=state.game.inventory.at[:, 0].set(1))
    state, _, _, done, info = jax.jit(environment.step)(state._replace(game=game), jnp.full((1, 2), 2, jnp.int32))
    assert bool((done & info["game_over"] & info["full_game_reset"]).all())
    assert named(info["game_events"][0, 0]) == {} and named(info["game_events"][0, 1]) == {}
    # The room and the game facts are those of the ended game, not of the new one.
    np.testing.assert_array_equal(info["room_after"], [[START_ROOM + 1, START_ROOM + 1]])
    assert bool(info["game_left_start_room"].all())
    np.testing.assert_array_equal(state.game.room_id, [START_ROOM, START_ROOM])


def test_first_step_index_marks_absent_flags_with_minus_one():
    """The first true step along time, per flag; a flag that is never true gives -1."""
    flags = jnp.asarray([[False, False, True], [False, True, True], [False, True, False]])
    np.testing.assert_array_equal(first_step_index(flags), [-1, 1, 0])


class ScriptedGame(NamedTuple):
    """The one field of the game state the trainer reads: the room of every environment."""

    room_id: jax.Array


class ScriptedState(NamedTuple):
    """The stand-in environment's state: a step counter per environment and the game's rooms."""

    counters: jax.Array
    game: ScriptedGame


class ScriptedEventEnvironment:
    """A stand-in environment that reports fixed events as a function of its step counter."""

    action_count = 18
    room_count = 3
    report_game_events = True

    def __init__(self, num_envs):
        """Hold the fixed number of vector environments for each independent policy."""
        self.num_envs = num_envs

    def observe(self, counters):
        """Make four distinct uint8 frames from each environment's integer counter."""
        offsets = jnp.arange(4, dtype=jnp.int32)[None, None, :, None, None]
        return jnp.broadcast_to(((counters[..., None, None, None] + offsets) % 256).astype(jnp.uint8),
                                 counters.shape + (4, 84, 84))

    def reset(self, seeds):
        """Start every vector environment at zero while retaining an independent policy axis."""
        counters = jnp.zeros((len(seeds), self.num_envs), jnp.int32)
        return ScriptedState(counters, ScriptedGame(jnp.zeros_like(counters))), self.observe(counters)

    def step(self, state, actions):
        """Advance the counter; a game ends at every third count; events follow the script below."""
        counters = state.counters + 1
        done = counters % 3 == 0
        lane = jnp.broadcast_to(jnp.arange(self.num_envs, dtype=jnp.int32), counters.shape)
        # Script, with the counter c after the step and the environment index n:
        # - key_pickup (column 0): once in environment 1 at c == 2
        # - door_opened (column 5): once in every environment at c == 4 and at c == 6
        # - room after the step: environment 0 is in room 2 from c == 3 on, everything else in room 0
        # - the game has the key: environment 0 always; it has left the start room: environment 1 from c == 4
        events = jnp.zeros(counters.shape + (len(GAME_EVENT_NAMES),), jnp.int32)
        events = events.at[..., 0].set(((lane == 1) & (counters == 2)).astype(jnp.int32))
        events = events.at[..., 5].set(((counters == 4) | (counters == 6)).astype(jnp.int32))
        rooms = jnp.where((lane == 0) & (counters >= 3), 2, 0).astype(jnp.int32)
        return ScriptedState(counters, ScriptedGame(rooms)), self.observe(counters), \
            (actions % 2).astype(jnp.float32), done, {
                "episode_return": counters.astype(jnp.float32), "episode_done": done,
                "final_observation": self.observe(counters), "full_game_reset": done,
                "game_events": events, "room_after": rooms,
                "game_has_key": lane == 0, "game_left_start_room": (lane == 1) & (counters >= 4)}


def test_trainer_sums_events_per_copy_and_keeps_every_other_number():
    """One update reports the scripted counts, first steps and game facts; nothing else changes."""
    configuration = PixelPPOConfig(n_copies=2, n_envs=2, num_steps=6, num_minibatches=1, update_epochs=1,
                                   observation_warmup_iterations=0, total_timesteps=256)
    environment = ScriptedEventEnvironment(2)
    plain = PixelPPORND(configuration, environment)
    state = plain.warmup(plain.initialize((19, 37)))
    expected, expected_metrics = plain.train_iteration(state)
    recording = PixelPPORND(replace(configuration, record_game_events=True), environment)
    actual, metrics = recording.train_iteration(state)
    jax.block_until_ready(actual)
    assert set(expected_metrics) == ARCHIVED_METRIC_NAMES
    assert set(metrics) == ARCHIVED_METRIC_NAMES | EVENT_METRIC_NAMES

    # Rollout step t ends with counter t + 1. Both copies run the same script.
    counts = np.zeros(len(GAME_EVENT_NAMES), np.int64)
    counts[0], counts[5] = 1, 4
    first = np.full(len(GAME_EVENT_NAMES), -1)
    first[0], first[5] = 1, 3
    np.testing.assert_array_equal(metrics["event_counts"], [counts, counts])
    np.testing.assert_array_equal(metrics["event_first_step"], [first, first])
    np.testing.assert_array_equal(metrics["room_first_step"], [[0, -1, 2], [0, -1, 2]])
    # Games end at counters 3 and 6 in both environments: two of them with the key (environment
    # 0 twice), one of them after leaving the start room (environment 1 at counter 6).
    np.testing.assert_array_equal(metrics["episode_count"], [4, 4])
    np.testing.assert_array_equal(metrics["games_with_key"], [2, 2])
    np.testing.assert_array_equal(metrics["games_left_start_room"], [1, 1])

    # Recording the events changes no other number of the update.
    for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(before, after)
    for name in ARCHIVED_METRIC_NAMES:
        np.testing.assert_array_equal(expected_metrics[name], metrics[name])


def test_trainer_refuses_event_recording_without_an_environment_report():
    """The learner option needs the environment option; otherwise the trainer says so at once."""
    configuration = PixelPPOConfig(n_envs=2, num_steps=2, num_minibatches=1, update_epochs=1,
                                   observation_warmup_iterations=0, total_timesteps=64, record_game_events=True)
    silent = ScriptedEventEnvironment(2)
    silent.report_game_events = False
    with pytest.raises(ValueError, match="report_game_events=True"):
        PixelPPORND(configuration, silent)

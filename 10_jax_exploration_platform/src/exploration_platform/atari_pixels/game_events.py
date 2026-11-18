"""Events of one agent step of pixel Montezuma, read from the game state before and after it.

Everything here is an integer comparison of two game states. Nothing changes the game, its random
numbers or its pixels.

The environment compares the game before and after the four frames of an agent step, and, for a
game that goes on after a lost life, before and after the waiting frames of that reset.

Known limits:
- a pickup followed by a room change inside the same comparison is not seen, because the item
  arrays then belong to the new room;
- a gem picked up in the bonus room reappears in the same frame and is not seen;
- a key picked up and used on a door inside the same comparison leaves the key count unchanged, so
  that door is not seen (the pickup is);
- the waiting frames at the start of a new game are not compared (the player stands at the start);
- the random steps of the observation warm-up before the first update are not recorded. A game
  can therefore end in an early update with the key although no pickup event was recorded.
"""

import jax.numpy as jnp

from .game_event_names import GAME_EVENT_NAMES, ITEM_PICKUP_EVENTS, LIFE_LOST_EVENTS  # noqa: F401


def game_events(before, after, start_room):
    """Count every event of GAME_EVENT_NAMES between two states of one game: int32 [events]."""
    # Item pickups: an item of the room that was there before and is gone after, by its type.
    # before: items_active [1, 0, 0], items_type [0, 0, 0] ; after: items_active [0, 0, 0]
    # gives key_pickup 1 and every other pickup 0. A room change replaces the item arrays, so
    # nothing is counted then.
    same_room = before.room_id == after.room_id
    picked = (before.items_active == 1) & (after.items_active == 0) & same_room
    pickups = [jnp.sum(picked & (before.items_type == code)) for code in range(len(ITEM_PICKUP_EVENTS))]

    # A door takes one key and a kill takes one sword; nothing else lowers either count.
    # before: inventory [1, 0, 0, 0] ; after: [0, 0, 0, 0] gives door_opened 1.
    doors_opened = jnp.maximum(before.inventory[0] - after.inventory[0], 0)
    enemies_killed = jnp.maximum(before.inventory[1] - after.inventory[1], 0)

    # Leaving the start room. The game stores a room's doors when the room is left, so the stored
    # doors of the start room say whether both were still locked at that moment.
    left = (before.room_id == start_room) & (after.room_id != start_room)
    left_locked = left & jnp.all(after.global_doors_active[start_room] == 1)

    # A lost life, by the game's own cause code (1 fall or pit, 2 enemy, 3 laser).
    lost = after.lives < before.lives
    by_cause = [lost & (after.death_type == code) for code in range(1, len(LIFE_LOST_EVENTS))]
    by_other_cause = lost & ((after.death_type < 1) | (after.death_type >= len(LIFE_LOST_EVENTS)))

    # One count per name of GAME_EVENT_NAMES, in that order.
    counts = [*pickups, doors_opened, enemies_killed, left, left_locked, *by_cause, by_other_cause]
    return jnp.stack([jnp.asarray(count, jnp.int32) for count in counts])


def game_has_key(game, start_room):
    """True when this game has picked up the start room's key (read when the game ends)."""
    # While the player is in the start room the room's own item array is current; elsewhere the
    # game keeps the start room's items in its per-room store. The key is item 0 of that room.
    return jnp.where(game.room_id == start_room, game.items_active[0] == 0,
                     game.global_items_active[start_room, 0] == 0)


def game_left_start_room(game, start_room):
    """True when this game has left the start room at some time (read when the game ends)."""
    # Outside the start room now, or back in it with a door that was open when it was last left.
    return (game.room_id != start_room) | jnp.any(game.global_doors_active[start_room] == 0)

"""Names of the game events of pixel Montezuma, in the order of the columns of the event arrays.

This file imports nothing, so the standard-library record tools
(scripts/atari_pixel_campaign/game_records.py) load it by its path without importing a numerical
library. The rules that produce the events are in game_events.py.
"""

# One pickup event per item type, in the order of the game's item type codes 0 to 4.
ITEM_PICKUP_EVENTS = ("key_pickup", "gem_pickup", "amulet_pickup", "sword_pickup", "torch_pickup")
# One event per cause of a lost life: the game's death_type codes 1, 2, 3, then any other value.
LIFE_LOST_EVENTS = ("life_lost_by_fall", "life_lost_by_enemy", "life_lost_by_laser",
                    "life_lost_by_other_cause")
GAME_EVENT_NAMES = ITEM_PICKUP_EVENTS + (
    "door_opened", "enemy_killed", "left_start_room",
    "left_start_room_with_both_doors_locked") + LIFE_LOST_EVENTS
# The room every game starts in (the game's INITIAL_ROOM_ID; a test compares the two).
START_ROOM = 4

# Exploration metrics on pixel Montezuma's Revenge

The Atari part of `exploration-metrics.md`, for the pixel Montezuma study that started 2026-11-17 (plan
`~/.claude/plans/2026-11-17-19-00_montezuma-fast-last-layer-against-rnd.md`). User decisions of 2026-11-17.

## The game as implemented

- The game is the JAXtari recreation with the corrected room geometry, not the Atari original: 24 rooms on 4 floors,
  start room 4, 6 lives per game, at most 27,000 agent steps per game (frame skip 4), sticky actions 0.25.
- Points differ from the Atari original: key 100, door 300, gem, sword and torch 1,000 each, amulet 100, enemy killed
  with the sword 100. Scores are therefore never compared with published numbers.
- **The key must be picked up before the agent can leave the start room.** The start room's only exits are its two
  locked doors on the top floor (the collision map has no other opening the player can reach); a door opens only for
  a player holding a key, and the room has one key. Checked 2026-11-17 from the collision map, the game code and the
  archived baseline's records (the first game leaving the room scored exactly 400 = key + door). The records count
  exits from the start room with both doors still locked; the expected count is zero, and any other count is
  reported as an error of the logging or the game.

## The primary metric: mean score over the run

Define a policy's score per game as the raw game score of a finished training game (not the clipped learning reward),
and a window as 61 updates (999,424 agent steps per policy).

- **Window value:** the mean score of the policy's games that finished in the window; a window with no finished game
  takes the previous window's value (0 before the first).
- **Mean score over the run (chooses configurations):** $S(B)$, the mean of the window values up to the budget $B$
  and over the $K$ policies (the area under the score curve divided by the budget).
- **Final level:** $S_{\text{last}}(B)$, the same mean over the last 10% of the windows.

Ties in $S(B)$ are broken by the larger $S_{\text{last}}(B)$.

## What "ahead" means

The method is ahead of RND when, on seeds that did not choose either configuration, the 95% interval over seeds of
the difference in $S(B)$ (method minus RND) is entirely above zero and the 95% interval of the difference in
$S_{\text{last}}(B)$ is not entirely below zero. Intervals resample seeds, never games or windows.

## Reported beside it (never used to choose)

Define $T_i$ as the training step at which policy $i$ first leaves the start room ($T_i = \infty$ if it never does).

1. **Mean capped first exit from the start room:** $E(B) = \dfrac{1}{K} \sum_{i=1}^{K} \min(T_i, B)$, smaller is
   better; ties broken by the share of policies that left.
2. **Rooms reached:** the number of the 24 rooms a policy has been in at least once by $B$.
3. **First-time steps** of the first key, first door, first exit (and the room it led to), each room, each item type
   and the first enemy killed, each as `(update - 1) * 16,384 + (rollout step + 1) * 128`, exact to one step of the
   policy's 128 environments. Events of the 50 random warm-up rollouts before training are not counted.
4. **Per window:** the share of finished games that picked up the key and that left the start room, and lives lost by
   cause (fall, enemy, laser).
5. **Score levels:** the first window whose mean score per game is at or above 400, 1,000 and 2,500.

## The four parts of `exploration-metrics.md`

| part | definition here |
|---|---|
| goal event | leaving the start room (which needs the key first); first-time steps of all events are recorded |
| coverage unit | the 24 rooms; rooms reached by $B$ |
| evaluation episode | none separate: every measure is read from the training games |
| budget $B$ | 200 million steps per policy in the sweep, 1 billion and 2 billion in the later stages; the learning-rate schedule always has a 2-billion-step horizon |

User's words (2026-11-17): "mean score over the run, but also report the other two as metrics, also if the agent has
to pick up the key before leaving the first room, also write this down".

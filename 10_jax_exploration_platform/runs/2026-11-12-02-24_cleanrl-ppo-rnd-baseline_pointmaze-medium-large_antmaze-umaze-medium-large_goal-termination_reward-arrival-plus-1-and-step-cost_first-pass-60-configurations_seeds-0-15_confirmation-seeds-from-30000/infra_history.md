# Infrastructure history

Times are Pacific.

- 2026-11-12 02:27 to 04:47 PT: PointMaze first pass, 16 units on 16 cards under both accounts (RTX 5080, A40,
  RTX 4000 Ada, RTX A4000), none failed. Measured per worker card: about one busy processor and 2.6 GiB of host
  memory (sacct MaxRSS 2.5 GiB for a one-card job, 10.4 GiB for a four-card job) against the 8 processors and 16 GiB
  asked for.
- 2026-11-12 10:48 PT: PointMaze second pass, 16 units. The planner found no node with a free card and room for
  8 processors and 16 GiB: the nodes with free cards have their memory held by processor-only jobs of other
  sessions of these two accounts. It therefore sent 16 workers that ask for a card type on any node, 8 of them
  for H100 NVL cards and 4 for A100 cards, which may wait for days here.
- 2026-11-12 10:52 to 10:56 PT: the submission got options for the request per card, and the workers now ask for
  2 processors and 4 GiB per card (the measured need plus a margin). Waiting workers cancelled by their ledger ids,
  none had started: 6884319 to 6884327 and 6884333 to 6884335 (the 12 that asked for H100 NVL or A100 cards), then
  6884457 to 6884462 (6 that asked for A40 cards, all in use) and 6884318, 6884336, 6884337, 6884338 (4 with the
  large request that waited on priority). Sent instead: 6884454, 6884455, 6884463, 6884467, 6884468, 6884456, and
  6884493 to 6884496, on RTX 2080 Ti, RTX 4000 Ada, GTX 1080 Ti and RTX 5080 cards. All 16 units were running by
  10:57 PT (times read from the scheduler's records; an earlier version of this entry gave later times that I
  had estimated instead of reading).
- The planner's list of card classes is older than the hardware: lynx01, lynx02, lynx05 and affogato13 now hold
  RTX 4000 Ada cards, which it does not pin to. Workers that ask for that card type on any node are placed there by
  the scheduler.
- 2026-11-12 12:07 to 12:24 PT: the three units of "development at the budget" waited with no worker.
  The planner counted two cards of the four-card job 6884496 as idle workers, but their worker processes had ended
  when the queue was empty (the job goes on for its other two cards). The planner now counts only the cards of
  waiting jobs as workers without a unit. The pinned worker 6886277 (ai02) waits on priority; workers 6886680
  (ai04) and 6886681 (any RTX 5080) were added at 12:24 PT (times from the scheduler's records).
- 2026-11-12 12:24 to 12:27 PT: the three added workers did not start although cards, processors and memory were
  free on several nodes. The card partition had 3,334 waiting jobs (2,901 of one other user), and our accounts'
  priority is low (815 to 1,220), so the scheduler did not reach them. At 12:26:54 PT one worker was started on an
  idle card inside job 6884496 (lynx04, yuxinchen; two of its four cards had finished their units), through
  `srun --jobid=6884496 --overlap`; its log is `logs/worker_on_idle_card_of_job_6884496.log`. The three waiting
  workers stay in the queue; whichever starts finds the remaining units or exits.

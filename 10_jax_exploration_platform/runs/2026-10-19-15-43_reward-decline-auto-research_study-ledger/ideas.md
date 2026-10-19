# Ideas (20 planned), status

| id | batch | idea | run folder | outcome |
|---|---|---|---|---|
| E1 | 1 | episode ends at the goal, +1 once | | planned |
| E2 | 1 | episode ends at the goal; 0 at the goal, -1 elsewhere | | planned |
| E3 | 1 | episode ends at the goal; pays remaining steps of the cap at once | | planned |
| A1 | 1 | intrinsic weight decayed linearly to 0 | | planned |
| A2 | 1 | intrinsic weight x (1 - running goal-reaching rate) | | planned |
| A3 | 1 | clipped PPO, 4 epochs x 4 minibatches | | planned |
| A4 | 1 | entropy bonus 0.01 | | planned |
| A5 | 1 | task reward / max(1, running std of its discounted sum) | | planned |
| A6 | 1 | extrinsic discount 0.99 | | planned |
| A7 | 1 | gradient clip per network | | planned |

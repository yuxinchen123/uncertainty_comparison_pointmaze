# What the trained ants do, against reference MuJoCo

Measurements of 2026-11-13, 08:46 to 10:35 PT (Slurm jobs 6905236 to 6905566, ledger
`slurm/submitted_jobids_motion-check_sl5nw.txt` of this folder). The measurements and scripts are by a background
worker of the main session; it stopped (usage limit) before writing these notes, and the main session wrote them
from its records between 10:49 and 10:52 PT. Times are Pacific, converted from machines running Eastern.

## Question

Late in the longer RND runs on the U-maze (phase `budget-probe`; this folder's `infra_history.md`, entry of 07:08
PT) 9 to 18% of all steps were above the walls, torso speeds reached 14 to 26 m/s, and the fault guard (30 m/s,
100 rad/s) fired up to once per 0.3 million steps in the best configurations. Is that motion something reference
MuJoCo also allows, or something only our physics (MuJoCo Warp, implicitfast, 4 solver iterations, 0.005 s, 10
physics steps per environment step) allows? And does the return depend on it?

## What was run

1. **Policies:** the two `budget-probe` checkpoints, copied from Gilbreth (`checkpoint_copy/`):
   - 16 environments per policy, iteration 19,328: 39.6 million environment steps per policy;
   - 128 environments per policy, iteration 5,088: 83.4 million environment steps per policy.

   Each holds 128 policies: 8 configurations (learning rate, entropy 0.001, intrinsic weight) × 16 seeds.
2. **Warp rollout** (`scripts/roll_out_and_record.py`, on a card): the saved policies from the spawn in the
   environment they were trained in, fault guard on, no learning; 179,200 environment steps per policy (16
   environments) and 358,400 (128 environments). The states and actions of the first 2,100 steps of 512
   environments were recorded.
3. **Closed loop in C MuJoCo** (`scripts/c_mujoco_closed_loop.py`, processors, double precision): the same
   policies, 64 episodes each (1,024 per configuration), on the environment's own model (ant, U-maze walls, the
   ceiling on the wall tops). The episode rules are the environment's, without the guard (its bounds are only
   counted). Four physics settings:

   | setting | integrator | solver and line-search iterations | time step | physics steps per environment step |
   |---|---|---|---|---|
   | ours | implicitfast | 4, 8 | 0.005 s | 10 |
   | reference (MuJoCo's defaults, the reference ant model) | RK4 | 100, 50 | 0.01 s | 5 |
   | RK4 at our step and solver | RK4 | 4, 8 | 0.005 s | 10 |
   | implicit at our step and solver | implicit | 4, 8 | 0.005 s | 10 |

4. **Replays from identical states** (`scripts/c_mujoco_from_warp_records.py`): the recorded Warp action sequences
   applied open loop in C MuJoCo from the same start states, and the contacts before every sudden speed change.

## Result 1: the ants hop, in our physics and in reference MuJoCo alike

Share of steps per configuration in the Warp records (first 2,100 steps of 512 environments): about 80% of
states have no contact at all, 13 to 17% touch the floor, 4% a wall, 0.8 to 2.0% the ceiling over the walls. The
median torso height is 0.81 to 1.06 m, and 33 to 50% of states are above 1 m with no contact. The ant hops and
bounces; it does not walk.

The same policies closed loop, the two best configurations of each environment count:

| environments per policy, configuration | setting | episodes that arrive | steps above 1 m | steps with no contact | largest speed, median over episodes (m/s) | largest speed (m/s) |
|---|---|---|---|---|---|---|
| 16; learning rate 1e-4, weight 0.0625 | ours | 0.099 | 0.55 | 0.81 | 10.0 | 18.7 |
| | reference | 0.119 | 0.56 | 0.79 | 10.1 | 14.3 |
| | RK4 at our step and solver | 0.110 | 0.56 | 0.79 | 10.2 | 13.8 |
| | implicit at our step and solver | 0.085 | 0.53 | 0.81 | 9.9 | 15.2 |
| 128; learning rate 1e-2, weight 16 | ours | 0.060 | 0.48 | 0.81 | 9.2 | 13.1 |
| | reference | 0.062 | 0.47 | 0.79 | 9.3 | 14.5 |
| | RK4 at our step and solver | 0.069 | 0.47 | 0.79 | 9.3 | 13.0 |
| | implicit at our step and solver | 0.054 | 0.44 | 0.81 | 9.0 | 12.5 |

Over all 16 configurations the arrival rate under the reference settings is within −0.017 and +0.021 of ours (it
is higher in 11 of 16); the share of steps above 1 m differs by −0.010 to +0.014, and the share without contact is
0.020 to 0.023 lower under the reference settings. The hopping, the
heights and speeds of 9 to 10 m/s within an episode are reproduced by reference MuJoCo. **The motion is a
property of the ant model and the task, not of our physics,** and the return the policies earn does not depend
on something only our physics allows: under the reference settings they arrive as often or slightly more often.

Sudden speed changes (more than 4 m/s within one environment step, about 3,900 per million recorded steps) begin
on the floor in 49%, against a wall in 20%, in the air in 34% and at the ceiling in 0.4% of cases; the median
speed goes from 2.1 to 7.2 m/s. States above 10 m/s are 0.035% of recorded states; 68% of them had no contact one
step before and 27% touched a wall or the ceiling one or two steps before.

## Result 2: the late guard respawns come from our physics settings, in a few configurations

Fault-guard respawns in the Warp rollout per configuration: with 16 environments per policy 0 or 1 in each; with
128 environments per policy 24, 114, 14 and 77 in the four configurations with learning rate 1e-2 and none in
the four with 3e-2. In C MuJoCo, the configuration with 128 environments, learning rate 1e-2 and weight 64:

| setting | episodes beyond the guard's bounds (of 1,024) | MuJoCo's own "unstable" resets | largest speed (m/s) |
|---|---|---|---|
| ours | 9 | 5 | 2.45 × 10^7 |
| reference | 0 | 0 | 14.3 |
| RK4 at our step and solver | 0 | 0 | 13.5 |
| implicit at our step and solver | 0 | 0 | 12.7 |

With weight 1 the same pattern is weaker (ours: 4 episodes beyond the bounds, largest speed 33.5 m/s; the other
three settings 0). So the runaways the guard catches are produced by the implicitfast integrator at 4 solver
iterations, in C MuJoCo as in Warp; the two other integrators at the same step and solver do not produce them.
They are rare (at most 9 of 1,024 episodes) and confined to some configurations; the guard puts the ant back at
the start, which costs that episode its return.

## Result 3: two simulators of a hopping ant part within a few steps

Open loop, from identical states, the first environment step at which the torso positions are more than 0.1 m
apart (quantiles 10%, 50%, 90% over 256 replayed episodes; 16 environments per policy, the 128 case is the same):

| pair | 10% | 50% | 90% |
|---|---|---|---|
| Warp against C MuJoCo with our settings | 6 | 8 | 21 |
| C MuJoCo with our settings against reference settings | 6 | 8 | 10 |
| Warp against C MuJoCo with reference settings | 5 | 7 | 10 |
| C MuJoCo with our settings against its copy moved by one micrometre | 36 | 55 | 73 |
| C MuJoCo with our settings against its copy rounded to single precision | 50 | 73 | 94 |

Warp parts from C MuJoCo at the same settings sooner than rounding would explain, but as soon as C MuJoCo with
our settings parts from C MuJoCo with the reference settings. Paths of a hopping ant are not reproducible across
simulators after about 8 environment steps; the statistics of Result 1 are.

## Energy check

Define E(t) as the kinetic plus potential energy before environment step t and W(t) as the motors' work during the
step. With no contact, exact physics gives E(t+1) − E(t) − W(t) ≤ 0. The sum of its positive values over the
steps without contact is 140 to 290 J per episode in all four settings, the reference included, against about
200,000 J of motor work per episode (absolute). It is the same in every setting, so I read it as the error of the
work estimate (torque times the mean of the joint speeds before and after a physics step), not as energy one
setting creates.

## What this means for the study

1. The caveat of 07:08 PT ("ant results provisional for every arm") is withdrawn for the motion: reference MuJoCo
   allows it, and the arrival rates do not depend on our physics.
2. Two things remain, for every method alike:
   - the ant model with this task hops rather than walks; the reference ant model (gear 150, Gymnasium's) is the
     same, so this is the task the plan defined;
   - the implicitfast integrator at 4 solver iterations produces rare runaways that the guard catches, mostly in
     the configurations with 128 environments per policy and learning rate 1e-2. Changing it (for example to the
     implicit integrator, which C MuJoCo shows does not run away at the same step) changes the environment for
     every method and would need the speed and gate checks again; that is a decision for the user, not taken
     here.

## Not known

1. Whether MuJoCo Warp offers the implicit or RK4 integrator at a usable speed.
2. Question 3 in full: the arrival rate of the same policies with episodes above a plausible speed or height not
   counted was not computed. The agreement of Result 1 across settings makes a large dependence unlikely.
3. Whether policies trained in reference physics would hop as much (only policies trained in our physics were run).

## Files

- `scripts/roll_out_and_record.py`, `scripts/card_job.sh`: the Warp rollout (jobs 6905236, 6905237 on cheetah04).
- `scripts/c_mujoco_closed_loop.py`, `scripts/processor_job.sh`: closed loop in C MuJoCo (jobs 6905414, 6905416,
  6905563, 6905565 on puma01; the later two are the final records).
- `scripts/c_mujoco_from_warp_records.py`: contacts, speed changes, open-loop replays (jobs 6905415, 6905417,
  6905564, 6905566).
- `scripts/summarize.py`: tables from the records.
- `records/`: the outputs named in the scripts; `logs/`: the jobs' logs; `checkpoint_copy/`: the two checkpoints
  and their configuration files.

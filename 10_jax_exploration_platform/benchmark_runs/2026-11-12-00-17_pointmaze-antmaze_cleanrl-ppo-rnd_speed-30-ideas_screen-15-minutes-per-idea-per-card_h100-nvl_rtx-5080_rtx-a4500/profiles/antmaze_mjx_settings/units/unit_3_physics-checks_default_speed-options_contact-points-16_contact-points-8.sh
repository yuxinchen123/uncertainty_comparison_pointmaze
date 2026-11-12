#!/bin/bash
# EXPECTED_MINUTES=13
# AntMaze MJX settings, unit 3: the physics checks (random torques, torques that drive into walls, no torques)
# of the default model, of the two speed options alone, and of contact points 16 and 8 on top of them.
U="$(dirname "$0")"
B="solver_loop=fixed_count,carried_state=next_step_inputs"
exec bash "$U/run_unit.sh" "unit 3 physics checks: default, speed options, contact points 16 and 8" \
    --part checks --batch 4096 --models default $B max_contact_points=16,$B max_contact_points=8,$B

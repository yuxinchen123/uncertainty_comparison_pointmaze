#!/bin/bash
# EXPECTED_MINUTES=13
# AntMaze MJX settings, unit 4: the physics checks of contact points 8 with geometry pairs 4, of 4 line-search
# iterations, and of 3 and 2 solver iterations, each on top of the two speed options.
U="$(dirname "$0")"
B="solver_loop=fixed_count,carried_state=next_step_inputs"
exec bash "$U/run_unit.sh" "unit 4 physics checks: geometry pairs 4, line search 4, solver iterations 3 and 2" \
    --part checks --batch 4096 --models max_contact_points=8,max_geom_pairs=4,$B ls_iterations=4,$B solver_iterations=3,$B solver_iterations=2,$B

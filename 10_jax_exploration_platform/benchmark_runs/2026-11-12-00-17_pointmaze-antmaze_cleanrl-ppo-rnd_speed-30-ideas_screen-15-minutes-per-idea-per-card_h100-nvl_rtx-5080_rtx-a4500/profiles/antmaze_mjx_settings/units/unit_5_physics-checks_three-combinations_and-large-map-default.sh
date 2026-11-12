#!/bin/bash
# EXPECTED_MINUTES=13
# AntMaze MJX settings, unit 5: the physics checks of the three combinations (contact points 16 or 8, geometry
# pairs 8 or 4, with 4 line-search iterations and 3 solver iterations) on the U-maze, then of the default model on
# the Large map.
U="$(dirname "$0")"
B="solver_loop=fixed_count,carried_state=next_step_inputs"
exec bash "$U/run_unit.sh" "unit 5 physics checks: three combinations, and the default model on the large map" \
    --part checks --batch 4096 --models max_contact_points=16,ls_iterations=4,solver_iterations=3,$B max_contact_points=8,ls_iterations=4,solver_iterations=3,$B max_contact_points=8,max_geom_pairs=4,ls_iterations=4,solver_iterations=3,$B -- \
    --part checks --batch 4096 --map large --models default

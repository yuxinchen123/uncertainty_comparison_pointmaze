#!/bin/bash
# EXPECTED_MINUTES=13
# AntMaze MJX settings, unit 2: every setting alone and the combinations, on top of the two speed options,
# against the default model at 4,096 environments; then three combinations at 8,192 and 16,384 environments.
U="$(dirname "$0")"
B="solver_loop=fixed_count,carried_state=next_step_inputs"
ALONE="default $B max_contact_points=16,$B max_contact_points=8,$B max_contact_points=8,max_geom_pairs=4,$B ls_iterations=4,$B solver_iterations=3,$B solver_iterations=2,$B"
C16="max_contact_points=16,ls_iterations=4,solver_iterations=3,$B"
C16P4="max_contact_points=16,max_geom_pairs=4,ls_iterations=4,solver_iterations=3,$B"
C8="max_contact_points=8,ls_iterations=4,solver_iterations=3,$B"
C8P4="max_contact_points=8,max_geom_pairs=4,ls_iterations=4,solver_iterations=3,$B"
COMBINED="default $C16 $C16P4 $C8 $C8P4 max_contact_points=8,ls_iterations=4,$B max_contact_points=8,max_geom_pairs=4,ls_iterations=4,solver_iterations=2,$B"
exec bash "$U/run_unit.sh" "unit 2 settings alone and combined, speed" \
    --part speed --group "one setting at a time, on top of the two speed options" --batch 4096 --models $ALONE -- \
    --part speed --group "settings combined, on top of the two speed options" --batch 4096 --models $COMBINED -- \
    --part speed --group "settings combined, on top of the two speed options" --batch 8192 --models default $C16 $C8 $C8P4 -- \
    --part speed --group "settings combined, on top of the two speed options" --batch 16384 --models default $C16 $C8 $C8P4

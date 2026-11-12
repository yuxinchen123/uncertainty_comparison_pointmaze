#!/bin/bash
# EXPECTED_MINUTES=13
# AntMaze MJX settings, unit 1: the fixed-count solver loop (its passes written out, and as one scan) and the
# smaller carried state against the default model at 4,096, 1,024 and 8,192 environments (each batch size one
# cell, the five models timed in turn), then whether the options change the numbers on the card.
U="$(dirname "$0")"
FIVE="default solver_loop=fixed_count solver_loop=fixed_count_scan carried_state=next_step_inputs solver_loop=fixed_count,carried_state=next_step_inputs"
exec bash "$U/run_unit.sh" "unit 1 two speed options, speed at three batch sizes, equality" \
    --part speed --group "the two speed options" --batch 4096 --models $FIVE -- \
    --part speed --group "the two speed options" --batch 1024 --models $FIVE -- \
    --part speed --group "the two speed options" --batch 8192 --models $FIVE -- \
    --part equality --batch 1024

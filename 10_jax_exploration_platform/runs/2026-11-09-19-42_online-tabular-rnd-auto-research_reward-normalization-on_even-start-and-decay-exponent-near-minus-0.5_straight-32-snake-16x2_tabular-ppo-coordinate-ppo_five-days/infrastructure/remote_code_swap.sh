#!/bin/bash
# Run on a remote server inside its rnd_grid/code folder: make the previous copy's name and "current" point to the new copy.
set -e
cd "$1"
test -d source_9d038926840a128ac0141169ae85112722e1b693/10_jax_exploration_platform/src
if [ ! -L source_19581d3336a120df762c6251f16367ac20449a80 ]; then mv source_19581d3336a120df762c6251f16367ac20449a80 source_19581d3336a120df762c6251f16367ac20449a80.original; ln -s source_9d038926840a128ac0141169ae85112722e1b693 source_19581d3336a120df762c6251f16367ac20449a80; fi
ln -sfn source_9d038926840a128ac0141169ae85112722e1b693 current
ls -la | awk '{print $9, $10, $11}' | grep -E 'source_|current' | tr '\n' ';'
echo
find source_9d038926840a128ac0141169ae85112722e1b693 -name '*.py' | wc -l

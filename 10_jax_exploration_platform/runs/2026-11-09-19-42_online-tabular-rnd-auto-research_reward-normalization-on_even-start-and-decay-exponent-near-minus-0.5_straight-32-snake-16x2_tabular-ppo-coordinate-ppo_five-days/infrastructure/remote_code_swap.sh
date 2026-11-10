#!/bin/bash
# Run on a remote server inside its rnd_grid/code folder: make the previous copy's name and "current" point to the new copy.
set -e
cd "$1"
test -d source_84723a8069dcddc2db2354a1cf465729704f55a9/10_jax_exploration_platform/src
if [ ! -L source_9d038926840a128ac0141169ae85112722e1b693 ]; then mv source_9d038926840a128ac0141169ae85112722e1b693 source_9d038926840a128ac0141169ae85112722e1b693.original; ln -s source_84723a8069dcddc2db2354a1cf465729704f55a9 source_9d038926840a128ac0141169ae85112722e1b693; fi
ln -sfn source_84723a8069dcddc2db2354a1cf465729704f55a9 current
ls -la | awk '{print $9, $10, $11}' | grep -E 'source_|current' | tr '\n' ';'
echo
find source_84723a8069dcddc2db2354a1cf465729704f55a9 -name '*.py' | wc -l

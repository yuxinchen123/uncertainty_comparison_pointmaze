#!/bin/bash
# step_table.sh — distribution of the 600 run-2 runs by current training step.
# 12 rows: ten 100k-step bands (0-100K ... 900K-1M), then "not started" (configs still in queue/pending),
# then "finished" (one local JSON per completed run). Column 2 = (#runs in that band) / 600.
#
# How the running runs are bucketed: each slurm job is 8 workers on ONE node, so its 8 active runs train at
# ~the same pace; we read the latest "total_timesteps | N" from the job's log and count its 8 workers in
# that band. (Valid while a job's workers are all on their first run; good enough for an overview.)
RUN=/p/rlprojects/RND/07_reconstruction/train_runs/2026-06-24-21-24_run_2_after_reorganization
TOTAL=600

# finished = one JSON written per completed run; not started = configs never claimed
finished=$(find "$RUN/data/local" -name '*.json' 2>/dev/null | wc -l)
notstarted=$(ls "$RUN/queue/pending" 2>/dev/null | wc -l)

# zero the ten step bands
for i in $(seq 0 9); do buckets[$i]=0; done

# for each currently-RUNNING run-2 job, read its latest step and add its 8 workers to that band
while read -r jid; do
  [ -z "$jid" ] && continue
  L=$(ls -t "$RUN"/logs/*_"${jid}".log 2>/dev/null | head -1)
  [ -f "$L" ] || continue
  step=$(grep -oE 'total_timesteps \| [0-9]+' "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  [ -z "$step" ] && step=0
  b=$(( step / 100000 )); [ "$b" -gt 9 ] && b=9
  buckets[$b]=$(( ${buckets[$b]} + 8 ))
done < <(squeue -u sl5nw -h -t R -o "%i %j" 2>/dev/null | grep -E 'diff-prior|seq2graph|causal-rep|tab-bench|meta-icl' | awk '{print $1}')

# print the table
echo "Run-2 training-step distribution   ($(date '+%Y-%m-%d %H:%M %Z'))"
printf "%-14s | %s\n" "current step" "episodes at this stage / all runs"
echo "---------------+----------------------------------"
lab=(0-100K 100K-200K 200K-300K 300K-400K 400K-500K 500K-600K 600K-700K 700K-800K 800K-900K 900K-1M)
for i in $(seq 0 9); do printf "%-14s | %d/%d\n" "${lab[$i]}" "${buckets[$i]}" "$TOTAL"; done
printf "%-14s | %d/%d\n" "not started" "$notstarted" "$TOTAL"
printf "%-14s | %d/%d\n" "finished" "$finished" "$TOTAL"

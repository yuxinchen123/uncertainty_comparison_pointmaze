#!/bin/bash

#Define two variables with name job_id and cpu_core
job_id="scalar-ensemble-rnd-linear-ls/7p7u2o9e"

# for i in $(seq 1 1); do
#     sbatch slurm/01_run_gpu.slurm $job_id&
# done


for i in $(seq 1 5); do
    sbatch slurm/01_run_gpu.slurm $job_id&
done

for i in $(seq 1 5); do
    sbatch slurm/02_run_gnolim.slurm $job_id&
done

for i in $(seq 1 5); do
    sbatch slurm/03_run_cpu.slurm $job_id&
done



#wait 180 seconds
sleep 60
squeue -u sl5nw > squeue.log


#read and print each line from squeue.log
while IFS= read -r line
do
#seperate the line by space
    IFS=' ' read -r -a array <<< "$line"
    #if the last element does not start with "(" print the line, count the number of such line
    if [[ ! ${array[-1]} =~ ^\( ]]; then
        count=$((count+1))
    fi
   
done < squeue.log

printf "running count: %d\n" $count

sleep 120

while IFS= read -r line
do
    IFS=' ' read -r -a array <<< "$line"
    if [[ ${array[-2]} =~ ^[0-9]+$ ]]; then
        if [ ${array[-2]} -gt 3 ]; then
            echo $line
            job_id=${array[0]}
            scancel $job_id
        fi
    fi

if [ $count -ge 50 ]; then
    scancel -u sl5nw -t PD
fi 

done < squeue.log

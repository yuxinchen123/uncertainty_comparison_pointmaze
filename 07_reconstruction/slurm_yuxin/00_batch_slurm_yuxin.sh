#!/bin/bash

#Define two variables with name job_id and cpu_core
job_id="catresearch/rnd_07_reconstruction/ctccd8lp"
job_name="rl"

# for i in $(seq 1 1); do
#     sbatch --job-name="$job_name" slurm_yuxin/01_run_gpu.slurm $job_id &
# done


for i in $(seq 1 15); do
    sbatch --job-name="$job_name" slurm_yuxin/01_run_gpu.slurm $job_id &
done

for i in $(seq 1 15); do
    sbatch --job-name="$job_name" slurm_yuxin/02_run_gnolim.slurm $job_id &
done

# # sleep 20


for i in $(seq 1 26); do
    sbatch --job-name="$job_name" slurm_yuxin/03_run_cpu.slurm $job_id &
done

for i in $(seq 1 26); do
    sbatch --job-name="$job_name" slurm_yuxin/03_run_nolim.slurm $job_id &
done


# # sleep 20

# for i in $(seq 1 4); do
#     sbatch --job-name="$job_name" slurm/04_run_reservation_gpu.slurm $job_id &
# done


# for i in $(seq 1 10); do
#     sbatch --job-name="$job_name" slurm/05_run_reservation_cpu.slurm $job_id &
# done

#wait 180 seconds
sleep 60
squeue -u yuxinchen > squeue.log


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
        if [ ${array[-2]} -gt 2 ]; then
            echo $line
            job_id=${array[0]}
            scancel $job_id
        fi
    fi

if [ $count -ge 64 ]; then
    scancel -u yuxinchen -t PD
fi 

done < squeue.log

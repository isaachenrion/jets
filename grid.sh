
S=" 10 --slurm --gpu 0 --lr .4,3,.3 --what who,when,where --slurm_array_job_id 4790119 --slurm_array_task_id=9 -a=[dm,phy,eye] --data_dir=/scratch/ih692/data"
commands="$(python _grid.py $S)"
IFS=$'\n'
commands=($commands)
for cmd in ${commands[@]}
do
    bash run.sh "$cmd"
done

ARGS=( "$@" )
counter=1
PYTHONARGS=()
for arg in ${ARGS[@]}
do
    if [[ $counter == 1 ]]
    then
        N_JOBS=$arg
    else
        PYTHONARGS="$PYTHONARGS $arg"
    fi
    counter+=1
done
read SRCDIR DATA_DIR MODELS_DIR GPU EXPT_TIME END_DIR < <(bash misc/paths.sh)
PYTHONARGS="$PYTHONARGS --data_dir $DATA_DIR --experiment_time $EXPT_TIME --models_dir $MODELS_DIR"
RES=$(sbatch --time=$EXPT_TIME:00:00 --parsable --array=1-$N_JOBS --gres=$GPU eval.s $PYTHONARGS)

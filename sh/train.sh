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

read _ DATA_DIR MODELS_DIR GPU EXPT_TIME END_DIR < <(bash $SRCDIR/sh/paths.sh)
PYTHONARGS="$PYTHONARGS --data_dir $DATA_DIR --experiment_time $EXPT_TIME --models_dir $MODELS_DIR --email $SRCDIR/email_addresses.txt"
RES=$(sbatch --time=$EXPT_TIME:00:00 --parsable --array=1-$N_JOBS --gres=$GPU $SRCDIR/sh/slurm/train.s $PYTHONARGS)
echo $RES
sbatch --dependency=afterok:${RES} $SRCDIR/sh/slurm/summarize_training.s $RES $MODELS_DIR $END_DIR

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
    #echo $arg
done
#echo args $ARGS
#N_JOBS="${ARGS[0]}"
#echo n_jobs $N_JOBS
#PYTHONARGS="${ARGS[1]}"
#echo $1

#echo $N_JOBS
#echo pa $PYTHONARGS


read SRCDIR DATA_DIR MODELS_DIR GPU EXPT_TIME END_DIR < <(bash $SRCDIR/sh/paths.sh)
PYTHONARGS="$PYTHONARGS --data_dir $DATA_DIR --experiment_time $EXPT_TIME --models_dir $MODELS_DIR"
RES=$(sbatch --time=$EXPT_TIME:00:00 --parsable --array=1-$N_JOBS --gres=$GPU $SRCDIR/sh/train.s $PYTHONARGS)
echo $RES
sbatch --dependency=afterok:${RES} $SRCDIR/sh/summarize_training.s $RES $MODELS_DIR $END_DIR

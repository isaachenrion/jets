ARGS=$@
N_JOBS=$1
PYTHONARGS="${ARGS:2}"
echo $PYTHONARGS


read SRCDIR DATA_DIR MODELS_DIR GPU EXPT_TIME END_DIR < <(bash misc/paths.sh)

PYTHONARGS="$PYTHONARGS --data_dir $DATA_DIR --experiment_time $EXPT_TIME --models_dir $MODELS_DIR"
#echo sbatch --time=$EXPT_TIME:00:00 --parsable --array=1-$N_JOBS --gres=$GPU train.s $PYTHONARGS
RES=$(sbatch --time=$EXPT_TIME:00:00 --parsable --array=1-$N_JOBS --gres=$GPU train.s $PYTHONARGS)
echo $RES


sbatch --dependency=afterok:${RES} summarize_training.s $RES $MODELS_DIR $END_DIR

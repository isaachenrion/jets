PYTHONARGS="${@:2}"


read SRCDIR DATA_DIR MODELS_DIR GPU EXPT_TIME < <(bash misc/paths.sh)

PYTHONARGS="$PYTHONARGS --data_dir $DATA_DIR --experiment_time $EXPT_TIME --models_dir $MODELS_DIR"
RES=$(sbatch --time=$EXPT_TIME:00:00 --parsable --array=1-$1 --gres=$GPU train.s $PYTHONARGS)
echo $RES


sbatch --dependency=afterok:${RES} summarize_training.s $RES $MODELS_DIR

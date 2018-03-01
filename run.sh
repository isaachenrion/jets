PYTHONARGS="${@:2}"


read SRCDIR DATADIR MODELDIR GPU EXPT_TIME < <(bash misc/paths.sh)

PYTHONARGS="$PYTHONARGS --data_dir $DATADIR --experiment_time $EXPT_TIME"
RES=$(sbatch --time=$EXPT_TIME:00:00 --parsable --array=1-$1 --gres=$GPU train.s $PYTHONARGS)
echo $RES


sbatch --dependency=afterok:${RES} summarize_training.s $RES

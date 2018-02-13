SLURMARGS="${@:2}"

read SRCDIR DATADIR GPU QOS < <(bash misc/paths.sh)

SLURMARGS="$SRCDIR $SLURMARGS --data_dir $DATADIR"
RES=$(sbatch --parsable --array=1-$1 --gres=$GPU train.s $SLURMARGS)
echo $RES
MODEL_RUNDIR=$(find models/running $RES)
MODEL_OUTDIR=$(find models/finished $RES)
echo $MODEL_RUNDIR
echo $MODEL_OUTDIR

sbatch --dependency=afterok:${RES} summarize_training.s $MODEL_OUTDIR

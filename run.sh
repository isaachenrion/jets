SLURMARGS="${@:2}"

read SRCDIR DATADIR GPU QOS < <(bash misc/paths.sh)

SLURMARGS="$SRCDIR $SLURMARGS --data_dir $DATADIR"
RES=$(sbatch --parsable --array=1-$1 --gres=$GPU train.s $SLURMARGS)
echo $RES
sbatch --dependency=afterok:${RES} summarize_training.s ${RES}

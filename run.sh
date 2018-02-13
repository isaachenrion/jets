#SUBMISSION_OUT=$(sbatch --array=1-$1 train-hpc.s $SLURMARGS)
#echo $SUBMISSION_OUT
SLURMARGS="${@:2}"

read SRCDIR DATADIR GPU QOS < <(bash misc/paths.sh)

SLURMARGS="$SRCDIR $SLURMARGS --data_dir $DATADIR"
RES=$(sbatch --parsable --array=1-$1 --gres=$GPU --qos=$QOS train.s $SLURMARGS)
sbatch --dependency=afterok:${RES} summarize_training.s ${RES}

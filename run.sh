SLURMARGS="${@:2}"
#SUBMISSION_OUT=$(sbatch --array=1-$1 train-hpc.s $SLURMARGS)
#echo $SUBMISSION_OUT
RES=$(sbatch --parsable --array=1-$1 train-hpc.s $SLURMARGS)
sbatch --dependency=afterok:${RES} summarize_training-hpc.s ${RES}

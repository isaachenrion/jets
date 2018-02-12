SLURMARGS="${@:2}"
sbatch --array=$1 train-hpc.s $SLURMARGS | SUBMISSION_OUT
echo $SUBMISSION_OUT

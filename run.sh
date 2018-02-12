SLURMARGS="${@:2}"
SUBMISSION_OUT = $(sbatch --array=1-$1 train-hpc.s $SLURMARGS)
echo $SUBMISSION_OUT

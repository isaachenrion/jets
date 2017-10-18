#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=6GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=END
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
##SBATCH --array=1-10

module purge
SRCDIR=$HOME/jets
DATA_DIR=$SCRATCH/data
cd $SRCDIR
source activate jets
./slurm_run.sh -d $DATA_DIR -m 5 -n 3

##

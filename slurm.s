#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=6GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=END
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
##SBATCH --array=1-10

module purge
SRCDIR=$HOME/jets
cd $SRCDIR
source jets activate
##module load python3/intel/3.6.3
##module load pytorch/0.2.0_1
##module load scikit-learn/intel/0.18.1
chmod +x ./slurm_run.sh
./slurm_run.sh -d $SCRATCH -m 5 -n 1

##93YFYhK7DfH8

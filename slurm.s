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

module purge
##SRCDIR='/misc/kcgscratch1/ChoGroup/isaac/jets'
##RUNDIR=$SRCDIR
jets
./many_runs.sh -m 5 -n 1

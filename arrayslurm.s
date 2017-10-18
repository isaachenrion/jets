#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=END
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
##SBATCH --gres=gpu:1

module purge
SRCDIR=$HOME/jets
DATA_DIR=$SCRATCH/data/w-vs-qcd/pickles
cd $SRCDIR
source activate jets
model_type=5
COUNTER=$SLURM_ARRAY_TASK_ID
let 'SEED = COUNTER * 10000'
python train.py --data_dir $DATA_DIR -m $model_type --seed $SEED -g 1 &
disown %1
sleep 5

##./slurm_run.sh -d $DATA_DIR -m 5 -n 3
#########
##

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=END
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

SRCDIR=$HOME/jets
cd $SRCDIR
source activate jets

## variables
DATA_DIR=$SCRATCH/data/w-vs-qcd/pickles
MODEL_TYPE=5
EPOCHS=2
N=1000
let 'SEED = SLURM_ARRAY_TASK_ID * 10000'

##qprintf 'python train.py --data_dir %s -m %s --seed %s -v -g 1 &\n' $DATA_DIR $model_type $SEED
##python train.py --data_dir $DATA_DIR -m $model_type --seed $SEED -g 1 &
python train.py -v -m $MODEL_TYPE --data_dir $DATA_DIR  -e $EPOCHS -n $N &
disown %1
sleep 5

##./slurm_run.sh -d $DATA_DIR -m 5 -n 3
#########
##

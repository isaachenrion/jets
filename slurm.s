#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=END
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
module purge

SRCDIR=$HOME/jets
cd $SRCDIR

source activate jets

DATA_DIR=$SCRATCH/data/w-vs-qcd/pickles
MODEL_TYPE=7
EPOCHS=100
N=-1
ITERS=5
GPU=0
BATCH_SIZE=100
STEP_SIZE=0.001
let 'SEED=SLURM_ARRAY_TASK_ID * 1000'
printf 'CUDA VISIBLE DEVICES : %s\n' $CUDA_VISIBLE_DEVICES
printf 'N = %s, Model = %s, DATA_DIR = %s\n' $N $MODEL_TYPE $DATA_DIR
python train.py --iters $ITERS --step_size $STEP_SIZE -b $BATCH_SIZE -v -m $MODEL_TYPE --data_dir $DATA_DIR  -e $EPOCHS -n $N -g $GPU --seed $SEED

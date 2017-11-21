#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
##SBATCH --gres=gpu:1
#SBATCH --array=1-8
module purge

SRCDIR=$HOME/jets
cd $SRCDIR

source activate jets

DATA_DIR=$SCRATCH/data/w-vs-qcd/pickles
MODEL_TYPE=4
EPOCHS=50
N=-1
ITERS=1
GPU=0
BATCH_SIZE=100
STEP_SIZE=0.001
EXTRA_TAG=$SLURM_ARRAY_TASK_ID
##MODEL_DIR=$SRCDIR/models/MPNN/set/Oct-26/
##MODEL_DIR=$MODEL_DIR$SLURM_ARRAY_TASK_ID
##let 'SEED=SLURM_ARRAY_TASK_ID * 1000'
printf 'CUDA VISIBLE DEVICES : %s\n' $CUDA_VISIBLE_DEVICES
printf 'N = %s, Model = %s, DATA_DIR = %s\n' $N $MODEL_TYPE $DATA_DIR
sleep 5
python train.py --extra_tag $EXTRA_TAG --iters $ITERS --step_size $STEP_SIZE -b $BATCH_SIZE -s -m $MODEL_TYPE --data_dir $DATA_DIR  -e $EPOCHS -n $N -g $GPU
##python train.py --extra_tag $EXTRA_TAG -b $BATCH_SIZE -s -l $MODEL_DIR -r --data_dir $DATA_DIR  -m $MODEL_TYPE -e $EPOCHS -n $N -g $GPU
##python evaluation.py -m recnn/simple

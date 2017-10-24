#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=END
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --gres=gpu:1

module purge

SRCDIR=$HOME/jets
cd $SRCDIR

source activate jets

DATA_DIR=$SCRATCH/data/w-vs-qcd/pickles
MODEL_TYPE=7
EPOCHS=25
N=1000
N_RUNS=1
GPU=0
COUNTER=0
BATCH_SIZE=10

printf 'CUDA VISIBLE DEVICES : %s\n' $CUDA_VISIBLE_DEVICES
nvidia-smi
printf 'N = %s, Model = %s, DATA_DIR = %s\n' $N $MODEL_TYPE $DATA_DIR
while [  $COUNTER -lt $N_RUNS ];
do
  let 'SEED = COUNTER * 10000'
  python train.py -b $BATCH_SIZE -v -m $MODEL_TYPE --data_dir $DATA_DIR  -e $EPOCHS -n $N -g $GPU &
  ##python dummy.py
  disown %1
  sleep 5
  let COUNTER=COUNTER+1
done

##./slurm_run.sh -d $DATA_DIR -m 5 -n 3
#########
##

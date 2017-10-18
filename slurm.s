#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=12GB
#SBATCH --job-name=jets-experiment
#SBATCH --mail-type=END
#SBATCH --mail-user=henrion@nyu.edu
#SBATCH --output=slurm_%j.out
##SBATCH --array=1-10

module purge
SRCDIR=$HOME/jets
DATA_DIR=$SCRATCH/data/w-vs-qcd/pickles
cd $SRCDIR
source activate jets
model_type=5
N=3
COUNTER=0
printf 'N = %s, Model = %s, DATA_DIR = %s\n' $N $model_type $DATA_DIR
while [  $COUNTER -lt $N ];
do
  let 'SEED = COUNTER * 10000'
  printf 'Running with seed = %s\n' $SEED
  python train.py --debug --data_dir $DATA_DIR  -v -m $model_type --seed $SEED &
  disown %1
  sleep 3
  let COUNTER=COUNTER+1
done

##./slurm_run.sh -d $DATA_DIR -m 5 -n 3
#########
##

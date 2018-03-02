#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name
#SBATCH --job-name=Summarize-Training%j
#################
#a file for job output, you can check job progress
#SBATCH --output=slurm_out/Summarize-Training%j.out
#################
# a file for errors from the job
#SBATCH --error=slurm_out/Summarize-Training%j.err
#################
#time you think you need; default is one hour
#in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the faster your job will run.
# Default is one hour, this example will run in  less that 5 minutes.
#SBATCH --time=1:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 8 (or how ever many are on the node/card)
# We are submitting to the gpu partition, if you can submit to the hns partition, change this to -p hns_gpu.

#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=4000
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=henrion@nyu.edu

RES=$1
MODELS_DIR=$2
MODEL_RUNDIR=$(find $MODELS_DIR/running -depth -name $RES)

COMMAND_FILE="$MODEL_RUNDIR/command.txt"
NEW_COMMAND_FILE=$(echo $COMMAND_FILE | sed -e 's/\//-/g')
mv $COMMAND_FILE $NEW_COMMAND_FILE

rm -rf $MODEL_RUNDIR

MODEL_OUTDIR=$(find $MODELS_DIR -depth -name $RES)
mv $NEW_COMMAND_FILE $MODEL_OUTDIR
echo $MODEL_RUNDIR
echo $MODEL_OUTDIR

PYTHONARGS="-j $MODEL_OUTDIR -e"

read SRCDIR _DATADIR _GPU _QOS < <(bash misc/paths.sh)

cd $SRCDIR
source activate jets

python $SRCDIR/summary.py $PYTHONARGS

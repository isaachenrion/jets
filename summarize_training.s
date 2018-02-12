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
#SBATCH --qos=batch
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

SLURMARGS="$@"
PYTHONARGS="-j $SLURMARGS"

HOME='/misc/kcgscratch1/ChoGroup/isaac'
SRCDIR=$HOME/jets

cd $SRCDIR
source activate jets

##python -m visdom.server &
python $SRCDIR/summary.py $PYTHONARGS

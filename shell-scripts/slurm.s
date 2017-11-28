
#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name
#SBATCH --job-name=JetsExpt
#################
#a file for job output, you can check job progress
#SBATCH --output=JetsExpt.out
#################
# a file for errors from the job
#SBATCH --error=JetsExpt.err
#################
#time you think you need; default is one hour
#in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the faster your job will run.
# Default is one hour, this example will run in  less that 5 minutes.
#SBATCH --time=24:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 8 (or how ever many are on the node/card)
#SBATCH --gres gpu:1
# We are submitting to the gpu partition, if you can submit to the hns partition, change this to -p hns_gpu.
#SBATCH --qos=batch
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=12000
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=henrion@nyu.edu

SRCDIR=$HOME/jets
cd $SRCDIR

source activate jets

DATA_DIR=$SCRATCH/data/w-vs-qcd/pickles
MODEL_TYPE=5
EPOCHS=50
N=-1
ITERS=2
GPU=0
BATCH_SIZE=100
STEP_SIZE=0.001
EXTRA_TAG=$SLURM_ARRAY_TASK_ID
##MODEL_DIR=$SRCDIR/models/MPNN/set/Oct-26/
##MODEL_DIR=$MODEL_DIR$SLURM_ARRAY_TASK_ID
##let 'SEED=SLURM_ARRAY_TASK_ID * 1000'
##printf 'CUDA VISIBLE DEVICES : %s\n' $CUDA_VISIBLE_DEVICES
##printf 'N = %s, Model = %s, DATA_DIR = %s\n' $N $MODEL_TYPE $DATA_DIR
sleep 5
python train.py --extra_tag $EXTRA_TAG --iters $ITERS --step_size $STEP_SIZE -b $BATCH_SIZE -s -m $MODEL_TYPE --data_dir $DATA_DIR  -e $EPOCHS -n $N -g $GPU
##python train.py --extra_tag $EXTRA_TAG -b $BATCH_SIZE -s -l $MODEL_DIR -r --data_dir $DATA_DIR  -m $MODEL_TYPE -e $EPOCHS -n $N -g $GPU
##python evaluation.py -m recnn/simple

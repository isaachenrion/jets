# Bash scripts for running jobs on slurm

## Instructions
Here we provide some utility scripts for running jobs on slurm.

We provide three scripts.

### train.sh
The first argument is an integer specifying the number of jobs you want to run, i.e. random seeds.
After that, it just takes the arguments you would pass to train.py

### test.sh 
The first argument is an integer specifying the number of jobs you want to run, i.e. random seeds.
After that, it just takes the arguments you would pass to eval.py


### grid.sh
The first argument is an integer specifying the number of jobs you want to run, i.e. random seeds.
After that, it just takes the arguments you would pass to train.py, but comma-separated values will spawn different jobs in a combinatorial fashion.

e.g. bash grid.sh 5 --lr 0.1, 0.01, 0.001 --batch_size 16,32,64 would create 5 random seeds * 3 learning rates * 3 batch sizes = 45 jobs.

This is great for hyperparameter sweeps.

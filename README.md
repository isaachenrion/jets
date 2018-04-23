# Graph neural nets for jet physics


## Instructions

### Requirements

- python 3
- pytorch 0.3.1
- scikit-learn
- GPUtil for GPU utilization logging
- matplotlib
- numpy 

### Email results
If you want email results, you must create a file called email_addresses.txt
in the misc directory. The file must contain:
First line: recipient email address
Second line: sender email address
Third line: sender password
(You are advised to create a dummy account to send results.)
If you want to turn email off, add the flag --no_email.

### Data

You need to unzip the tars and put the raw pickle files into data/w-vs-qcd/pickles/raw (make this directory).
The training script will look for data in  data/w-vs-qcd/pickles/preprocessed and if it doesn't find it, make it from the raw stuff.

### Usage

Classification of W vs QCD jets:

```
# Training
python train.py [argparse args]
# Test
python evaluation.py [argparse args]
```

### Directory structure
- sh contains bash and slurm scripts for submitting jobs onto a slurm cluster
- src contains the python source code

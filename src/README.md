# Python source code


## Overview
This library is written in PyTorch.

## Directory Structure
- admin contains useful experiment administrator code for logging, emailing, saving, organizing
- architectures contains all the basic neural network components used for building models
- data_ops contains basic preprocessing code like padding, wrapping etc.
- experiment contains some wrapper code, providing access to the proteins and jets scripts
- jets contains code for training and evaluating jet models
- misc contains constants and email addresses, and some deprecated stuff
- monitors contains code for monitoring and measuring experiments: losses, gradient norms, accuracy, time etc.
- proteins contains code for training and evaluating protein models
- scripts contains the main python scripts to run - these take command line arguments
- utils contains the shadow apparatus for running experiments
- visualizing contains visualization code for creating graphs, histograms etc.

## How to add your own problem
- create a directory for your problem "/problem"
- you can add your own preprocessing logic in a folder called problem/data_ops
- any models you build should go in problem/models

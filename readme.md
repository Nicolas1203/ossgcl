Contrastive Learning for Online Semi-Supervised General Continual Learning
==========================================================================

This repository contains the code of our method presented at ICIP'22.

# File structure

This repository contains the following files:

```bash
    |-- Dockerfile
    |-- commands.sh     # bash file containing commands to run for reproducing experiments
    |-- config          
    |   -- parser.py    # Argument parser 
    |-- logs            # log files        
    |-- main.py
    |-- readme.md       # This file
    |-- results         # Folder containing experiments results
    |-- requirements.txt
    |-- src             
        |-- buffers         # Memory strategies
        |   |-- buffer.py               
        |   |-- reservoir.py
        |-- datasets            
        |   |-- __init__.py
        |   |-- cifar10.py
        |   |-- cifar100.py
        |   |-- split_cifar10.py
        |   |-- split_cifar100.py
        |   |-- split_tiny.py
        |   |-- tinyImageNet.py
        |-- learners        # Methods compared in the publications
        |   |-- base.py
        |   |-- baselines
        |   |   |-- er.py
        |   |   |-- scr.py
        |   |-- ce.py
        |   |-- sscr.py
        |   |-- supcon.py
        |-- models          # Model used
        |   |-- resnet.py
        |-- utils
            |-- data.py
            |-- losses.py
            |-- metrics.py
            |-- name_match.py   # Matching parser arguments and classes
            |-- tensorboard.py
            |-- utils.py
```

# Installation
To run the code, you have to install the necessary dependencies. It is advise to use either a virtual environment or docker

## Using pip
```
pip install torch==1.9.0
pip install -r requirements.txt
```

## Using docker
```
docker build -t ossgcl .
docker run -it --rm --name [CONTAINER_NAME] -v $PWD:/workspace -v [LOCAL_DATA_FOLDER]:/data --gpus all --shm-size=8G ossgcl
```

# Training
## Usage
Command line usage of the current repository with available parameters.

```
usage: main.py [-h] [--epochs EPOCHS] [--start-epoch START_EPOCH]
               [-b BATCH_SIZE] [--learning-rate LEARNING_RATE]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--optim {Adam,SGD}] [--seed SEED] [--memory-only]
               [--resume RESUME] [--tag TAG] [--tb-root TB_ROOT]
               [--print-freq PRINT_FREQ] [--verbose] [--logs LOGS]
               [--logs-root LOGS_ROOT] [--results-root RESULTS_ROOT]
               [--ckpt-root CKPT_ROOT] [--data-root-dir DATA_ROOT_DIR]
               [--min-crop MIN_CROP] [--dataset {cifar10,cifar100,tiny}]
               [--training-type {uni,inc}] [--n-classes N_CLASSES]
               [--img-size IMG_SIZE] [--num-workers NUM_WORKERS]
               [--n-tasks N_TASKS]
               [--labels-order LABELS_ORDER [LABELS_ORDER ...]]
               [--nb-channels NB_CHANNELS] [--temperature TEMPERATURE]
               [--alpha ALPHA] [--mem-size MEM_SIZE]
               [--mem-batch-size MEM_BATCH_SIZE] [--buffer {reservoir}]
               [--drop-method {random}] [--learner {ER,CE,SCR,SSCL,SC}]
               [--eval-mem] [--eval-random] [--n-runs N_RUNS]
               [--run-id RUN_ID] [--lab-pc LAB_PC] [--mem-iters MEM_ITERS]
               [--review] [--rv-iters RV_ITERS]

Pytorch Continual learning repo.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of total epochs to run
  --start-epoch START_EPOCH
                        manual epoch number (useful on restarts)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for stream data (default: 10)
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Initial learning rate
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay (default: 0)
  --optim {Adam,SGD}
  --seed SEED           Random seed to use.
  --memory-only, -mo    Training using only the memory ?
  --resume RESUME, -r RESUME
                        Resume old training. Load model given as resume
                        parameter
  --tag TAG, -t TAG     Base name for graphs and checkpoints
  --tb-root TB_ROOT     Where do you want tensorboards graphs ?
  --print-freq PRINT_FREQ, -p PRINT_FREQ
                        Number of batches between each print (default: 200)
  --verbose, -v         Verbose logs.
  --logs LOGS, -l LOGS  Name of the log file
  --logs-root LOGS_ROOT
                        Defalt root folder for writing logs.
  --results-root RESULTS_ROOT
                        Where you want to save the results ?
  --ckpt-root CKPT_ROOT
                        Directory where to save the model.
  --data-root-dir DATA_ROOT_DIR
                        Root dir containing the dataset to train on.
  --min-crop MIN_CROP   Minimum size for cropping in data augmentation. range
                        (0-1)
  --dataset {cifar10,cifar100,tiny}, -d {cifar10,cifar100,tiny}
                        Dataset to train on
  --training-type {uni,inc}
                        How to feed the data to the network (incremental
                        (continual learning) or uniform (offline))
  --n-classes N_CLASSES
                        Number of classes in database.
  --img-size IMG_SIZE   Size of the square input image
  --num-workers NUM_WORKERS, -w NUM_WORKERS
                        Number of workers to use for dataloader.
  --n-tasks N_TASKS     How many tasks do you want ?
  --labels-order LABELS_ORDER [LABELS_ORDER ...]
                        In which order to you want to see the labels ? Random
                        if not specified.
  --nb-channels NB_CHANNELS
                        Number of channels for the input image.
  --temperature TEMPERATURE
                        temperature parameter for softmax
  --alpha ALPHA         Weight for stream in SemiCon loss. Section 3.2 in paper.
  --mem-size MEM_SIZE   Memory size for continual learning
  --mem-batch-size MEM_BATCH_SIZE, -mbs MEM_BATCH_SIZE
                        How many images do you want to retrieve from the
                        memory/ltm
  --buffer {reservoir}  Type of buffer.
  --drop-method {random}
                        How to drop images from memory when adding new ones.
  --learner {ER,CE,SCR,SSCL,SC}
                        What learner do you want ?
  --eval-mem
  --eval-random
  --n-runs N_RUNS       Number of runs, with different seeds each time.
  --run-id RUN_ID       Id of the current run in multi run.
  --lab-pc LAB_PC       Number of labeled images per class when evaluating
                        outside of memory.
  --mem-iters MEM_ITERS
                        Number of times to make a grad update on memory at
                        each step
  --review, -rv         Review trick on last memory.
  --rv-iters RV_ITERS, -rvi RV_ITERS
                        Number of iteration to do on last memory after
                        training.
```
## commands.sh
Example of possible commands used to reproduce results are written in the ```commands.sh``` file.

# Visualization

## Tensorboard
While running experiment, the loss it plotted to tensorboard graphs. You can view them by running a tensorboard session with the corresponding logdir. This can be simply done using the following command.

    tensorboard --logdir=tb_runs


## Analyzing results
A sample code can be found in graph.ipynb to read load aggregated results in a dataframe.

## Docker commands

    docker run -it --rm -u $(id -u):$(id -g) --name ucl -v $PWD:/home/micheln -v /data/dataset/torchvision:/data --gpus all --shm-size=8G ucl

## Tensorboard docker

    tensorboard --logdir=runs --host 0.0.0.0

# Cite

```bibtex
@inproceedings{michel2022contrastive,
  title={Contrastive learning for online semi-supervised general continual learning},
  author={Michel, Nicolas and Negrel, Romain and Chierchia, Giovanni and Bercher, Jean-Fmn{\c{c}}ois},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1896--1900},
  year={2022},
  organization={IEEE}
}
```
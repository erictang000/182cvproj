#!/bin/bash
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH --job-name=train
#SBATCH --output=train-resnets.out
#SBATCH --error=train-resnets.err
#SBATCH -t 40
#SBATCH -c 20

srun train.sh

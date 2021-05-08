#!/bin/bash
#SBATCH -C gpu
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=2
#SBATCH --time 4:00:00
#SBATCH -J train.sh
#SBATCH --output=trainresnet.out
#SBATCH --error=trainresnet.err

# Setup software
module load pytorch/1.7.1-gpu

# Run the training
srun -l -u python train.py -o results -nl 60 

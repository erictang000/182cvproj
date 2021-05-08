#!/bin/bash
#SBATCH -C gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time 2:00:00
#SBATCH -J train.sh
#SBATCH --output=trainvit-startepoch9.out
#SBATCH --error=trainvit-startepoch9.err

# Setup software
module load pytorch/1.7.1-gpu

# Run the training
srun -l -u python train-checkpointed.py -o vit -nl 90 --start_epoch=10 --model="vit_base_patch16_224"

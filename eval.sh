#!/bin/bash
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time 2:00:00
#SBATCH -J eval.sh
#SBATCH --output=outputfiles/vit_sparse_pgd.out
#SBATCH --err=outputfiles/vit_sparse_pgd.err

# Setup software
module load pytorch/1.7.1-gpu

# Run the training
srun -l -u python eval.py -cr 1 -sa 16 --model="vit" --checkpoint="vit_sparse/epoch10"

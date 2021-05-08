#!/bin/sh
sbatch -p gpu_jsteinhardt -w shadowfax --gres=gpu:1 train.sh

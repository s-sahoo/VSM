#!/bin/bash
#SBATCH --mail-type=END                  # Request status by email
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 72:00:00                      # Time limit (hh:mm:ss)
#SBATCH --mem 164000                      # Memory
#SBATCH --partition=gpu                  # Partition
#SBATCH --constraint="gpu-high"          # GPU constraint
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH -N 1                             # Number of nodes
#SBATCH--output=watch_folder/%x-%j.log   # Output file name
#SBATCH --requeue                        # Requeue job if it fails

# Setup python path and env
source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate db
cd /share/kuleshov/chkao/VSM

# Run script
# shellcheck disable=SC2154

# model names
# python single_gpu_main.py --lr 5e-5 --batch_size 128 --timestep 'layerwise' --from_scratch 0
# python single_gpu_main.py --lr 5e-5 --batch_size 128 --timestep 'layerwise' --from_scratch 1

# python single_gpu_main_SSM.py --lr 5e-5 --batch_size 128 --timestep 'layerwise' --from_scratch false --mamba "mamba"
# python single_gpu_main_SSM.py --lr 5e-5 --batch_size 128 --timestep 'layerwise' --from_scratch false --mamba "mamba_BD"
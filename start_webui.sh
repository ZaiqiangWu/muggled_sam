#!/bin/bash
#SBATCH -p 032-partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -o ngf32_dp2ta.out

# 加载 conda
source ~/workspace/anaconda3/etc/profile.d/conda.sh
conda activate sam3

echo "Running on $(hostname)"
echo "Python: $(which python)"

nvidia-smi
python webui/server.py --https
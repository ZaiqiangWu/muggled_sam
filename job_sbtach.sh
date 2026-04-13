#!/bin/bash
#SBATCH -p 032-partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o ca0_hr_rnn_dp2ta.out

# 加载 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sam3

echo "Running on $(hostname)"
echo "Python: $(which python)"

nvidia-smi
python -u load_prompts_run_video.py --prompt_path ./pl0.pt --input_video ./videos/pl0/pl0_00.mp4
python -u load_prompts_run_video.py --prompt_path ./pl0.pt --input_video ./videos/pl0/pl0_01.mp4
python -u load_prompts_run_video.py --prompt_path ./pl0.pt --input_video ./videos/pl0/pl0_02.mp4
python -u load_prompts_run_video.py --prompt_path ./pl0.pt --input_video ./videos/pl0/pl0_03.mp4
python -u load_prompts_run_video.py --prompt_path ./pl0.pt --input_video ./videos/pl0/pl0_04.mp4




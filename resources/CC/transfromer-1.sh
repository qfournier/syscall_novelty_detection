#!/bin/bash
#SBATCH --account=def-aloise

# RESSOURCES
#SBATCH --cpus-per-task=40		# Number of CPUs
#SBATCH --mem=64000M			# Memory
#SBATCH --gres=gpu:v100:4		# Number of GPUs
#SBATCH --time=03-00:00		    # Brackets: 3h, 12h, 1d, 3d, 7d

# JOB SPECIFICATION
#SBATCH --job-name=transfromer-1
#SBATCH --output=/home/qfournie/anomaly_detection_private/logs/%x-%j

# LOAD VIRTUAL ENVIRONMENT
source /home/qfournie/anaconda3/etc/profile.d/conda.sh
conda activate py3

# TASK
cd /home/qfournie/anomaly_detection_private
python main.py --log_folder logs/transfromer-1 --data_path /lustre04/scratch/qfournie/data/large --train_folder "Train:train" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket,Valid OOD (SSL):valid_ood_ssl" --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --max_token 2048 --model transformer --n_hidden 672 --n_layer 2 --n_head 4 --dim_sys 48 --dim_proc 48 --dim_entry 12 --dim_ret 12 --dim_pid 12 --dim_tid 12 --dim_time 12 --dim_order 12 --activation "swiglu" --optimizer adam --n_update 1000000 --eval 1000 --lr 0.001 --warmup_steps 5000 --ls 0.1 --batch 16 --gpu "0,1,2,3" --chk --amp --reduce_lr_patience 5 --early_stopping_patience 20 --dropout 0.01 --analysis --seed 1
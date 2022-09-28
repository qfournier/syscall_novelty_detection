#!/bin/bash
#SBATCH --account=def-aloise

# RESSOURCES
#SBATCH --cpus-per-task=40		# Number of CPUs
#SBATCH --mem=64000M			# Memory
#SBATCH --time=03-00:00		    # Brackets: 3h, 12h, 1d, 3d, 7d

# JOB SPECIFICATION
#SBATCH --job-name=2-gram
#SBATCH --output=/home/qfournie/anomaly_detection_private/logs/%x-%j

# LOAD VIRTUAL ENVIRONMENT
source /home/qfournie/anaconda3/etc/profile.d/conda.sh
conda activate py3

# TASK
cd /home/qfournie/anomaly_detection_private
python main.py --log_folder logs/2-gram --data_path /lustre04/scratch/qfournie/data/large --train_folder "Train:train" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket,Valid OOD (SSL):valid_ood_ssl" --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --dataset_stat --model ngram --order 3 --analysis
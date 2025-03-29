#!/bin/bash
#SBATCH --job-name=msnn_train
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
# UNCOMMENT BELOW FOR FULL TEST, WITH SIZE OF dims BELOW
# #SBATCH --array=0-2

module load miniconda3/22.11.1
export LD_LIBRARY_PATH=/home/akavuturu/.conda/envs/msnn-env/lib/:$LD_LIBRARY_PATH
source /data/apps/extern/spack_on/gcc/9.3.0/miniconda3/22.11.1-7f5s6r5uqyngliaca4moeawkxnnsmwkq/etc/profile.d/conda.sh
conda activate msnn-env

# uncomment for full test
# dims=(1 5 10)
# dim=${dims[$SLURM_ARRAY_TASK_ID]}
dim=10

log_file="logs/MSNN_DIM_${dim}.log"

python MSNN_GPU.py --dim "$dim" --plot True 2>&1 | tee "$log_file"

# To run:
# sbatch msnn_train.slurm

# To check:
# squeue -u akavuturu
# tail -f logs/job_<job_id>_<taskid>.out
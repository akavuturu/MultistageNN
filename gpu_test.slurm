#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test.out
#SBATCH --error=gpu_test.err
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH -A sgoswam4_gpu

module load miniconda3/22.11.1
export LD_LIBRARY_PATH=/home/akavuturu/.conda/envs/msnn-env/lib/:$LD_LIBRARY_PATH
source /data/apps/extern/spack_on/gcc/9.3.0/miniconda3/22.11.1-7f5s6r5uqyngliaca4moeawkxnnsmwkq/etc/profile.d/conda.sh
conda activate msnn-env

python gpu_test.py
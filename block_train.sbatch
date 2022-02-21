#!/bin/bash
#SBATCH --job-name=sample_ddpgfd
#SBATCH --open-mode=append
#SBATCH --output=./log/%j_%x.out
#SBATCH --error=./log/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G

singularity exec --nv --bind $HOME/mujocopy-buildlock:/ext3/miniconda3/lib/python3.9/site-packages/mujoco_py/generated/mujocopy-buildlock --overlay /scratch/ax2030/tmp/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate dex_manip
cd /scratch/ax2030/dex_manip/DDPGfD/
python block_train.py config/block_s1.yaml
"

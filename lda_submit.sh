#!/bin/bash  
#SBATCH --ntasks-per-node=8 # core count
#SBATCH -o sb_submit.sh.log-%j
#SBATCH -a 

module load anaconda/2020a 

python -u ldawikipedia.py wiki10k wiki1k $SLURM_ARRAY_TASK_ID 16
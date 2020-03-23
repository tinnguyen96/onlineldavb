#!/bin/bash  
#SBATCH -N  10 # node count 
#SBATCH --ntasks-per-node=4 # core count

module load anaconda/2020a 

for ((s=1; s<11; s++)) do
    python -u sbwikipedia.py wiki10k wiki1k $s &
done
wait
#!/bin/bash  
#SBATCH -N  5 # node count 
#SBATCH --ntasks-per-node=8 # core count

module load anaconda/2020a 

for ((s=10; s<15; s++)) do
    python -u sbwikipedia.py wiki10k wiki1k $s 16 &
done
wait
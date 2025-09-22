#!/bin/bash
# Usage: bash perlmutter_interactive.sh
# Requests an interactive GPU node for 60 minutes
salloc -A <YOUR_ACCOUNT> -C gpu -q interactive -t 01:00:00 -N 1 -G 1
module purge
module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 ./cuda_hw --task 1 --M 4096 --N 4096 --K 4096 --TILE 16

#!/bin/bash
#SBATCH -J dgemm-scale
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1

# Bind OpenMP threads well on Perlmutter CPU
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun ./hw_driver --mode dgemm --m 2048 --n 2048 --k 2048 --scale

#!/bin/bash
#SBATCH -J attn-direct-loop
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o slurm_attn_direct_loop.out

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# export OMP_SCHEDULE=static

for t in 1 2 4 8 16 32; do
  export OMP_NUM_THREADS=$t
  echo "=== attn_direct | OMP_THREADS=$t ==="
  srun ./hw_driver --mode attn_direct --L 2048 --D 1024
done

#!/bin/bash
#SBATCH -J attn-via-loop
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o slurm_attn_via_loop.out

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# export OMP_SCHEDULE=static

echo "== Attention via DGEMM (naive) =="
for t in 1 2 4 8 16 32; do
  export OMP_NUM_THREADS=$t
  echo "=== attn_via | OMP_THREADS=$t ==="
  srun ./hw_driver --mode attn_via --L 2048 --D 1024
done

echo "== Attention via DGEMM (blocked) =="
for t in 1 2 4 8 16 32; do
  export OMP_NUM_THREADS=$t
  echo "=== attn_via (blocked) | OMP_THREADS=$t ==="
  srun ./hw_driver --mode attn_via --L 2048 --D 1024 --blocked --BM 128 --BN 128 --BK 64
done

#!/bin/bash
#SBATCH -J dgemm-loop
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o slurm_dgemm_loop_dynamic.out

module purge
module load PrgEnv-intel

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export MKL_THREADING_LAYER=INTEL
export MKL_NUM_THREADS=1
# Optionally try a schedule:
# export OMP_SCHEDULE=static
export OMP_SCHEDULE=dynamic,16
# export OMP_SCHEDULE=guided

for t in 1 2 4 8 16 32; do
  export OMP_NUM_THREADS=$t
  echo "=== DGEMM | OMP_THREADS=$t ==="
  srun ./hw_driver --mode dgemm --m 2048 --n 2048 --k 2048
done

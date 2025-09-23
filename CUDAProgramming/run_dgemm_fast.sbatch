#!/bin/bash
#SBATCH -A m5101_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -J dgemm_fast
#SBATCH -D /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming
#SBATCH -o /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/dgemm_fast.%j.out
#SBATCH -e /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/dgemm_fast.%j.err

set -euo pipefail
LOGDIR="/global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs"
BIN=./cuda_hw
M=10240; N=10240; K=10240

source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh
module purge; module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

echo "=== DGEMM cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 2 --M $M --N $N --K $K | tee -a "$LOGDIR/dgemm_cublas.csv"

echo "=== DGEMM Tiled sweep ==="
for TILE in 1 4 16 32; do
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 1 --M $M --N $N --K $K --TILE $TILE | tee -a "$LOGDIR/dgemm_tiled_T${TILE}.csv"
done

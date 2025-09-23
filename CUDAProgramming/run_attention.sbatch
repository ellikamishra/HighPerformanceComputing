#!/bin/bash
#SBATCH -A m5101_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -J attn_suite
#SBATCH -D /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming
#SBATCH -o /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/attn_suite.%j.out
#SBATCH -e /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/attn_suite.%j.err

set -euo pipefail
LOGDIR="/global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs"
BIN=./cuda_hw
L=10240; D=4096

source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh
module purge; module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

echo "=== Attention cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 5 --L $L --D $D | tee -a "$LOGDIR/attn_cublas.csv"

echo "=== Attention Tiled sweep ==="
for TILE in 1 4 16 32; do
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 4 --L $L --D $D --TILE $TILE | tee -a "$LOGDIR/attn_tiled_T${TILE}.csv"
done

echo "=== Attention Naive ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 3 --L $L --D $D | tee -a "$LOGDIR/attn_naive.csv"

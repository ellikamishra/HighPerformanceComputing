#!/bin/bash
#SBATCH -A m5101_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -J dgemm_naive_sweep
#SBATCH -D /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming
#SBATCH -o /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/dgemm_naive.%j.out
#SBATCH -e /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/dgemm_naive.%j.err

set -euo pipefail
LOGDIR="/global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs"
BIN=./cuda_hw
M=10240; N=10240; K=10240

source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh
module purge; module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

echo "=== DGEMM Naive (block sweep) ==="
for BX in 1 4 16; do
  for BY in 1 4 16; do
    echo "Naive BX=$BX BY=$BY"
    NBX=$BX NBY=$BY srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
      $BIN --task 0 --M $M --N $N --K $K | tee -a "$LOGDIR/dgemm_naive_B${BX}x${BY}.csv"
  done
done

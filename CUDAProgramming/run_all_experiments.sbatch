#!/bin/bash
#SBATCH -A m5101_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -J hw2_all
#SBATCH -o logs/hw2_all.%j.out
#SBATCH -e logs/hw2_all.%j.err

set -euo pipefail

source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh

module purge
module load cudatoolkit

mkdir -p logs
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

BIN=./cuda_hw

M=10240; N=10240; K=10240
echo "=== DGEMM Naive (block sweep) ==="
for BX in 1 4 16; do
  for BY in 1 4 16; do
    echo "Naive BX=$BX BY=$BY"
    NBX=$BX NBY=$BY srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 0 --M $M --N $N --K $K | tee -a logs/dgemm_naive_B${BX}x${BY}.csv
  done
done

# Problem 2: Tiled DGEMM — sweep tile sizes
echo "=== DGEMM Tiled ==="
for TILE in 1 4 16 32; do
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 1 --M $M --N $N --K $K --TILE $TILE | tee -a logs/dgemm_tiled_T${TILE}.csv
done

# cuBLAS DGEMM reference
echo "=== DGEMM cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 2 --M $M --N $N --K $K | tee -a logs/dgemm_cublas.csv

# Problem 3: Attention — sweep TILE (for tiled path) and record naive/tiled/cuBLAS
L=10240; D=4096
echo "=== Attention (naive/tiled/cuBLAS) ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 3 --L $L --D $D | tee -a logs/attn_naive.csv
for TILE in 1 4 16 32; do
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 4 --L $L --D $D --TILE $TILE | tee -a logs/attn_tiled_T${TILE}.csv
done
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 $BIN --task 5 --L $L --D $D | tee -a logs/attn_cublas.csv

echo "Done. Logs in logs/*.csv and job stdout/err."

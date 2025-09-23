#!/bin/bash
set -euo pipefail
LOGDIR="./logs"
BIN=./cuda_hw
M=10240; N=10240; K=10240

mkdir -p "$LOGDIR"
module purge
module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

echo "=== DGEMM cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 2 --M $M --N $N --K $K | tee -a "$LOGDIR/dgemm_cublas.csv"

echo "=== DGEMM Tiled sweep ==="
for TILE in 1 4 16 32; do
  echo "TILE=$TILE"
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
    $BIN --task 1 --M $M --N $N --K $K --TILE $TILE | tee -a "$LOGDIR/dgemm_tiled_T${TILE}.csv"
done

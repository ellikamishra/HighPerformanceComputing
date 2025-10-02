#!/bin/bash
set -euo pipefail
LOGDIR="./logs"
BIN=./cuda_hw
L=10240; D=4096

mkdir -p "$LOGDIR"
module purge
module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

echo "=== Attention cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 5 --L $L --D $D | tee -a "$LOGDIR/attn_cublas.csv"

echo "=== Attention Tiled sweep ==="
for TILE in 1 4 16 32; do
  echo "ATTN TILE=$TILE"
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
    $BIN --task 4 --L $L --D $D --TILE $TILE | tee -a "$LOGDIR/attn_tiled_T${TILE}.csv"
done

echo "=== Attention Naive ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 3 --L $L --D $D | tee -a "$LOGDIR/attn_naive.csv"

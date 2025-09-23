#!/bin/bash
set -euo pipefail
LOGDIR="./logs"
BIN=./cuda_hw
M=10240; N=10240; K=10240

mkdir -p "$LOGDIR"
module purge
module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

echo "=== DGEMM Naive (block sweep) ==="
for BX in 1 4 16; do
  for BY in 1 4 16; do
    echo "Naive BX=$BX BY=$BY"
    NBX=$BX NBY=$BY srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
      $BIN --task 0 --M $M --N $N --K $K | tee -a "$LOGDIR/dgemm_naive_B${BX}x${BY}.csv"
  done
done

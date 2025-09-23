#!/usr/bin/env bash
# One-shot launcher: allocates a GPU node (salloc) then runs the full HW2 sweep on it.

set -euo pipefail

# ---- Tunables (override via env: export ACCOUNT=m5101_g, TIME=03:00:00, etc.) ----
ACCOUNT="${ACCOUNT:-m5101_g}"     # your NERSC repo
QUEUE="${QUEUE:-regular}"         # regular|interactive (avoid debug for long runs)
TIME="${TIME:-03:00:00}"          # walltime for the allocation
NODES="${NODES:-1}"               # number of nodes
GPUS="${GPUS:-1}"                 # GPUs per job (we run 1 GPU per srun)

# Absolute working dir + logs (so paths are robust)
ROOT="$(pwd)"
LOGDIR="$ROOT/logs"

# ---- The work that will run *inside* the allocation ----
read -r -d '' INSIDE <<'EOS'
set -euo pipefail

module purge
module load cudatoolkit

mkdir -p logs
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

BIN=./cuda_hw
M=10240; N=10240; K=10240
L=10240; D=4096

echo "===== START $(date) on $(hostname) ====="
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi || true

# 1) DGEMM cuBLAS (fast reference)
echo "=== DGEMM cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 2 --M $M --N $N --K $K | tee -a logs/dgemm_cublas.csv

# 2) DGEMM Tiled sweep (TILE = 1,4,16,32)
echo "=== DGEMM Tiled sweep ==="
for TILE in 1 4 16 32; do
  echo "TILE=$TILE"
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
    $BIN --task 1 --M $M --N $N --K $K --TILE $TILE | tee -a logs/dgemm_tiled_T${TILE}.csv
done

# 3) Attention cuBLAS + Tiled sweep + Naive
echo "=== Attention cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 5 --L $L --D $D | tee -a logs/attn_cublas.csv

echo "=== Attention Tiled sweep ==="
for TILE in 1 4 16 32; do
  echo "ATTN TILE=$TILE"
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
    $BIN --task 4 --L $L --D $D --TILE $TILE | tee -a logs/attn_tiled_T${TILE}.csv
done

echo "=== Attention Naive ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 3 --L $L --D $D | tee -a logs/attn_naive.csv

# 4) DGEMM Naive block sweep (slowest; do last)
echo "=== DGEMM Naive (block sweep) ==="
for BX in 1 4 16; do
  for BY in 1 4 16; do
    echo "Naive BX=$BX BY=$BY"
    NBX=$BX NBY=$BY srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
      $BIN --task 0 --M $M --N $N --K $K | tee -a logs/dgemm_naive_B${BX}x${BY}.csv
  done
done

echo "===== END $(date) ====="
EOS

# ---- Kick off the allocation and run the inside payload on the allocated node ----
echo "Requesting allocation: -A $ACCOUNT -C gpu -q $QUEUE -t $TIME -N $NODES -G $GPUS"
# Use a login shell inside the allocation so modules are available
salloc -A "$ACCOUNT" -C gpu -q "$QUEUE" -t "$TIME" -N "$NODES" -G "$GPUS" \
  bash -lc "$INSIDE"

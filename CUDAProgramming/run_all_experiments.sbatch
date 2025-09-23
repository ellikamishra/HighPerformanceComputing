#!/bin/bash
#SBATCH -A m5101_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -J hw2_all
#SBATCH -D /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming
#SBATCH -o /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/hw2_all.%j.out
#SBATCH -e /global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs/hw2_all.%j.err

set -euo pipefail

# --- Paths & sizes ---
LOGDIR="/global/u2/e/ellikam/csce654-hw1/HighPerformanceComputing/CUDAProgramming/logs"
BIN=./cuda_hw
M=10240; N=10240; K=10240         # DGEMM sizes
L=10240; D=4096                   # Attention sizes

# Make sure CSV location exists (stdout/err dir must already exist at submit time)
mkdir -p "$LOGDIR"

# --- Modules & build ---
module purge
module load cudatoolkit
make -f Makefile_Perlmutter clean && make -f Makefile_Perlmutter

echo "===== START $(date) on $(hostname) ====="
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi || true

# ========== 1) DGEMM cuBLAS (fast reference) ==========
echo "=== DGEMM cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 2 --M $M --N $N --K $K \
  | tee -a "$LOGDIR/dgemm_cublas.csv"

# ========== 2) DGEMM Tiled sweep (TILE = 1,4,16,32) ==========
echo "=== DGEMM Tiled sweep ==="
for TILE in 1 4 16 32; do
  echo "TILE=$TILE"
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
    $BIN --task 1 --M $M --N $N --K $K --TILE $TILE \
    | tee -a "$LOGDIR/dgemm_tiled_T${TILE}.csv"
done

# ========== 3) Attention (cuBLAS + Tiled sweep) ==========
echo "=== Attention cuBLAS ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 5 --L $L --D $D \
  | tee -a "$LOGDIR/attn_cublas.csv"

echo "=== Attention Tiled sweep ==="
for TILE in 1 4 16 32; do
  echo "ATTN TILE=$TILE"
  srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
    $BIN --task 4 --L $L --D $D --TILE $TILE \
    | tee -a "$LOGDIR/attn_tiled_T${TILE}.csv"
done

# (Optional) Attention naive single run (kept before the long naive DGEMM)
echo "=== Attention Naive ==="
srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
  $BIN --task 3 --L $L --D $D \
  | tee -a "$LOGDIR/attn_naive.csv"

# ========== 4) DGEMM Naive block sweep (slowest; do LAST) ==========
echo "=== DGEMM Naive (block sweep) ==="
for BX in 1 4 16; do
  for BY in 1 4 16; do
    echo "Naive BX=$BX BY=$BY"
    NBX=$BX NBY=$BY srun -n 1 -c 32 --gpus=1 --gpu-bind=single:1 \
      $BIN --task 0 --M $M --N $N --K $K \
      | tee -a "$LOGDIR/dgemm_naive_B${BX}x${BY}.csv"
  done
done

echo "===== END $(date) ====="

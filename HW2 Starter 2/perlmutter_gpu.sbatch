#!/bin/bash
#SBATCH -A <your_project>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -J cuda_hw

module purge
module load cudatoolkit

make clean && make

## Change these
srun -n 1 -G 1 ./cuda_hw --task 0 --M 2048 --N 2048 --K 2048
srun -n 1 -G 1 ./cuda_hw --task 1 --M 2048 --N 2048 --K 2048 --TILE 32
srun -n 1 -G 1 ./cuda_hw --task 2 --M 2048 --N 2048 --K 2048

srun -n 1 -G 1 ./cuda_hw --task 3 --L 1024 --D 256
srun -n 1 -G 1 ./cuda_hw --task 4 --L 1024 --D 256
srun -n 1 -G 1 ./cuda_hw --task 5 --L 1024 --D 256

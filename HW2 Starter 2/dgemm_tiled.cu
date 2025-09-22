#include "common.cuh"

// I am using TILE as a C++ template. 
// You can use  __shared__ double As[TILE][TILE] to allocate memory in this case.

template<int TILE>
__global__ void dgemm_tiled_kernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double* __restrict__ C,
                                   int M, int N, int K)
{
   // write your CUDA code
}

// Host code
void dgemm_tiled_gpu(const double* dA, const double* dB, double* dC,
                     int M, int N, int K, int tile=32, cudaStream_t stream=0)
{
  // define grids and blocks and call the CUDA kernel
}

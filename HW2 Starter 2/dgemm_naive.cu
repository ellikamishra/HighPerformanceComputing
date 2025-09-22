#include "common.cuh"

// Device code
__global__ void dgemm_naive_kernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double* __restrict__ C,
                                   int M, int N, int K)
{
  // write your code here
}

// Host code
void dgemm_naive_gpu(const double* dA, const double* dB, double* dC,
                     int M, int N, int K, cudaStream_t stream=0)
{
   // create grids and blocks and launch the CUDA kernel
}

#include "common.cuh"
#include <cmath>

__global__ void transpose_LD_to_DL(const double* __restrict__ K,
                                   double* __restrict__ KT,
                                   int L, int D)
{
  // write your transpose code 
}

__global__ void softmax_rows(double* __restrict__ S, int L, double inv_sqrt_d)
{
`  // write your softmax code similar to HW1
}

void dgemm_naive_gpu(const double*, const double*, double*, int,int,int, cudaStream_t);
void dgemm_tiled_gpu(const double*, const double*, double*, int,int,int, int, cudaStream_t);

void attention_via_dgemm(const double* dQ, const double* dK, const double* dV,
                         double* dO, int L, int D, int mode,
                         cublasHandle_t h, cudaStream_t stream=0)
{
  // Write your code here
  // You can use mode to call different versions of DGEMM
  // For example 
  // if (mode == 0)       dgemm_naive_gpu(dQ, dKT, dS, L, L, D, stream);
  // else if (mode == 1)  dgemm_tiled_gpu(dQ, dKT, dS, L, L, D, 32, stream);
  // else                 cublas_rowmajor_dgemm(h, L, L, D, dQ, dKT, dS);

}

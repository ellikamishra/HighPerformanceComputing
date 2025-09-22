
#include "common.cuh"

// Each thread computes one C[row,col]
__global__ void dgemm_naive_kernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double* __restrict__ C,
                                   int M, int N, int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  double acc = 0.0;
  // Row-major: A[row, t] at row*K + t ; B[t, col] at t*N + col
  for (int t=0; t<K; ++t) {
    acc += A[(size_t)row*K + t] * B[(size_t)t*N + col];
  }
  C[(size_t)row*N + col] = acc;
}

// Host wrapper
void dgemm_naive_gpu(const double* dA, const double* dB, double* dC,
                     int M, int N, int K, cudaStream_t stream)
{
  int bx = 16, by = 16;
if (const char* s = std::getenv("NBX")) bx = std::max(1, atoi(s));
if (const char* s = std::getenv("NBY")) by = std::max(1, atoi(s));
dim3 block(bx,by);
  dim3 grid( (N + block.x - 1)/block.x, (M + block.y - 1)/block.y );
  dgemm_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

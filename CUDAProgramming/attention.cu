
#include "common.cuh"
#include <cmath>

// Transpose K (L x D row-major) into KT (D x L row-major)
template<int TILE=32>
__global__ void transpose_LD_to_DL(const double* __restrict__ K,
                                   double* __restrict__ KT,
                                   int L, int D)
{
  __shared__ double tile[TILE][TILE+1]; // avoid bank conflicts
  int x = blockIdx.x * TILE + threadIdx.x; // D-dim (cols in K)
  int y = blockIdx.y * TILE + threadIdx.y; // L-dim (rows in K)

  if (y < L && x < D) {
    tile[threadIdx.y][threadIdx.x] = K[(size_t)y*D + x];
  }
  __syncthreads();

  int xo = blockIdx.y * TILE + threadIdx.x; // becomes cols in KT
  int yo = blockIdx.x * TILE + threadIdx.y; // becomes rows in KT
  if (yo < D && xo < L) {
    KT[(size_t)yo*L + xo] = tile[threadIdx.x][threadIdx.y];
  }
}

// Row-wise numerically-stable softmax for S (L x L)
__global__ void softmax_rows(double* __restrict__ S, int L, double inv_sqrt_d)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= L) return;

  // 1) scale by inv_sqrt_d and find row max
  double m = -1e300;
  for (int j=0;j<L;++j){
    double s = S[(size_t)row*L + j] * inv_sqrt_d;
    m = (s > m) ? s : m;
  }
  // 2) sum exp
  double sum = 0.0;
  for (int j=0;j<L;++j){
    double s = S[(size_t)row*L + j] * inv_sqrt_d;
    sum += exp(s - m);
  }
  sum = (sum == 0.0) ? 1.0 : sum;
  // 3) write normalized
  for (int j=0;j<L;++j){
    double s = S[(size_t)row*L + j] * inv_sqrt_d;
    S[(size_t)row*L + j] = exp(s - m) / sum;
  }
}

void dgemm_naive_gpu(const double*, const double*, double*, int,int,int, cudaStream_t);
void dgemm_tiled_gpu(const double*, const double*, double*, int,int,int, int, cudaStream_t);

void attention_via_dgemm(const double* dQ, const double* dK, const double* dV,
                         double* dO, int L, int D, int mode,
                         cublasHandle_t h, cudaStream_t stream)
{
  // Allocate intermediates
  double *dKT=nullptr, *dS=nullptr;
  CUDA_CHECK(cudaMalloc(&dKT, (size_t)D*L*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dS,  (size_t)L*L*sizeof(double)));

  // 1) K^T
  dim3 tblock(32,32);
  dim3 tgrid( (D+31)/32, (L+31)/32 );
  transpose_LD_to_DL<32><<<tgrid, tblock, 0, stream>>>(dK, dKT, L, D);
  CUDA_CHECK(cudaGetLastError());

  // 2) S = Q * K^T  (L x L)
  if (mode == 0) {
    dgemm_naive_gpu(dQ, dKT, dS, L, L, D, stream);
  } else if (mode == 1) {
    dgemm_tiled_gpu(dQ, dKT, dS, L, L, D, 32, stream);
  } else {
    cublas_rowmajor_dgemm(h, L, L, D, dQ, dKT, dS, 1.0, 0.0);
  }

  // 3) A = softmax(S / sqrt(D)) row-wise
  double inv_sqrt_d = 1.0 / std::sqrt((double)D);
  int threads = 256;
  int blocks = (L + threads - 1) / threads;
  softmax_rows<<<blocks, threads, 0, stream>>>(dS, L, inv_sqrt_d);
  CUDA_CHECK(cudaGetLastError());

  // 4) O = A * V (L x D)
  if (mode == 0) {
    dgemm_naive_gpu(dS, dV, dO, L, D, L, stream);
  } else if (mode == 1) {
    dgemm_tiled_gpu(dS, dV, dO, L, D, L, 32, stream);
  } else {
    cublas_rowmajor_dgemm(h, L, D, L, dS, dV, dO, 1.0, 0.0);
  }

  CUDA_CHECK(cudaFree(dKT));
  CUDA_CHECK(cudaFree(dS));
}

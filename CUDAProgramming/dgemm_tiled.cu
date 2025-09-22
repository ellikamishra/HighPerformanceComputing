
#include "common.cuh"

template<int TILE>
__global__ void dgemm_tiled_kernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double* __restrict__ C,
                                   int M, int N, int K)
{
  __shared__ double As[TILE][TILE];
  __shared__ double Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  double acc = 0.0;

  // Number of tiles over K dimension
  int numTiles = (K + TILE - 1) / TILE;
  for (int t=0; t<numTiles; ++t) {
    int Acol = t*TILE + threadIdx.x;
    int Brow = t*TILE + threadIdx.y;

    // Load tiles with bounds checks
    As[threadIdx.y][threadIdx.x] = (row < M && Acol < K)
        ? A[(size_t)row*K + Acol] : 0.0;
    Bs[threadIdx.y][threadIdx.x] = (Brow < K && col < N)
        ? B[(size_t)Brow*N + col] : 0.0;

    __syncthreads();

    for (int k=0; k<TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[(size_t)row*N + col] = acc;
  }
}

// Host launcher selecting TILE at runtime
void dgemm_tiled_gpu(const double* dA, const double* dB, double* dC,
                     int M, int N, int K, int tile, cudaStream_t stream)
{
  // choose from supported tile sizes
  dim3 block, grid;
  auto launch = [&](auto kernel){
    block = dim3(tile, tile);
    grid = dim3( (N + tile - 1)/tile, (M + tile - 1)/tile );
    kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
  };

  switch (tile) {
    case 1:  launch(dgemm_tiled_kernel<1>);  break;
    case 4:  launch(dgemm_tiled_kernel<4>);  break;
    case 8:  launch(dgemm_tiled_kernel<8>);  break;
    case 16: launch(dgemm_tiled_kernel<16>); break;
    case 32: launch(dgemm_tiled_kernel<32>); break;
    default: // fallback to 16
      fprintf(stderr, "[WARN] Unsupported TILE=%d, using 16\n", tile);
      launch(dgemm_tiled_kernel<16>);
  }
}

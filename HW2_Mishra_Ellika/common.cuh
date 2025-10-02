
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(expr) do { \
  cudaError_t _e = (expr); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "[CUDA] %s:%d error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

#define CUBLAS_CHECK(expr) do { \
  cublasStatus_t _s = (expr); \
  if (_s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "[cuBLAS] %s:%d status=%d\n", __FILE__, __LINE__, (int)_s); \
    std::exit(1); \
  } \
} while(0)

inline float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms;
}

inline void cublas_rowmajor_dgemm(cublasHandle_t h,
                                  int m, int n, int k,
                                  const double* dA, const double* dB, double* dC,
                                  const double alpha=1.0, const double beta=0.0) {
  const double *Acol = dB; 
  const double *Bcol = dA;
  double *Ccol = dC;

  CUBLAS_CHECK(cublasDgemm(h,
      CUBLAS_OP_N, CUBLAS_OP_N,
      n, m, k,
      &alpha,
      Acol, n,
      Bcol, k,
      &beta,
      Ccol, n));
}

inline void fill_host(double* x, size_t n, unsigned long long seed=42ULL) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U(-0.5, 0.5);
  for (size_t i=0;i<n;++i) x[i] = U(rng);
}

inline double max_abs_diff_host(const double* a, const double* b, size_t n) {
  double m=0.0; for (size_t i=0;i<n;++i) m = std::max(m, std::abs(a[i]-b[i])); return m;
}

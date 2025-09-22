#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include "common.cuh"

void dgemm_naive_gpu (const double*, const double*, double*, int,int,int, cudaStream_t);
void dgemm_tiled_gpu (const double*, const double*, double*, int,int,int, int, cudaStream_t);
void attention_via_dgemm(const double*, const double*, const double*, double*, int,int, int, cublasHandle_t, cudaStream_t);

static int argi(char** a, int &i){ return std::atoi(a[++i]); }

int main(int argc, char** argv) {
  int M=1024, N=1024, K=1024;
  int L=512, D=256;
  int task=0;
  int TILE=32;

  for (int i=1;i<argc;i++) {
    if (!strcmp(argv[i],"--task") && i+1<argc) task = argi(argv, i);
    else if (!strcmp(argv[i],"--M") && i+1<argc) M = argi(argv, i);
    else if (!strcmp(argv[i],"--N") && i+1<argc) N = argi(argv, i);
    else if (!strcmp(argv[i],"--K") && i+1<argc) K = argi(argv, i);
    else if (!strcmp(argv[i],"--L") && i+1<argc) L = argi(argv, i);
    else if (!strcmp(argv[i],"--D") && i+1<argc) D = argi(argv, i);
    else if (!strcmp(argv[i],"--TILE") && i+1<argc) TILE = argi(argv, i);
  }

  CUDA_CHECK(cudaSetDevice(0));
  cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasSetStream(h, stream));

  std::vector<double> hA((size_t)M*K), hB((size_t)K*N), hC((size_t)M*N, 0.0), hCref((size_t)M*N,0.0);
  std::vector<double> hQ((size_t)L*D), hK((size_t)L*D), hV((size_t)L*D), hO((size_t)L*D), hOref((size_t)L*D);

  fill_host(hA.data(), hA.size(), 1); fill_host(hB.data(), hB.size(), 2);
  fill_host(hQ.data(), hQ.size(), 3); fill_host(hK.data(), hK.size(), 4); fill_host(hV.data(), hV.size(), 5);

  double *dA=nullptr, *dB=nullptr, *dC=nullptr;
  double *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr, *dCref=nullptr, *dOref=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dCref, hCref.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dQ, hQ.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dK, hK.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dV, hV.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dO, hO.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOref, hOref.size()*sizeof(double)));

  CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), hA.size()*sizeof(double), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(), hB.size()*sizeof(double), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dQ, hQ.data(), hQ.size()*sizeof(double), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dK, hK.data(), hK.size()*sizeof(double), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dV, hV.data(), hV.size()*sizeof(double), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(dC, 0, hC.size()*sizeof(double), stream));
  CUDA_CHECK(cudaMemsetAsync(dO, 0, hO.size()*sizeof(double), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));


 
  cublas_rowmajor_dgemm(h, M, N, K, dA, dB, dCref);
  attention_via_dgemm(dQ, dK, dV, dOref, L, D, 2, h, stream);

  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

  // Naive 
  if (task == 0) {
    CUDA_CHECK(cudaEventRecord(e0, stream));
    dgemm_naive_gpu(dA, dB, dC, M, N, K, stream);
    CUDA_CHECK(cudaEventRecord(e1, stream)); CUDA_CHECK(cudaEventSynchronize(e1));
    double ms = elapsed_ms(e0,e1);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size()*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCref.data(), dCref, hCref.size()*sizeof(double), cudaMemcpyDeviceToHost));
    double mad = max_abs_diff_host(hC.data(), hCref.data(), hC.size());
    double gflops = (2.0*(double)M*N*K) / (ms*1e6);
    printf("[DGEMM naive] time=%.3f ms rate=%.2f GF/s diff=%.3e\n", ms,gflops,mad);
  }
  else if (task == 1) { // blocked
    CUDA_CHECK(cudaEventRecord(e0, stream));
    dgemm_tiled_gpu(dA, dB, dC, M, N, K, TILE, stream);
    CUDA_CHECK(cudaEventRecord(e1, stream)); CUDA_CHECK(cudaEventSynchronize(e1));
    double ms = elapsed_ms(e0,e1);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size()*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCref.data(), dCref, hCref.size()*sizeof(double), cudaMemcpyDeviceToHost));
    double mad = max_abs_diff_host(hC.data(), hCref.data(), hC.size());
    double gflops = (2.0*(double)M*N*K) / (ms*1e6);
    printf("[DGEMM tiled] time=%.3f ms rate=%.2f GF/s diff=%.3e\n", ms,gflops,mad);
  }
  else if (task == 2) { // DGEMM cuBLAS
    CUDA_CHECK(cudaEventRecord(e0, stream));
    cublas_rowmajor_dgemm(h, M, N, K, dA, dB, dC);
    CUDA_CHECK(cudaEventRecord(e1, stream)); CUDA_CHECK(cudaEventSynchronize(e1));
    double ms = elapsed_ms(e0,e1);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size()*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCref.data(), dCref, hCref.size()*sizeof(double), cudaMemcpyDeviceToHost));
    double mad = max_abs_diff_host(hC.data(), hCref.data(), hC.size());
    double gflops = (2.0*(double)M*N*K) / (ms*1e6);
    printf("[DGEMM cuBLAS] time=%.3f ms rate=%.2f GF/s diff=%.3e\n", ms,gflops,mad);
  }
  else if (task >= 3 && task <= 5) { // attention via DGEMM with your naive DGEMM or tiled DGEMM or cuBLAS
    int mode = (task==3)?0: (task==4)?1:2;
    CUDA_CHECK(cudaEventRecord(e0, stream));
    attention_via_dgemm(dQ, dK, dV, dO, L, D, mode, h, stream);
    CUDA_CHECK(cudaEventRecord(e1, stream)); CUDA_CHECK(cudaEventSynchronize(e1));
    double ms = elapsed_ms(e0,e1);
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size()*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hOref.data(), dOref, hOref.size()*sizeof(double), cudaMemcpyDeviceToHost));
    double mad = max_abs_diff_host(hO.data(), hOref.data(), hO.size());
    double flops = 4.0 * (double)L*L*D;
    double gflops = flops / (ms*1e6);
    const char* tag = (mode==0)?"naive":(mode==1)?"tiled":"cuBLAS";
    printf("[ATTN %s] time=%.3f ms rate=%.2f GF/s diff=%.3e\n", tag,ms,gflops,mad);
  }

  CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dCref));
  CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dO)); CUDA_CHECK(cudaFree(dOref));
  CUBLAS_CHECK(cublasDestroy(h)); CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

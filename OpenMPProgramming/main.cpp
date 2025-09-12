#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include "verify_mkl.hpp"
//Take help for SLurm job scheduling?
void dgemm_naive(const double* A, const double* B, double* C,
                 int m, int n, int k);

static int argi(char** a, int i){ return std::atoi(a[i]); }

int main(int argc, char** argv) {
    int m=2048, n=2048, k=2048;
    

    const std::size_t Asz = (std::size_t)m*k;
    const std::size_t Bsz = (std::size_t)k*n;
    const std::size_t Csz = (std::size_t)m*n;

    std::vector<double> A(Asz), B(Bsz), C(Csz, 0.0);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ud(-0.5,0.5);
    for (auto& x : A) x = ud(rng);
    for (auto& x : B) x = ud(rng);

    const double flop = 2.0 * (double)m * n * k;
    
    double t0 = omp_get_wtime();
    
    dgemm_naive(A.data(), B.data(), C.data(), m, n, k);
    
    double t1 = omp_get_wtime();

    double gflops = (flop / (t1 - t0)) * 1e-9;
    std::printf("Time: %.6f s,   Rate: %.2f GFLOP/s\n", (t1 - t0), gflops);

    try { verify_dgemm_vs_mkl(A.data(), B.data(), C.data(), m, n, k); }
    catch (const std::exception& e) { std::printf("[verify] %s\n", e.what()); }

    // similarly call other functions 
    return 0;
}

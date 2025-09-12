#pragma once
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstdio>

#ifdef USE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
#endif

inline double max_abs_diff(const double* C1, const double* C2, int m, int n) {
    double mad = 0.0;
    const std::size_t N = static_cast<std::size_t>(m) * n;
    for (std::size_t i = 0; i < N; ++i) {
        mad = std::max(mad, std::abs(C1[i] - C2[i]));
    }
    return mad;
}

inline void dgemm_mkl_ref(const double* A, const double* B, double* C,
                          int m, int n, int k, bool zeroC=true) {
#ifndef USE_MKL
    (void)A; (void)B; (void)C; (void)m; (void)n; (void)k; (void)zeroC;
    throw std::runtime_error("MKL not enabled.");
#else
    const double alpha = 1.0;
    const double beta  = zeroC ? 0.0 : 1.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
#endif
}

inline double verify_dgemm_vs_mkl(const double* A, const double* B,
                                  const double* C_student, int m, int n, int k,
                                  double tol = 1e-10) {
#ifndef USE_MKL
    std::puts("[verify] MKL not available. Skipping.");
    return 0.0;
#else
    std::vector<double> C_ref(static_cast<std::size_t>(m) * n, 0.0);
    dgemm_mkl_ref(A, B, C_ref.data(), m, n, k, true);
    double mad = max_abs_diff(C_student, C_ref.data(), m, n);
    std::printf("[verify] max |C_student - C_mkl| = %.3e %s\n",
                mad, (mad <= tol ? "PASS" : "FAIL"));
    return mad;
#endif
}

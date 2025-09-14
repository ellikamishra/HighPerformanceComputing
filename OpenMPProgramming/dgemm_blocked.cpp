// dgemm_blocked.cpp
#include <omp.h>
#include <algorithm>
#include <cstddef>

// Blocked DGEMM with tiles BM x BN x BK
void dgemm_blocked(const double* A, const double* B, double* C,
                   int m, int n, int k, int BM, int BN, int BK) {
    // Zero C
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        double* Ci = C + (std::size_t)i * n;
        for (int j = 0; j < n; ++j) Ci[j] = 0.0;
    }

    // Tile loops; parallelize over tile grid (i0,j0)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < m; i0 += BM) {
        for (int j0 = 0; j0 < n; j0 += BN) {
            const int imax = std::min(i0 + BM, m);
            const int jmax = std::min(j0 + BN, n);
            for (int k0 = 0; k0 < k; k0 += BK) {
                const int kmax = std::min(k0 + BK, k);
                for (int i = i0; i < imax; ++i) {
                    const double* Ai = A + (std::size_t)i * k;
                    double* Ci = C + (std::size_t)i * n;
                    for (int kk = k0; kk < kmax; ++kk) {
                        const double aik = Ai[kk];
                        const double* Bk = B + (std::size_t)kk * n + j0;
                        // vectorize across the j-tile
                        #pragma omp simd
                        for (int j = j0; j < jmax; ++j) {
                            Ci[j] += aik * Bk[j - j0];
                        }
                    }
                }
            }
        }
    }
}

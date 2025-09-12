#include <omp.h>
#include <algorithm>

// Blocked DGEMM with tunable block sizes BM, BN, BK
void dgemm_blocked(const double* A, const double* B, double* C,
                   int m, int n, int k, int BM, int BN, int BK) {
    // Zero C
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }

    // TODO: Add OpenMP parallelization (collapse over i0,j0 tile loops)
    for (int i0 = 0; i0 < m; i0 += BM) {
        for (int j0 = 0; j0 < n; j0 += BN) {
            int imax = std::min(i0 + BM, m);
            int jmax = std::min(j0 + BN, n);
            for (int k0 = 0; k0 < k; k0 += BK) {
                int kmax = std::min(k0 + BK, k);
                for (int i = i0; i < imax; i++) {
                    for (int kk = k0; kk < kmax; kk++) {
                        double a = A[i * k + kk];
                        for (int j = j0; j < jmax; j++) {
                            C[i * n + j] += a * B[kk * n + j];
                        }
                    }
                }
            }
        }
    }
}

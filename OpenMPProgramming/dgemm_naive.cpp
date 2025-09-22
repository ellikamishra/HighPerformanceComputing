#include <omp.h>
#include <cstddef>

// Naive DGEMM: C(m x n) = A(m x k) * B(k x n)
// Row-major, flattened arrays
void dgemm_naive(const double* A, const double* B, double* C,
                 int m, int n, int k) {
    // Zero C
    // parallelize with OpenMP
#ifdef SCHED_RUNTIME //using dynamic scheduling if enabled with flag to check performance of different scheduling techiniques
    #pragma omp parallel for schedule(runtime)
#else
    #pragma omp parallel for schedule(static) // static scheduling for good locality per row used as default scheduling
#endif
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }

    // Parallelize outer i-loop
#ifdef SCHED_RUNTIME
    #pragma omp parallel for schedule(runtime)
#else
    #pragma omp parallel for schedule(static) 
#endif
    for (int i = 0; i < m; ++i) {
        double* Ci = C + (std::size_t)i * n;
        const double* Ai = A + (std::size_t)i * k;
        for (int k1 = 0; k1 < k; ++k1) {
            const double aik = Ai[k1];
            const double* Bk = B + (std::size_t)k1 * n;
            #pragma omp simd                              //using simd to vectorize the innermost loop
            for (int j = 0; j < n; ++j) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

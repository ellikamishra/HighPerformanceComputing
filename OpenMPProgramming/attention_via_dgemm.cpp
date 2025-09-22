#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

void dgemm_naive(const double* A, const double* B, double* C,
                 int m, int n, int k);
// Add these prototypes near the top of attention_via_dgemm.cpp

void dgemm_blocked(const double* A, const double* B, double* C,
                   int m, int n, int k, int BM, int BN, int BK);


// Helper: transpose K(LxD) to KT(DxL)
static void transpose(const double* K, double* KT, int L, int D) 
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int d = 0; d < D; ++d)
        for (int l = 0; l < L; ++l)
            KT[(std::size_t)d * L + l] = K[(std::size_t)l * D + d];
}

// Helper: row-wise softmax
static void softmax_rows(double* S, int L) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < L; i++) {
        double* row = S + i * L;
        double m = row[0];
        for (int j = 1; j < L; j++) m = std::max(m, row[j]);
        double sum = 0.0;
        for (int j = 0; j < L; j++) {
            row[j] = std::exp(row[j] - m);
            sum += row[j];
        }
        for (int j = 0; j < L; j++) row[j] /= sum;
    }
}

void attention_via_dgemm(const double* Q, const double* K, const double* V,
                         double* O, int L, int D,
                         bool use_blocked, int BM, int BN, int BK) {
    
    // implement your code
    std::vector<double> KT((std::size_t)D * L);
    transpose(K, KT.data(), L, D);
    std::vector<double> S((std::size_t)L * L, 0.0);
    if (use_blocked) {
        dgemm_blocked(Q, KT.data(), S.data(), L, L, D, BM, BN, BK);
    } else {
        dgemm_naive(Q, KT.data(), S.data(), L, L, D);
    }
    const double scl = 1.0 / std::sqrt((double)D);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < L; ++i) {
        double* Si = S.data() + (std::size_t)i * L;
        for (int j = 0; j < L; ++j) Si[j] *= scl;
    }
    softmax_rows(S.data(), L); //softmax the rows of S is A=softmax(S)
    if (use_blocked) {
        dgemm_blocked(S.data(), V, O, L, D, L, BM, BN, BK);
    } else {
        dgemm_naive(S.data(), V, O, L, D, L);
    }
}

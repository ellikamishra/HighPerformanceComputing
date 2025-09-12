#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

void dgemm_naive(const double* A, const double* B, double* C,
                 int m, int n, int k);


// Helper: transpose K(LxD) to KT(DxL)
static void transpose(const double* K, double* KT, int L, int D) 
{
}

// Helper: row-wise softmax
static void softmax_rows(double* S, int L) {
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
}

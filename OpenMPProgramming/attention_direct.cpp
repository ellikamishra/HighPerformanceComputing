#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Direct loop-based attention (no DGEMM)
// We added suggested template without OpenMP
// Feel free to modify

void attention_direct(const double* Q, const double* K, const double* V,
                      double* O, int L, int D) {
    double scale = 1.0 / std::sqrt((double)D);
    
    // Initialize the output matrix
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < L; i++) {

        // compute one row of the attention matrix 

        
        // Compute softmax for this row
        // take a look at the helper code in attentiobn_via_dgemm.cpp
        // use simd here 
        std::vector<double> s(L);
        const double* Q1 = Q + (std::size_t)i * D;

        for (int j = 0; j< L; j++) {
            const double* Kj = K + (std::size_t)j * D;
            double dotp = 0.0;
            // vectorizing inner dot-product across D
            #pragma omp simd reduction(+:dotp)
            for (int d = 0; d < D; ++d) dotp += Q1[d] * Kj[d];
            s[j] = scale * dotp;
        }

        // stable softmax on s
        double mx = s[0];
        for (int j = 1; j< L; j++) mx = std::max(mx,s[j]);
        double sum = 0.0;
        for (int j = 0; j< L; ++j) {
            s[j] = std::exp(s[j] - mx);
            sum += s[j];
        }
        for (int j = 0; j< L; j++) s[j] /= sum;

        // generating one row of the output

        double* Oi = O + (std::size_t)i * D;
        for (int d = 0; d< D; ++d) Oi[d] = 0.0;

        for (int j = 0; j< L; ++j) {
            const double w = s[j];
            const double* Vj = V + (std::size_t)j * D; //Computing Vj here to avoid recomputation
            // vectorizing accumulation across D
            #pragma omp simd
            for (int d = 0; d< D; ++d) Oi[d] += w * Vj[d];
        }
    }
}

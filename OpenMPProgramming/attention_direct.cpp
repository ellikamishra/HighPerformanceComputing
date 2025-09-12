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
    

    for (int i = 0; i < L; i++) {

        // compute one row of the attention matrix 

        
        // Compute softmax for this row
        // take a look at the helper code in attentiobn_via_dgemm.cpp
        // use simd here 
        
        for (int j = 0; j < L; j++) s[j] /= sum;
        // end of softmax normalization

        // generate one row of the output
    }
}

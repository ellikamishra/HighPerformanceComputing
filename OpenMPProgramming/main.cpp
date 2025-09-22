// main.cpp
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "verify_mkl.hpp"   // uses MKL if built with USE_MKL=1; otherwise prints "Skipping."

// Prototypes
void dgemm_naive(const double* A, const double* B, double* C, int m, int n, int k);
void dgemm_blocked(const double* A, const double* B, double* C, int m, int n, int k,
                   int BM, int BN, int BK);
void attention_via_dgemm(const double* Q, const double* K, const double* V,
                         double* O, int L, int D,
                         bool use_blocked=false, int BM=128, int BN=128, int BK=64);
void attention_direct(const double* Q, const double* K, const double* V,
                      double* O, int L, int D);

static int argi(char** a, int i){ return std::atoi(a[i]); }
static void set_threads(int t) { omp_set_num_threads(t); }
static double now() { return omp_get_wtime(); }
static void banner(const char* what, int thr) {
    std::printf("=== %s | OMP_THREADS=%d ===\n", what, thr);
}

int main(int argc, char** argv) {
    // Defaults
    std::string mode = "dgemm"; // dgemm | attn_via | attn_direct
    int m=2048, n=2048, k=2048;
    int L=2048, D=1024;
    bool blocked=false;
    int BM=128, BN=128, BK=64;

    // Threads are controlled via OMP_NUM_THREADS from the environment.

    // Parse args
    for (int i=1;i<argc;i++){
        if (!std::strcmp(argv[i],"--mode") && i+1<argc) mode = argv[++i];
        else if (!std::strcmp(argv[i],"--m") && i+1<argc) m = argi(argv, ++i);
        else if (!std::strcmp(argv[i],"--n") && i+1<argc) n = argi(argv, ++i);
        else if (!std::strcmp(argv[i],"--k") && i+1<argc) k = argi(argv, ++i);
        else if (!std::strcmp(argv[i],"--L") && i+1<argc) L = argi(argv, ++i);
        else if (!std::strcmp(argv[i],"--D") && i+1<argc) D = argi(argv, ++i);
        else if (!std::strcmp(argv[i],"--blocked")) blocked = true;
        else if (!std::strcmp(argv[i],"--BM") && i+1<argc) BM = argi(argv, ++i);
        else if (!std::strcmp(argv[i],"--BN") && i+1<argc) BN = argi(argv, ++i);
        else if (!std::strcmp(argv[i],"--BK") && i+1<argc) BK = argi(argv, ++i);
    }

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ud(-0.5,0.5);

    auto do_dgemm = [&](int threads){
        set_threads(threads);
        std::vector<double> A((std::size_t)m*k), B((std::size_t)k*n), C((std::size_t)m*n,0.0);
        for (auto& x:A) x=ud(rng); for (auto& x:B) x=ud(rng);
        const double flop = 2.0 * (double)m * n * k;

        banner(blocked? "DGEMM (blocked)":"DGEMM (naive)", threads);
        double t0 = now();
        if (blocked) dgemm_blocked(A.data(), B.data(), C.data(), m, n, k, BM, BN, BK);
        else         dgemm_naive  (A.data(), B.data(), C.data(), m, n, k);
        double t1 = now();
        double gflops = flop / (t1-t0) * 1e-9;
        std::printf("Time: %.6f s  Rate: %.2f GFLOP/s\n", (t1-t0), gflops);

        // Optional verification (uses MKL if built with USE_MKL=1; else prints "Skipping.")
        try { verify_dgemm_vs_mkl(A.data(), B.data(), C.data(), m, n, k); }
        catch (const std::exception& e) { std::printf("[verify] %s\n", e.what()); }
    };

    auto do_attn_via = [&](int threads){
        set_threads(threads);
        std::vector<double> Q((std::size_t)L*D), K((std::size_t)L*D), V((std::size_t)L*D), O((std::size_t)L*D,0.0);
        for (auto& x:Q) x=ud(rng); for (auto& x:K) x=ud(rng); for (auto& x:V) x=ud(rng);
        // FLOPs (approx): Q*K^T (2*L*D*L) + A*V (2*L*L*D)
        const double flop = 4.0 * (double)L * L * D;

        banner(blocked? "Attention via DGEMM (blocked)":"Attention via DGEMM (naive)", threads);
        double t0 = now();
        attention_via_dgemm(Q.data(), K.data(), V.data(), O.data(), L, D, blocked, BM, BN, BK);
        double t1 = now();
        double gflops = flop / (t1-t0) * 1e-9;
        std::printf("Time: %.6f s  Rate: %.2f GFLOP/s (approx)\n", (t1-t0), gflops);
    };

    auto do_attn_direct = [&](int threads){
        set_threads(threads);
        std::vector<double> Q((std::size_t)L*D), K((std::size_t)L*D), V((std::size_t)L*D), O((std::size_t)L*D,0.0);
        for (auto& x:Q) x=ud(rng); for (auto& x:K) x=ud(rng); for (auto& x:V) x=ud(rng);
        // FLOPs (approx): scores + output accumulation ~ 4*L*L*D
        const double flop = 4.0 * (double)L * L * D;

        banner("Attention (direct)", threads);
        double t0 = now();
        attention_direct(Q.data(), K.data(), V.data(), O.data(), L, D);
        double t1 = now();
        double gflops = flop / (t1-t0) * 1e-9;
        std::printf("Time: %.6f s  Rate: %.2f GFLOP/s (approx)\n", (t1-t0), gflops);
    };

    if (mode=="dgemm") {
        int t = omp_get_max_threads();
        do_dgemm(t);
    } else if (mode=="attn_via") {
        int t = omp_get_max_threads();
        do_attn_via(t);
    } else if (mode=="attn_direct") {
        int t = omp_get_max_threads();
        do_attn_direct(t);
    } else {
        std::fprintf(stderr, "Unknown --mode %s\n", mode.c_str());
        return 2;
    }
    return 0;
}

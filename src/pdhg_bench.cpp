#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "pdhg.hpp"


template <int N>
void run(const std::string& name, const LP& lp, int iters, double eta) {
    std::vector<double> C((long)lp.A.n * N);
    for (int j = 0; j < lp.A.n; j++)
        for (int l = 0; l < N; l++) C[(long)j * N + l] = lp.c[j];
    double t0 = omp_get_wtime();
    if (N == 1) pdhg(lp, 0.0, iters, iters + 1, eta);
    else pdhg_batch<N>(lp, C, 0.0, iters, iters + 1, eta);
    double t = omp_get_wtime() - t0;
    printf("%s,%s,%d,%d,%d,%.4f,%.5f\n", name.c_str(), N == 1 ? "single" : "batch", N,
           omp_get_max_threads(), iters, t, t * 1e3 / iters / N);
}

// run serially 
void copies(const std::string& name, const LP& lp, int iters, double eta) {
    int P = omp_get_max_threads();
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static, 1)
    for (int p = 0; p < P; p++) pdhg(lp, 0.0, iters, iters + 1, eta);
    double t = omp_get_wtime() - t0;
    printf("%s,copies,%d,%d,%d,%.4f,%.5f\n", name.c_str(), P, P, iters, t, t * 1e3 / iters / P);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.mps [iters] [nocopies]\n", argv[0]); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 1000;
    LP lp = read_mps(argv[1]);
    std::string name = argv[1];
    name = name.substr(name.find_last_of('/') + 1);
    name = name.substr(0, name.find('.'));
    printf("matrix,mode,N,threads,iters,t_s,t_per_lp_iter_ms\n");
    double eta = 0.998 / op_norm(lp);
    run<1>(name, lp, iters, eta);
    run<8>(name, lp, iters, eta);
    if (argc < 4) copies(name, lp, iters, eta);
}

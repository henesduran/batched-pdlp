#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "mps.hpp"
#include "spmm.hpp"

template <int N>
void bench(const std::string& name, const char* op, const CSR& A, int reps) {
    std::vector<double> X((long)A.n * N, 1.0), Y((long)A.m * N);
    for (int r = 0; r < 3; r++) spmm<N>(A, X.data(), Y.data());

    std::vector<double> t(reps);
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        spmm<N>(A, X.data(), Y.data());
        t[r] = omp_get_wtime() - t0;
    }
    std::sort(t.begin(), t.end());
    double med = t[reps / 2];
    double bytes = 12.0 * A.nnz() + 4.0 * (A.m + 1) + 8.0 * N * ((double)A.n + A.m);
    printf("%s,%s,%d,%d,%d,%.4f,%.4f,%.2f\n", name.c_str(), op, N, omp_get_max_threads(),
           reps, med * 1e3, med * 1e3 / N, bytes / med / 1e9);
}

template <int N>
void both(const std::string& name, const LP& lp, int reps) {
    bench<N>(name, "Ax", lp.A, reps);
    bench<N>(name, "ATy", lp.At, reps);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.mps [reps]\n", argv[0]); return 1; }
    int reps = argc > 2 ? atoi(argv[2]) : 50;
    LP lp = read_mps(argv[1]);
    std::string name = argv[1];
    name = name.substr(name.find_last_of('/') + 1);
    name = name.substr(0, name.find('.'));

    printf("matrix,op,N,threads,reps,t_ms,t_per_lp_ms,GBps\n");
    both<1>(name, lp, reps);  both<2>(name, lp, reps);  both<4>(name, lp, reps);
    both<8>(name, lp, reps);  both<16>(name, lp, reps); both<32>(name, lp, reps);
    both<64>(name, lp, reps);
}

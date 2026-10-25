#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "mps.hpp"
#include "spmm.hpp"

// results are same or not?
template <int N>
bool check(const CSR& A) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> X((long)A.n * N), Y((long)A.m * N), x(A.n), y(A.m);
    for (double& v : X) v = U(rng);
    spmm<N>(A, X.data(), Y.data());

    double err = 0.0;
    for (int l = 0; l < N; l++) {
        for (int j = 0; j < A.n; j++) x[j] = X[(long)j * N + l];
        spmv(A, x.data(), y.data());
        for (int i = 0; i < A.m; i++)
            err = std::fmax(err, std::fabs(Y[(long)i * N + l] - y[i]) / (1.0 + std::fabs(y[i])));
    }
    printf("N=%2d  max rel err = %.2e\n", N, err);
    return err < 1e-12;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.mps\n", argv[0]); return 1; }
    LP lp = read_mps(argv[1]);
    bool ok = true;
    for (const CSR* M : {&lp.A, &lp.At}) {
        printf("%s (%d x %d)\n", M == &lp.A ? "A" : "At", M->m, M->n);
        ok &= check<1>(*M) & check<2>(*M) & check<4>(*M) & check<8>(*M)
            & check<16>(*M) & check<32>(*M) & check<64>(*M);
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

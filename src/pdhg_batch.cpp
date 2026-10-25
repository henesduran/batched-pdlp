#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "pdhg.hpp"

constexpr int N = 8;


int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.mps [delta] [max_iter]\n", argv[0]); return 1; }
    LP lp = read_mps(argv[1]);
    double delta = argc > 2 ? atof(argv[2]) : 0.01;
    int max_iter = argc > 3 ? atoi(argv[3]) : 1000000;
    const int n = lp.A.n;
    std::vector<double> C((long)n * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int j = 0; j < n; j++)
        for (int l = 0; l < N; l++) C[(long)j * N + l] = lp.c[j] * (l ? 1 + delta * U(rng) : 1.0);

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    std::vector<Sol> b = pdhg_batch<N>(lp, C, 1e-4, max_iter);
    double tb = std::chrono::duration<double>(clk::now() - t0).count();

    double ts = 0;
    int same = 0;
    printf("lp  iters_batch  iters_single  pobj_batch        pobj_single       bitwise\n");
    for (int l = 0; l < N; l++) {
        LP v = lp;
        for (int j = 0; j < n; j++) v.c[j] = C[(long)j * N + l];
        t0 = clk::now();
        Sol s = pdhg(v, 1e-4, max_iter);
        ts += std::chrono::duration<double>(clk::now() - t0).count();
        bool eq = b[l].ok && s.ok && b[l].x == s.x && b[l].y == s.y;
        same += eq;
        printf("%d   %11d  %12d  %-16.10g  %-16.10g  %s\n", l, b[l].iters, s.iters, b[l].pobj, s.pobj,
               eq ? "yes" : "NO");
    }
    printf("bitwise equal: %d/%d   time batch %.3f s, %d singles %.3f s\n", same, N, tb, N, ts);
    return same == N ? 0 : 1;
}

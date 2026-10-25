#include <algorithm>
#include <cmath>
#include <cstdio>
#include "pdhg.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.mps\n", argv[0]); return 1; }
    LP lp = read_mps(argv[1]);
    const CSR& A = lp.A;
    CSR Att = transpose(A);
    bool ok = Att.row_ptr == lp.At.row_ptr && Att.col_idx == lp.At.col_idx && Att.val == lp.At.val;

    int rmin = A.n, rmax = 0;
    for (int i = 0; i < A.m; i++) {
        int len = A.row_ptr[i + 1] - A.row_ptr[i];
        rmin = std::min(rmin, len); rmax = std::max(rmax, len);
    }
    printf("m=%d n=%d nnz=%ld\n", A.m, A.n, A.nnz());
    printf("row nnz: min=%d max=%d avg=%.2f\n", rmin, rmax, (double)A.nnz() / A.m);
    printf("A size: %.1f MB (val+col_idx)\n", A.nnz() * 12.0 / 1e6);
    int eq = 0, rngd = 0, fr = 0, xfree = 0, xbox = 0;
    for (int i = 0; i < A.m; i++) {
        if (lp.l[i] == lp.u[i]) eq++;
        else if (std::isinf(lp.l[i]) && std::isinf(lp.u[i])) fr++;
        else if (!std::isinf(lp.l[i]) && !std::isinf(lp.u[i])) rngd++;
    }
    for (int j = 0; j < A.n; j++) {
        if (std::isinf(lp.xl[j]) && std::isinf(lp.xu[j])) xfree++;
        else if (!std::isinf(lp.xl[j]) && !std::isinf(lp.xu[j])) xbox++;
    }
    printf("rows: %d eq, %d ranged, %d one-sided, %d free\n", eq, rngd, A.m - eq - rngd - fr, fr);
    printf("cols: %d boxed, %d free, %d one-sided; obj_off=%g\n", xbox, xfree, A.n - xbox - xfree, lp.obj_off);
    printf("||A||_2 ~ %.6g\n", op_norm(lp));
    printf("transpose(A) == At: %s\n", ok ? "yes" : "NO");
    return ok ? 0 : 1;
}

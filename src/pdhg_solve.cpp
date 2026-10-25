#include <cstdio>
#include <cstdlib>
#include "pdhg.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.mps [max_iter] [eps]\n", argv[0]); return 1; }
    LP lp = read_mps(argv[1]);
    int max_iter = argc > 2 ? atoi(argv[2]) : 1000000;
    Sol s = pdhg(lp, argc > 3 ? atof(argv[3]) : 1e-4, max_iter);
    printf("%s iters=%d pobj=%.10g dobj=%.10g rp=%.3g rd=%.3g\n",
           s.ok ? "OPTIMAL" : "MAXITER", s.iters, s.pobj, s.dobj, s.rp, s.rd);
    return s.ok ? 0 : 2;
}

// USAGE:
//   ./solve single <file.mps> [key=val ...]
//       Solve a single LP
//
//   ./solve batch <file.mps> <delta> [key=val ...]
//       Solve 8 LPs in batch vs 8 singles. Verifies bitwise match
//
//   ./solve many <file.mps> <delta> <nlp> <chunk> [key=val ...]
//       Solve <nlp> LPs in batches of <chunk>
//       Keys: par=1 (run chunks in parallel), sol=<dir> (save solutions)
//
//   ./solve sb <file.mps> <nb> [key=val ...]
//       MIP Strong Branching: solve root LP, branch on top <nb> fractional vars
//
//   ./solve kkt <file.mps> <delta> <nlp> <eps> <sol[:lp]> ...
//       Validate a solution file (ours or OR-Tools) against KKT conditions
//
//   ./solve mps <file.mps> <delta> <lp> <out.mps>
//       Export one perturbed LP to standard MPS format (for OR-Tools)
//
// OPTIONS (key=val):
//   eps=1e-4          Convergence tolerance
//   maxit=1000000     Iteration limit
//   freq=200          KKT and restart check frequency
//   compact=1         Remove finished LPs from batch (1=on, 0=off)
//   restart=1         Adaptive restart (1=on, 0=off)
//   pid=1             PID update of primal weight w
//   halpern=1         Halpern iteration (1=on, 0=off)
//   gamma=1.0         Reflection parameter
//   geo=1,ruiz=1,pc=1 Matrix scaling toggles
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include "pdhg.hpp"

constexpr int N = 8;

//generate pertubated LPs
static std::vector<double> perturb(const LP& lp, double delta, int nl = N) {
    const int n = lp.A.n;
    std::vector<double> C((long)n * nl);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int j = 0; j < n; j++)
        for (int l = 0; l < nl; l++) C[(long)j * nl + l] = lp.c[j] * (l ? 1 + delta * U(rng) : 1.0);
    return C;
}

// pdhg_batch<k> for a runtime k <= K
template <int K>
std::vector<Sol> batch_k(int k, const LP& lp, const std::vector<double>& C, double eps, int maxit, Opts o,
                         const Branch* br = nullptr, const Sol* warm = nullptr, Timings* tm = nullptr) {
    if constexpr (K > 1) if (k < K) return batch_k<K - 1>(k, lp, C, eps, maxit, o, br, warm, tm);
    return pdhg_batch<K>(lp, C, eps, maxit, o, br, warm, tm);
}
constexpr int KMAX = 32;

// for b < nbatch as independent batches side by side, one thread each
//so depth is only 1
// (the batch's own parallel loops are nested, hence inactive: they run on that one thread)
template <class F>
void side_by_side(int nbatch, F run) {
    omp_set_max_active_levels(1);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < nbatch; b++) run(b);
}

static void write_mps(const LP& lp, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    const int m = lp.A.m, n = lp.A.n;
    fprintf(f, "NAME P\nROWS\n N OBJ\n");
    for (int i = 0; i < m; i++) {
        bool lo = std::isfinite(lp.l[i]), up = std::isfinite(lp.u[i]);
        fprintf(f, " %c R%d\n", lo && up && lp.l[i] == lp.u[i] ? 'E' : lo ? 'G' : 'L', i);
    }
    fprintf(f, "COLUMNS\n");
    for (int j = 0; j < n; j++) {
        fprintf(f, " C%d OBJ %.17g\n", j, lp.c[j]);
        for (int k = lp.At.row_ptr[j]; k < lp.At.row_ptr[j + 1]; k++)
            fprintf(f, " C%d R%d %.17g\n", j, lp.At.col_idx[k], lp.At.val[k]);
    }
    fprintf(f, "RHS\n");
    if (lp.obj_off != 0) fprintf(f, " RHS OBJ %.17g\n", -lp.obj_off);
    for (int i = 0; i < m; i++) {
        bool lo = std::isfinite(lp.l[i]);
        double b = lo ? lp.l[i] : lp.u[i];
        if (b != 0) fprintf(f, " RHS R%d %.17g\n", i, b);
    }
    fprintf(f, "RANGES\n");
    for (int i = 0; i < m; i++)
        if (std::isfinite(lp.l[i]) && std::isfinite(lp.u[i]) && lp.l[i] != lp.u[i])
            fprintf(f, " RNG R%d %.17g\n", i, lp.u[i] - lp.l[i]);
    fprintf(f, "BOUNDS\n");
    for (int j = 0; j < n; j++) {
        double lo = lp.xl[j], up = lp.xu[j];
        if (lo == up) { fprintf(f, " FX BND C%d %.17g\n", j, lo); continue; }
        if (lo == -INFINITY && up == INFINITY) { fprintf(f, " FR BND C%d\n", j); continue; }
        if (lo == -INFINITY) fprintf(f, " MI BND C%d\n", j);
        else if (lo != 0) fprintf(f, " LO BND C%d %.17g\n", j, lo);
        if (up != INFINITY) fprintf(f, " UP BND C%d %.17g\n", j, up);
    }
    fprintf(f, "ENDATA\n");
    fclose(f);
}

// solutions in OR-Tools' MPSolutionResponse text format, so one reader serves both solvers
static void write_sol(const Sol& s, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { perror(path.c_str()); exit(1); }
    for (double v : s.x) fprintf(f, "variable_value: %.17g\n", v);
    for (double v : s.y) fprintf(f, "dual_value: %.17g\n", v);
    fclose(f);
}

static bool read_sol(const std::string& path, std::vector<double>& x, std::vector<double>& y) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.compare(0, 16, "variable_value: ")) x.push_back(strtod(line.c_str() + 16, nullptr));
        else if (!line.compare(0, 12, "dual_value: ")) y.push_back(strtod(line.c_str() + 12, nullptr));
    }
    return true;
}

// one KKT test for every solver, on the unscaled LP, from (x, y) alone. y is first projected onto its sign
// constraints (y_i > 0 needs a finite l_i, y_i < 0 a finite u_i); the reduced cost r = c - A'y splits into the part
// the bounds of x can carry and the rest, the dual residual. Norms as in kkt()
struct Check { double rp, rd, gap, pobj, dobj, yproj; bool ok; };
static Check check_sol(const LP& lp, const std::vector<double>& x, std::vector<double> y, double eps) {
    const int m = lp.A.m, n = lp.A.n;
    double rp = 0, bn = 0, rd = 0, cn = 0, yproj = 0, pobj = lp.obj_off, dobj = lp.obj_off;
    for (int i = 0; i < m; i++) {
        double l = lp.l[i], u = lp.u[i];
        if ((y[i] > 0 && !std::isfinite(l)) || (y[i] < 0 && !std::isfinite(u))) { yproj = std::max(yproj, std::fabs(y[i])); y[i] = 0; }
        double a = 0;
        for (int k = lp.A.row_ptr[i]; k < lp.A.row_ptr[i + 1]; k++) a += lp.A.val[k] * x[lp.A.col_idx[k]];
        double v = a < l ? l - a : a > u ? a - u : 0.0;
        rp += v * v;
        if (std::isfinite(l)) bn += l * l;
        if (std::isfinite(u) && u != l) bn += u * u;
        dobj += y[i] > 0 ? y[i] * l : y[i] < 0 ? y[i] * u : 0.0;
    }
    for (int j = 0; j < n; j++) {
        double v = x[j] < lp.xl[j] ? lp.xl[j] - x[j] : x[j] > lp.xu[j] ? x[j] - lp.xu[j] : 0.0;  // x outside its bounds
        rp += v * v;
        double r = lp.c[j];
        for (int k = lp.At.row_ptr[j]; k < lp.At.row_ptr[j + 1]; k++) r -= lp.At.val[k] * y[lp.At.col_idx[k]];
        bool lo = std::isfinite(lp.xl[j]), up = std::isfinite(lp.xu[j]);
        double lam = lo && up ? r : lo ? std::max(r, 0.0) : up ? std::min(r, 0.0) : 0.0;
        rd += (r - lam) * (r - lam);
        cn += lp.c[j] * lp.c[j];
        pobj += lp.c[j] * x[j];
        dobj += lam > 0 ? lam * lp.xl[j] : lam < 0 ? lam * lp.xu[j] : 0.0;
    }
    Check c{std::sqrt(rp) / (1 + std::sqrt(bn)), std::sqrt(rd) / (1 + std::sqrt(cn)), 0, pobj, dobj, yproj, false};
    c.gap = std::fabs(pobj - dobj) / (1 + std::fabs(pobj) + std::fabs(dobj));
    c.ok = c.rp <= eps && c.rd <= eps && c.gap <= eps;
    return c;
}

// integer columns from the COLUMNS markers (read_mps skips them); columns numbered as read_mps does
static std::vector<char> int_cols(const char* path, int n) {
    std::ifstream f(path);
    std::vector<char> isint(n, 0);
    std::string line, sec, col, cur;
    bool in = false;
    int j = -1;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '*') continue;
        if (!isspace((unsigned char)line[0])) { sec = line.substr(0, line.find_first_of(" \t")); continue; }
        if (sec != "COLUMNS") continue;
        if (line.find("'MARKER'") != std::string::npos) { in = line.find("'INTORG'") != std::string::npos; continue; }
        std::istringstream(line) >> col;
        if (col != cur) { cur = col; j++; if (in && j < n) isint[j] = 1; }
    }
    return isint;
}

// one batch of nb children of the same LP, all with cost c
static std::vector<Sol> branch_batch(int nb, const LP& lp, const Branch* br, const Sol* warm, double eps, int maxit, Opts o) {
    std::vector<double> C((long)lp.A.n * nb);
    for (int j = 0; j < lp.A.n; j++)
        for (int l = 0; l < nb; l++) C[(long)j * nb + l] = lp.c[j];
    return batch_k<KMAX>(nb, lp, C, eps, maxit, o, br, warm);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s single|batch|mps file.mps [delta] [lp out.mps] [key=val ...]\n", argv[0]); return 1; }
    bool batch = !strcmp(argv[1], "batch"), many = !strcmp(argv[1], "many"), sb = !strcmp(argv[1], "sb");
    LP lp = read_mps(argv[2]);
    const int n = lp.A.n;
    if (!strcmp(argv[1], "kkt")) {
        if (argc < 7) { fprintf(stderr, "usage: %s kkt file.mps delta nlp eps sol[:lp] ...\n", argv[0]); return 1; }
        const double d = atof(argv[3]), eps = atof(argv[5]);
        const int nl = atoi(argv[4]);
        std::vector<double> C = perturb(lp, d, nl);
        fprintf(stderr, "# m n %d %d\n", lp.A.m, n);
        printf("file,lp,status,rel_p,rel_d,gap,pobj,dobj,y_projected,kkt_ok\n");
        for (int a = 6; a < argc; a++) {
            std::string f = argv[a];
            int l = 0;
            size_t c = f.rfind(':');
            if (c != std::string::npos) { l = atoi(f.c_str() + c + 1); f = f.substr(0, c); }
            std::vector<double> x, y;
            if (!read_sol(f, x, y) || (int)x.size() != n || (int)y.size() != lp.A.m) {
                printf("%s,%d,%s,,,,,,,0\n", f.c_str(), l, x.empty() ? "NOSOL" : "BADSIZE");
                continue;
            }
            LP v = lp;
            for (int j = 0; j < n; j++) v.c[j] = C[(long)j * nl + l];
            Check k = check_sol(v, x, y, eps);
            printf("%s,%d,OK,%.3e,%.3e,%.3e,%.12g,%.12g,%.1e,%d\n", f.c_str(), l, k.rp, k.rd, k.gap, k.pobj, k.dobj, k.yproj, k.ok);
        }
        return 0;
    }
    if (!strcmp(argv[1], "mps")) {
        if (argc < 6) { fprintf(stderr, "usage: %s mps file.mps delta lp out.mps [nlp]\n", argv[0]); return 1; }
        int nl = argc > 6 ? atoi(argv[6]) : N, l = atoi(argv[4]);
        std::vector<double> C = perturb(lp, atof(argv[3]), nl);
        for (int j = 0; j < n; j++) lp.c[j] = C[(long)j * nl + l];
        write_mps(lp, argv[5]);
        return 0;
    }
    double delta = (batch || many) && argc > 3 ? atof(argv[3]) : 0.0;
    if (many && argc < 6) { fprintf(stderr, "usage: %s many file.mps delta nlp chunk [key=val ...]\n", argv[0]); return 1; }
    double eps = 1e-4;
    int maxit = 1000000, pick = 0;
    Opts o;
    std::string tag = "full", out, sol;
    double capm = 10, ftol = 0.01, infp = 1e-3;
    bool check = false, par = false;
    int chunk = 0;
    for (int a = many ? 6 : batch || sb ? 4 : 3; a < argc; a++) {
        std::string kv = argv[a], k = kv.substr(0, kv.find('='));
        double v = atof(kv.substr(kv.find('=') + 1).c_str());
        if (k == "out" && sb) out = kv.substr(4);
        else if (k == "cap" && sb) capm = v;
        else if (k == "ftol" && sb) ftol = v;
        else if (k == "infp" && sb) infp = v;
        else if (k == "check" && sb) check = v;
        else if (k == "chunk" && sb) chunk = (int)v;
        else if (k == "par" && many) par = v;
        else if (k == "sol" && (many || sb)) sol = kv.substr(4);
        else if (k == "eps") eps = v;
        else if (k == "maxit") maxit = (int)v;
        else if (k == "gamma") o.gamma = v;
        else if (k == "halpern") o.halpern = v;
        else if (k == "restart") o.restart = v;
        else if (k == "pid") o.pid = v;
        else if (k == "geo") o.geo = v;
        else if (k == "ruiz") o.ruiz = v;
        else if (k == "pc") o.pc = v;
        else if (k == "bo") o.bound_obj = v;
        else if (k == "freq") o.freq = (int)v;
        else if (k == "compact") o.compact = v;
        else if (k == "delta" && !batch && !many && !sb) delta = v;
        else if (k == "lp" && !batch && !many && !sb) pick = (int)v;
        else { fprintf(stderr, "unknown key %s\n", k.c_str()); return 1; }
        tag = tag == "full" ? kv : tag + "," + kv;
    }
    std::string name = argv[2];
    name = name.substr(name.find_last_of('/') + 1);
    name = name.substr(0, name.find('.'));

    if (sb) {
        const int nb = atoi(argv[3]);
        std::vector<char> isint = int_cols(argv[2], n);
        double t0 = omp_get_wtime();
        Sol root = pdhg(lp, eps, maxit, o);
        const double troot = omp_get_wtime() - t0;
        if (!root.ok) { fprintf(stderr, "%s: root LP not solved\n", name.c_str()); return 2; }
        auto frac = [&](int j) { double f = root.x[j] - std::floor(root.x[j]); return std::min(f, 1 - f); };
        std::vector<int> cand;
        for (int j = 0; j < n; j++) if (isint[j] && frac(j) > ftol) cand.push_back(j);
        std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return frac(a) > frac(b); });
        printf("# root iters %d time %.3f s pobj %.10g, %zu fractional of %d integer\n", root.iters, troot, root.pobj,
               cand.size(), (int)std::count(isint.begin(), isint.end(), 1));
        if ((int)cand.size() < nb) { fprintf(stderr, "%s: only %zu fractional\n", name.c_str(), cand.size()); return 3; }
        std::vector<Branch> br[2] = {std::vector<Branch>(nb), std::vector<Branch>(nb)};  // down, up
        for (int l = 0; l < nb; l++) {
            int j = cand[l];
            br[0][l] = {j, lp.xl[j], std::floor(root.x[j])};
            br[1][l] = {j, std::ceil(root.x[j]), lp.xu[j]};
        }
        const int cap = (int)(capm * root.iters);
        std::vector<Sol> res[2][2];  // [warm, cold][down, up]
        double tm[2] = {};
        for (int w = 0; w < 2; w++) {
            const Sol* warm = w ? nullptr : &root;
            t0 = omp_get_wtime();
            if (!chunk) {
                for (int d = 0; d < 2; d++) res[w][d] = branch_batch(nb, lp, br[d].data(), warm, eps, cap, o);
            } else {  // children c = d * nb + l, batches of chunk consecutive children
                std::vector<Branch> all(br[0]);
                all.insert(all.end(), br[1].begin(), br[1].end());
                std::vector<Sol> out(2 * nb);
                side_by_side((2 * nb + chunk - 1) / chunk, [&](int b) {
                    int c0 = b * chunk, k = std::min(chunk, 2 * nb - c0);
                    std::vector<Sol> r = branch_batch(k, lp, all.data() + c0, warm, eps, cap, o);
                    for (int i = 0; i < k; i++) out[c0 + i] = std::move(r[i]);
                });
                for (int d = 0; d < 2; d++) res[w][d].assign(out.begin() + d * nb, out.begin() + (d + 1) * nb);
            }
            tm[w] = omp_get_wtime() - t0;
        }
        auto st = [&](const Sol& s) { return s.ok ? "OPTIMAL" : s.rp > infp ? "INFEAS?" : "LIMIT"; };
        printf("matrix,nb,side,lp,var,xroot,lo,hi,warm_status,warm_iters,warm_obj,warm_rp,cold_status,cold_iters,cold_obj\n");
        for (int d = 0; d < 2; d++)
            for (int l = 0; l < nb; l++) {
                const Branch& b = br[d][l];
                const Sol &a = res[0][d][l], &c = res[1][d][l];
                printf("%s,%d,%s,%d,%d,%.6g,%.6g,%.6g,%s,%d,%.10g,%.3g,%s,%d,%.10g\n", name.c_str(), nb, d ? "up" : "down", l,
                       b.j, root.x[b.j], b.lo, b.hi, st(a), a.iters, a.pobj, a.rp, st(c), c.iters, c.pobj);
            }
        printf("# time root %.3f s, cap %d, warm %.3f s, cold %.3f s, per child warm %.4f cold %.4f, chunk %d\n", troot, cap,
               tm[0], tm[1], tm[0] / (2 * nb), tm[1] / (2 * nb), chunk);
        fflush(stdout);
        if (!out.empty())
            for (int d = 0; d < 2; d++)
                for (int l = 0; l < nb; l++) {
                    LP v = lp;
                    v.xl[br[d][l].j] = br[d][l].lo; v.xu[br[d][l].j] = br[d][l].hi;
                    write_mps(v, (out + "/" + (d ? "u" : "d") + std::to_string(l) + ".mps").c_str());
                }
        if (!sol.empty())
            for (int w = 0; w < 2; w++)
                for (int d = 0; d < 2; d++)
                    for (int l = 0; l < nb; l++)
                        write_sol(res[w][d][l], sol + "/" + (w ? "c" : "w") + (d ? "u" : "d") + std::to_string(l) + ".sol");
        if (check) {  // each warm child alone, its bounds changed in the LP itself
            int same = 0;
            for (int d = 0; d < 2; d++)
                for (int l = 0; l < nb; l++) {
                    LP v = lp;
                    v.xl[br[d][l].j] = br[d][l].lo; v.xu[br[d][l].j] = br[d][l].hi;
                    Sol s = pdhg(v, eps, cap, o, &root);
                    const Sol& a = res[0][d][l];
                    same += s.ok == a.ok && s.iters == a.iters && s.x == a.x && s.y == a.y && s.rp == a.rp;
                }
            printf("# bitwise %d/%d\n", same, 2 * nb);
        }
        return 0;
    }
    if (many) {
        const int nl = atoi(argv[4]), ch = atoi(argv[5]);
        std::vector<double> C = perturb(lp, delta, nl);
        std::vector<Sol> all(nl);
        const int nbatch = (nl + ch - 1) / ch;
        std::vector<Timings> chunk_tm(nbatch);
        auto run = [&](int b) {
            int s0 = b * ch, k = std::min(ch, nl - s0);
            std::vector<double> Ck((long)n * k);
            for (int j = 0; j < n; j++)
                for (int l = 0; l < k; l++) Ck[(long)j * k + l] = C[(long)j * nl + s0 + l];
            Timings tm;
            std::vector<Sol> r = batch_k<KMAX>(k, lp, Ck, eps, maxit, o, nullptr, nullptr, &tm);
            for (int l = 0; l < k; l++) all[s0 + l] = std::move(r[l]);
            chunk_tm[b] = tm;
        };
        double t0 = omp_get_wtime(); //walltime
        if (par) side_by_side(nbatch, run);
        else for (int b = 0; b < nbatch; b++) run(b);
        double t = omp_get_wtime() - t0;
        Timings tot_tm;
        for (const auto& tm : chunk_tm) {
            tot_tm.t_setup       += tm.t_setup;
            tot_tm.t_spmm_main   += tm.t_spmm_main;
            tot_tm.t_spmm_check  += tm.t_spmm_check;
            tot_tm.t_check_other += tm.t_check_other;
            tot_tm.t_other       += tm.t_other;
            tot_tm.t_total       += tm.t_total;
        }
        double t_spmm_total = tot_tm.t_spmm_main + tot_tm.t_spmm_check;
        printf("matrix,delta,nlp,chunk,lp,status,iters,pobj,dobj\n");
        for (int l = 0; l < nl; l++)
            printf("%s,%g,%d,%d,%d,%s,%d,%.10g,%.10g\n", name.c_str(), delta, nl, ch, l,
                   all[l].ok ? "OPTIMAL" : "MAXITER", all[l].iters, all[l].pobj, all[l].dobj);
        long total_iters = 0;
        for (int l = 0; l < nl; l++) total_iters += all[l].iters;
        double ms_per_lp_iter = total_iters > 0 ? (t / total_iters) * 1000.0 : 0.0;
        double t_base = tot_tm.t_total > 0 ? tot_tm.t_total : t;
        printf("# time total %.3f s, per LP %.3f s, total_lp_iters %ld, per LP-iter %.4f ms, par %d\n",
               t, t / nl, total_iters, ms_per_lp_iter, (int)par);
        printf("# breakdown: setup %.3f s (%.1f%%), spmm_main %.3f s (%.1f%%), spmm_check %.3f s (%.1f%%), spmm_total %.3f s (%.1f%%), check_other %.3f s (%.1f%%), other %.3f s (%.1f%%)\n",
               tot_tm.t_setup, t_base > 0 ? 100.0 * tot_tm.t_setup / t_base : 0.0,
               tot_tm.t_spmm_main, t_base > 0 ? 100.0 * tot_tm.t_spmm_main / t_base : 0.0,
               tot_tm.t_spmm_check, t_base > 0 ? 100.0 * tot_tm.t_spmm_check / t_base : 0.0,
               t_spmm_total, t_base > 0 ? 100.0 * t_spmm_total / t_base : 0.0,
               tot_tm.t_check_other, t_base > 0 ? 100.0 * tot_tm.t_check_other / t_base : 0.0,
               tot_tm.t_other, t_base > 0 ? 100.0 * tot_tm.t_other / t_base : 0.0);
        fflush(stdout);
        if (!sol.empty())
            for (int l = 0; l < nl; l++) write_sol(all[l], sol + "/b" + std::to_string(l) + ".sol");
        return 0;
    }
    std::vector<double> C = perturb(lp, delta);

    if (!batch) {
        for (int j = 0; j < n; j++) lp.c[j] = C[(long)j * N + pick];
        double t0 = omp_get_wtime();
        Sol s = pdhg(lp, eps, maxit, o);
        printf("matrix,config,status,iters,pobj,dobj,rel_p,rel_d,time_s\n");
        printf("%s,\"%s\",%s,%d,%.10g,%.10g,%.3g,%.3g,%.3f\n", name.c_str(), tag.c_str(),
               s.ok ? "OPTIMAL" : "MAXITER", s.iters, s.pobj, s.dobj, s.rp, s.rd, omp_get_wtime() - t0);
        return s.ok ? 0 : 2;
    }
    //solve as batch
    double t0 = omp_get_wtime();
    std::vector<Sol> b = pdhg_batch<N>(lp, C, eps, maxit, o);
    double tb = omp_get_wtime() - t0, ts = 0;
    int same = 0;
    printf("# time batch %.3f s\n", tb);
    fflush(stdout);  // a timeout during the singles must not lose the batch time
    printf("matrix,delta,lp,iters_batch,iters_single,pobj_batch,pobj_single,bitwise,time_single_s\n");
    //solve individually
    for (int l = 0; l < N; l++) {
        LP v = lp;
        for (int j = 0; j < n; j++) v.c[j] = C[(long)j * N + l];
        t0 = omp_get_wtime();
        Sol s = pdhg(v, eps, maxit, o);
        double t = omp_get_wtime() - t0;
        ts += t;
        bool eq = b[l].ok && s.ok && b[l].x == s.x && b[l].y == s.y;
        same += eq;
        printf("%s,%g,%d,%d,%d,%.10g,%.10g,%s,%.3f\n", name.c_str(), delta, l, b[l].iters, s.iters, b[l].pobj,
               s.pobj, eq ? "yes" : "NO", t);
        fflush(stdout);
    }
    //!!!!!! USE -ffp-contract=off HERE!!! otherwise fma ruins bitwise check
    printf("# bitwise %d/%d, time batch %.3f s, %d singles %.3f s\n", same, N, tb, N, ts);
    return 0;
}

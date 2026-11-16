//main algorithm

#pragma once
#include <cmath>
#include <random>
#include <vector>
#include "mps.hpp"
#include "spmm.hpp"

inline double nrm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double a : v) s += a * a;
    return std::sqrt(s);
}

//operator Norm using power iteraiton = |A|
inline double op_norm(const LP& lp, int iters = 100) {
    std::vector<double> x(lp.A.n), Ax(lp.A.m), y(lp.A.n);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (double& v : x) v = U(rng);
    double s = nrm2(x);
    for (double& v : x) v /= s;
    for (int it = 0; it < iters; it++) {
        spmv(lp.A, x.data(), Ax.data());
        spmv(lp.At, Ax.data(), y.data());
        s = nrm2(y);
        for (int j = 0; j < lp.A.n; j++) x[j] = y[j] / s;
    }
    return std::sqrt(s);
}

//LP l replaces the bounds of one variable j by [lo, hi] (a strong-branching child); j < 0: none
struct Branch { int j = -1; double lo = 0, hi = 0; };

struct Sol { std::vector<double> x, y; double pobj, dobj, rp, rd, w; int iters; bool ok; };  // w: last primal weight 

//y0 = dual_current,a=constraint for ith row, l-u = bounds, sigma=dual step size
inline double dual_step(double y0, double a, double l, double u, double sigma) {
    //if constraint is too low,push up
    //if constraint is too high,push down
    double up = y0 + sigma * (l - a), dn = y0 + sigma * (u - a);
    return up > 0 ? up : dn < 0 ? dn : 0.0;
}

//kkt check = primal , dual , gap
inline bool kkt(const LP& lp, const double* c, const std::vector<double>& x, const std::vector<double>& y,
                const std::vector<double>& Ax, const std::vector<double>& r, const std::vector<double>& dres,
                double eps, Sol& s, const Branch& b = {}) {
    double rp = 0, bn = 0, rd = 0, cn = 0, pobj = lp.obj_off, dobj = lp.obj_off;
    for (int i = 0; i < lp.A.m; i++) {
        double l = lp.l[i], u = lp.u[i];
        double v = Ax[i] < l ? l - Ax[i] : Ax[i] > u ? Ax[i] - u : 0.0;
        rp += v * v;
        if (std::isfinite(l)) bn += l * l;
        if (std::isfinite(u) && u != l) bn += u * u;
        dobj += y[i] > 0 ? y[i] * l : y[i] < 0 ? y[i] * u : 0.0;  // y = 0, u = inf would give Nan
    }
    for (int j = 0; j < lp.A.n; j++) {
        double xl = j == b.j ? b.lo : lp.xl[j], xu = j == b.j ? b.hi : lp.xu[j], rc = r[j] + dres[j];
        bool lo = std::isfinite(xl), up = std::isfinite(xu);
        double lam = lo && up ? rc : lo ? std::max(rc, 0.0) : up ? std::min(rc, 0.0) : 0.0;
        rd += (rc - lam) * (rc - lam);
        cn += c[j] * c[j];
        pobj += c[j] * x[j];
        dobj += lam > 0 ? lam * xl : lam < 0 ? lam * xu : 0.0;
    }
    s.rp = std::sqrt(rp) / (1 + std::sqrt(bn)); s.rd = std::sqrt(rd) / (1 + std::sqrt(cn));
    s.pobj = pobj; s.dobj = dobj;

    //return whether it satisfies conditions
    return s.rp <= eps && s.rd <= eps &&
           std::fabs(pobj - dobj) <= eps * (1 + std::fabs(pobj) + std::fabs(dobj));
}

//switches to turn parts off for testing
struct Opts {
    double gamma = 1.0;  // reflection, 0.5 = plain Halpern PDHG
    bool halpern = true, restart = true, pid = true, geo = true, ruiz = true, pc = true, bound_obj = true;
    int freq = 200;      // termination-restart check period
    bool compact = true;   //pack finished LPs out of the batch
};

// spmm<w> for a runtime width w <= W (the batch shrinks under compaction)
template <int W>
void spmm_w(int w, const CSR& A, const double* X, double* Y) {
    if constexpr (W > 1) if (w < W) return spmm_w<W - 1>(w, A, X, Y);
    spmm<W>(A, X, Y);
}

// Geometric mean scaling => gi = (1/sqrt(lo*hi)) - 12 iterations
inline void geo_mean(LP& s, std::vector<double>& R, std::vector<double>& S, int iters = 12) {
    std::vector<double> gr(s.A.m, 1.0), gc(s.A.n, 1.0);
    auto f = [](const CSR& M, const std::vector<double>& other, std::vector<double>& g) {
        for (int i = 0; i < M.m; i++) {
            double lo = INFINITY, hi = 0;
            for (int k = M.row_ptr[i]; k < M.row_ptr[i + 1]; k++) {
                double v = std::fabs(M.val[k]) * other[M.col_idx[k]];
                if (!(v > 0 && std::isfinite(v))) continue;
                lo = std::min(lo, v); hi = std::max(hi, v);
            }
            if (hi > 0) g[i] = std::min(std::max(1 / (std::sqrt(lo) * std::sqrt(hi)), 1e-20), 1e20);
        }
    };
    for (int t = 0; t < iters; t++) { f(s.A, gc, gr); f(s.At, gr, gc); }
    for (int i = 0; i < s.A.m; i++) {
        R[i] /= gr[i];
        for (int k = s.A.row_ptr[i]; k < s.A.row_ptr[i + 1]; k++) s.A.val[k] *= gr[i] * gc[s.A.col_idx[k]];
    }
    for (int j = 0; j < s.A.n; j++) {
        S[j] /= gc[j];
        for (int k = s.At.row_ptr[j]; k < s.At.row_ptr[j + 1]; k++) s.At.val[k] *= gc[j] * gr[s.At.col_idx[k]];
    }
}

//  Ruiz and Pock-Chambolle
inline void rescale(LP& s, std::vector<double>& R, std::vector<double>& S, bool l1) {
    auto f = [l1](const CSR& M, std::vector<double>& d) {
        for (int i = 0; i < M.m; i++) {
            double v = 0;
            for (int k = M.row_ptr[i]; k < M.row_ptr[i + 1]; k++)
                v = l1 ? v + std::fabs(M.val[k]) : std::max(v, std::fabs(M.val[k]));
            d[i] = v < 1e-12 ? 1.0 : std::sqrt(v);
        }
    };
    std::vector<double> r(s.A.m), q(s.A.n);
    f(s.A, r); f(s.At, q);
    for (int i = 0; i < s.A.m; i++) {
        R[i] *= r[i];
        for (int k = s.A.row_ptr[i]; k < s.A.row_ptr[i + 1]; k++) s.A.val[k] /= r[i] * q[s.A.col_idx[k]];
    }
    for (int j = 0; j < s.A.n; j++) {
        S[j] *= q[j];
        for (int k = s.At.row_ptr[j]; k < s.At.row_ptr[j + 1]; k++) s.At.val[k] /= q[j] * r[s.At.col_idx[k]];
    }
}

// ||b||: finite constraint bounds, equality counted once
inline double bound_norm(const std::vector<double>& l, const std::vector<double>& u) {
    double b = 0;
    for (size_t i = 0; i < l.size(); i++) {
        if (std::isfinite(l[i]) && l[i] != u[i]) b += l[i] * l[i];
        if (std::isfinite(u[i])) b += u[i] * u[i];
    }
    return std::sqrt(b);
}

//setup,scalings
template <int N>
std::vector<Sol> pdhg_batch(const LP& lp, const std::vector<double>& C, double eps = 1e-4,
                            int max_iter = 1000000, Opts o = {}, const Branch* br = nullptr,
                            const Sol* warm = nullptr) {
    const int m = lp.A.m, n = lp.A.n;
    LP s = lp;
    std::vector<double> R(m, 1.0), S(n, 1.0);
    if (o.geo) geo_mean(s, R, S);
    if (o.ruiz) for (int t = 0; t < 10; t++) rescale(s, R, S, false);
    if (o.pc) rescale(s, R, S, true);
    //update bounds as well because A is scaled
    for (int i = 0; i < m; i++) { s.l[i] /= R[i]; s.u[i] /= R[i]; }
    for (int j = 0; j < n; j++) { s.xl[j] *= S[j]; s.xu[j] *= S[j]; }
    const double tb = o.bound_obj ? 1 / (1 + bound_norm(s.l, s.u)) : 1.0;
    for (int i = 0; i < m; i++) { s.l[i] *= tb; s.u[i] *= tb; }
    for (int j = 0; j < n; j++) { s.xl[j] *= tb; s.xu[j] *= tb; }

    int bj[N]; double blo[N] = {}, bhi[N] = {};
    std::vector<char> sp(n, 0);  // sp[j]: some column overrides variable j
    for (int l = 0; l < N; l++) {
        bj[l] = br ? br[l].j : -1;
        if (bj[l] < 0) continue;
        sp[bj[l]] = 1;
        blo[l] = br[l].lo * S[bj[l]] * tb; bhi[l] = br[l].hi * S[bj[l]] * tb;
    }
    auto lo = [&](int j, int l) { return sp[j] && bj[l] == j ? blo[l] : s.xl[j]; };
    auto hi = [&](int j, int l) { return sp[j] && bj[l] == j ? bhi[l] : s.xu[j]; };

    // per LP: objective scale theta_c, primal weight w, PID state
    double tc[N], w[N], bw[N], esum[N] = {}, fpe[N] = {}, fpe0[N] = {}, fpe_last[N];
    std::vector<double> Cs((long)n * N);
    for (int l = 0; l < N; l++) {
        double cs = 0, co = 0;
        for (int j = 0; j < n; j++) {
            long q = (long)j * N + l;

            //scaled Cs
            Cs[q] = C[q] / S[j];
            cs += Cs[q] * Cs[q]; co += C[q] * C[q];
        }
        tc[l] = o.bound_obj ? 1 / (1 + std::sqrt(cs)) : 1.0;
        w[l] = o.bound_obj ? 1.0 : (1 + std::sqrt(co)) / (1 + bound_norm(lp.l, lp.u));
        bw[l] = w[l]; fpe_last[l] = INFINITY;
        for (int j = 0; j < n; j++) Cs[(long)j * N + l] *= tc[l];
    }
    const double eta = 0.998 / op_norm(s);

    std::vector<double> X((long)n * N), X0, Xh((long)n * N), Xb((long)n * N), AtY((long)n * N), AtYh((long)n * N);
    std::vector<double> Y((long)m * N), Y0, Yh((long)m * N), Yb((long)m * N), AX((long)m * N);
    for (int j = 0; j < n; j++)
        for (int l = 0; l < N; l++)
            X[(long)j * N + l] = std::min(std::max(warm ? warm->x[j] * S[j] * tb : 0.0, lo(j, l)), hi(j, l));
    if (warm)
        {for (int i = 0; i < m; i++){
            for (int l = 0; l < N; l++) Y[(long)i * N + l] = warm->y[i] * R[i] * tc[l];
    }}
    X0 = X; Y0 = Y;
    std::vector<Sol> sol(N);
    bool done[N] = {}, fpe0_due[N] = {}, rs[N];
    int left = N, k[N] = {}, due = 0;  // k: iterations since this LP's last restart
    int W = N, id[N];                   // W: current batch width, id: column -> LP
    for (int l = 0; l < N; l++) id[l] = l;
    std::vector<double> x(n), y(m), Ax(m), r(n), dres(n), c(n);



    for (int it = 0; it < max_iter && left; it++) {
        //check if kontrol iteration
        const bool chk = (it + 1) % o.freq == 0, keep = chk || due;
        const double g = o.gamma;
        //tau = primal step size, sigma = dual step size
        double tau[N], sig[N], hw[N];
        for (int l = 0; l < W; l++) {
            tau[l] = eta / w[l]; 
            sig[l] = eta * w[l];

            //halpern weight
            hw[l] = o.halpern ? (k[l] + 1.0) / (k[l] + 2.0) : 1.0;
        }
        //get AtY with active kernel  
        spmm_w<N>(W, s.At, Y.data(), AtY.data());
        #pragma omp parallel for
        for (int j = 0; j < n; j++)
            for (int l = 0; l < W; l++) {
                long q = (long)j * W + l;
                if (done[l]) continue;
                
                //clampimg
                double xh = std::min(std::max(X[q] - tau[l] * (Cs[q] - AtY[q]), lo(j, l)), hi(j, l));
                //reflection
                double xb = 2 * xh - X[q];
                Xb[q] = xb;
                if (keep) Xh[q] = xh;

                //update X with halpern anchor
                X[q] = hw[l] * (g * xb + (1 - g) * X[q]) + (1 - hw[l]) * X0[q];
            }
        
        spmm_w<N>(W, s.A, Xb.data(), AX.data());
        #pragma omp parallel for
        for (int i = 0; i < m; i++)
            for (int l = 0; l < W; l++) {
                long q = (long)i * W + l;
                if (done[l]) continue;
                double yh = dual_step(Y[q], AX[q], s.l[i], s.u[i], sig[l]);
                //dual reflection
                double yb = 2 * yh - Y[q];
                if (keep) { Yh[q] = yh; Yb[q] = yb; }
                Y[q] = hw[l] * (g * yb + (1 - g) * Y[q]) + (1 - hw[l]) * Y0[q];
            }
        for (int l = 0; l < W; l++) k[l]++;
        if (!keep) continue;

        // fixed-point error(FPE) ||z - T(z)||_P (times eta), z = current iterate, T(z) = PDHG step
        spmm_w<N>(W, s.At, Yh.data(), AtYh.data());
        for (int l = 0; l < W; l++) {
            if (done[l]) continue;
            double dx2 = 0, dy2 = 0, cr = 0;
            for (int j = 0; j < n; j++) {
                long q = (long)j * W + l;
                double d = Xb[q] - Xh[q];
                dx2 += d * d; cr += d * (AtYh[q] - AtY[q]);
            }
            for (int i = 0; i < m; i++) {
                long q = (long)i * W + l;
                dy2 += (Yb[q] - Yh[q]) * (Yb[q] - Yh[q]);
            }
            fpe[l] = std::sqrt(w[l] * dx2 + dy2 / w[l] + 2 * eta * cr);
            if (fpe0_due[l]) { fpe0[l] = fpe[l]; fpe0_due[l] = false; due--; }
        }
        if (!chk) continue;


        //kontrol iteration 
        spmm_w<N>(W, s.A, Xh.data(), AX.data());
        for (int l = 0; l < W; l++) {
            if (done[l]) continue;
            for (int j = 0; j < n; j++) {
                long q = (long)j * W + l;
                double d = (Xb[q] - Xh[q]) / tau[l], u = S[j] / tc[l];  // d = (x^ - x) / tau
                //UNDO scalings here
                x[j] = Xh[q] / (S[j] * tb); c[j] = C[(long)j * N + id[l]];
                r[j] = (d + Cs[q] - AtY[q]) * u;
                dres[j] = -(d + AtYh[q] - AtY[q]) * u;
            }
            for (int i = 0; i < m; i++) {
                long q = (long)i * W + l;
                y[i] = Yh[q] / (R[i] * tc[l]); Ax[i] = AX[q] * R[i] / tb;
            }
            Sol& so = sol[id[l]];
            so.iters = it + 1; so.w = w[l];
            if (kkt(lp, c.data(), x, y, Ax, r, dres, eps, so, br ? br[id[l]] : Branch{})) {
                so.ok = true; so.x = x; so.y = y;
                done[l] = true; left--;
            }
        }

        //compaction
        if (N > 1 && o.compact && left && left < W) {
            // move unfinished LPs to columns 0..left-1 (in place: targets never pass sources)
            // Per-LP arithmetic is unchanged, so every LP's result stays the same.
            int map[N];
            for (int l = 0, p = 0; l < W; l++) if (!done[l]) map[p++] = l;
            auto pv = [&](std::vector<double>& V, long rows) {
                for (long r = 0; r < rows; r++)
                    for (int p = 0; p < left; p++) V[r * left + p] = V[r * W + map[p]];
            };
            for (auto* V : {&Cs, &X, &X0, &Xh, &Xb, &AtY, &AtYh}) pv(*V, n);
            for (auto* V : {&Y, &Y0, &Yh, &Yb, &AX}) pv(*V, m);

            //also squeeeze vectors
            auto ps = [&](auto* a) { for (int p = 0; p < left; p++) a[p] = a[map[p]]; };
            ps(tc); ps(w); ps(bw); ps(esum); ps(fpe); ps(fpe0); ps(fpe_last); ps(k); ps(fpe0_due); ps(id);
            ps(bj); ps(blo); ps(bhi);
            //fill done back
            std::fill(done, done + left, false);
            W = left;
        }
        if (!left || !o.restart) continue;

        // restart decision per LP
        int nrs = 0;
        for (int l = 0; l < W; l++) {
            double F = fpe[l], F0 = fpe0[l], FL = fpe_last[l];
            fpe_last[l] = F;
            rs[l] = !done[l] && (it + 1 == o.freq || F <= 0.2 * F0 || (F <= 0.5 * F0 && F > FL) ||
                                 k[l] >= 0.36 * (it + 1));
            if (!rs[l]) continue;
            nrs++;
            double pd = 0, dd = 0;
            for (int j = 0; j < n; j++) {
                long q = (long)j * W + l;
                pd += (Xh[q] - X0[q]) * (Xh[q] - X0[q]);
            }
            for (int i = 0; i < m; i++) {
                long q = (long)i * W + l;
                dd += (Yh[q] - Y0[q]) * (Yh[q] - Y0[q]);
            }
            // how far we have come since last restart
            pd = std::sqrt(pd); dd = std::sqrt(dd);
            if (o.pid) {
                //update w to balance dual and primal
                //this is better than rd/rp in cases where variables are bounded by two sides
                if (pd > 1e-16 && dd > 1e-16 && pd < 1e12 && dd < 1e12) {
                    double e = std::log(dd) - std::log(pd) - std::log(w[l]);
                    esum[l] = 0.3 * esum[l] + e;
                    w[l] *= std::exp(0.99 * e + 0.01 * esum[l]);
                } else { w[l] = bw[l]; esum[l] = 0; }
            }
            fpe_last[l] = INFINITY; k[l] = 0; fpe0_due[l] = true; due++;
        }
        if (!nrs) continue;
        // new anchor = PDHG point T(z)
        for (long q = 0; q < (long)n * W; q++) if (rs[q % W]) X0[q] = X[q] = Xh[q];
        for (long q = 0; q < (long)m * W; q++) if (rs[q % W]) Y0[q] = Y[q] = Yh[q];
    }
    return sol;
}

inline Sol pdhg(const LP& lp, double eps = 1e-4, int max_iter = 1000000, Opts o = {}, const Sol* warm = nullptr) {
    return pdhg_batch<1>(lp, lp.c, eps, max_iter, o, nullptr, warm)[0];
}

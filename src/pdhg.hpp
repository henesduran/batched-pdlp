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

struct Sol { std::vector<double> x, y; double pobj, dobj, rp, rd; int iters; bool ok; };


inline double dual_step(double y0, double a, double l, double u, double sigma) {
    double up = y0 + sigma * (l - a), dn = y0 + sigma * (u - a);
    return up > 0 ? up : dn < 0 ? dn : 0.0;
}

inline bool kkt(const LP& lp, const double* c, const std::vector<double>& x, const std::vector<double>& y,
                const std::vector<double>& Ax, const std::vector<double>& Aty, double eps, Sol& s) {
    double rp = 0, bn = 0, rd = 0, cn = 0, pobj = lp.obj_off, dobj = lp.obj_off;
    for (int i = 0; i < lp.A.m; i++) {
        double l = lp.l[i], u = lp.u[i];
        double v = Ax[i] < l ? l - Ax[i] : Ax[i] > u ? Ax[i] - u : 0.0;
        rp += v * v;
        if (std::isfinite(l)) bn += l * l;
        if (std::isfinite(u) && u != l) bn += u * u;
        dobj += y[i] > 0 ? y[i] * l : y[i] * u;
    }
    for (int j = 0; j < lp.A.n; j++) {
        double lam = c[j] - Aty[j];
        
        if (lam > 0 && std::isfinite(lp.xl[j])) dobj += lam * lp.xl[j];
        else if (lam < 0 && std::isfinite(lp.xu[j])) dobj += lam * lp.xu[j];
        else rd += lam * lam;
        cn += c[j] * c[j];
        pobj += c[j] * x[j];
    }
    s.rp = std::sqrt(rp); s.rd = std::sqrt(rd); s.pobj = pobj; s.dobj = dobj;
    return s.rp <= eps * (1 + std::sqrt(bn)) && s.rd <= eps * (1 + std::sqrt(cn)) &&
           std::fabs(pobj - dobj) <= eps * (1 + std::fabs(pobj) + std::fabs(dobj));
}


inline Sol pdhg(const LP& lp, double eps = 1e-4, int max_iter = 1000000, int check = 64,
                double eta = 0.0) {
    const int m = lp.A.m, n = lp.A.n;
    if (eta == 0.0) eta = 0.998 / op_norm(lp);
    std::vector<double> x(n), xn(n), Ax(m), Aty(n), y(m);
    for (int j = 0; j < n; j++) x[j] = std::min(std::max(0.0, lp.xl[j]), lp.xu[j]);
    Sol s{};
    for (int it = 0; it < max_iter; it++) {
        spmm<1>(lp.At, y.data(), Aty.data());
        if (it % check == 0) {
            spmm<1>(lp.A, x.data(), Ax.data());
            if (kkt(lp, lp.c.data(), x, y, Ax, Aty, eps, s)) { s.iters = it; s.ok = true; break; }
        }
        #pragma omp parallel for
        for (int j = 0; j < n; j++)
            xn[j] = std::min(std::max(x[j] - eta * (lp.c[j] - Aty[j]), lp.xl[j]), lp.xu[j]);
        #pragma omp parallel for
        for (int j = 0; j < n; j++) x[j] = 2 * xn[j] - x[j];  // extrapolated point, reused as buffer
        spmm<1>(lp.A, x.data(), Ax.data());
        #pragma omp parallel for
        for (int i = 0; i < m; i++) y[i] = dual_step(y[i], Ax[i], lp.l[i], lp.u[i], eta);
        x.swap(xn);
        s.iters = it + 1;
    }
    s.x = x; s.y = y;
    return s;
}


template <int N>
std::vector<Sol> pdhg_batch(const LP& lp, const std::vector<double>& C, double eps = 1e-4,
                            int max_iter = 1000000, int check = 64, double eta = 0.0) {
    const int m = lp.A.m, n = lp.A.n;
    if (eta == 0.0) eta = 0.998 / op_norm(lp);
    std::vector<double> X((long)n * N), Xn((long)n * N), AX((long)m * N), AtY((long)n * N), Y((long)m * N);
    for (int j = 0; j < n; j++)
        for (int l = 0; l < N; l++) X[(long)j * N + l] = std::min(std::max(0.0, lp.xl[j]), lp.xu[j]);
    std::vector<Sol> s(N);
    bool done[N] = {};
    int left = N;
    std::vector<double> x(n), y(m), Ax(m), Aty(n), c(n);
    auto col = [](const std::vector<double>& M, int l, std::vector<double>& v) {
        for (size_t k = 0; k < v.size(); k++) v[k] = M[k * N + l];
    };
    for (int it = 0; it < max_iter && left; it++) {
        spmm<N>(lp.At, Y.data(), AtY.data());
        if (it % check == 0) {
            spmm<N>(lp.A, X.data(), AX.data());
            for (int l = 0; l < N; l++) {
                if (done[l]) continue;
                col(X, l, x); col(Y, l, y); col(AX, l, Ax); col(AtY, l, Aty); col(C, l, c);
                if (kkt(lp, c.data(), x, y, Ax, Aty, eps, s[l])) {
                    s[l].iters = it; s[l].ok = true; s[l].x = x; s[l].y = y;
                    done[l] = true; left--;
                }
            }
        }
        #pragma omp parallel for
        for (int j = 0; j < n; j++)
            for (int l = 0; l < N; l++) {
                long k = (long)j * N + l;
                Xn[k] = done[l] ? X[k]
                      : std::min(std::max(X[k] - eta * (C[k] - AtY[k]), lp.xl[j]), lp.xu[j]);
                X[k] = 2 * Xn[k] - X[k];
            }
        spmm<N>(lp.A, X.data(), AX.data());
        #pragma omp parallel for
        for (int i = 0; i < m; i++)
            for (int l = 0; l < N; l++) {
                long k = (long)i * N + l;
                if (!done[l]) Y[k] = dual_step(Y[k], AX[k], lp.l[i], lp.u[i], eta);
            }
        X.swap(Xn);
        for (int l = 0; l < N; l++) if (!done[l]) s[l].iters = it + 1;
    }
    return s;
}

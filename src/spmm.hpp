#pragma once
#include <omp.h>
#include <algorithm>
#include "csr.hpp"

// nnz division
inline int row_begin(const CSR& A, int t, int T) {
    long target = A.nnz() * t / T;
    return int(std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), target) - A.row_ptr.begin());
}

inline void spmv(const CSR& A, const double* x, double* y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A.m; i++) {
        double sum = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++)
            sum += A.val[k] * x[A.col_idx[k]];
        y[i] = sum;
    }
}

// X = n x N, Y = m x N, both rowmajor: X[j*N + l] is LP l's value of x_j.
// template? ==> vectorization?
template <int N>
void spmm(const CSR& A, const double* X, double* Y) {
    #pragma omp parallel
    {
    int T = omp_get_num_threads(), t = omp_get_thread_num();

    int e = t == T - 1 ? A.m : row_begin(A, t + 1, T);
    for (int i = row_begin(A, t, T); i < e; i++) {
        double acc[N] = {};
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            const double a = A.val[k];
            const double* xj = X + (long)A.col_idx[k] * N;

            //N is compile time,this for loop turns to FMA 
            for (int l = 0; l < N; l++) acc[l] += a * xj[l];
        }
        for (int l = 0; l < N; l++) Y[(long)i * N + l] = acc[l];
    }
    }
}


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


/*
// hand written avx512 version for N = 8, kept here just for reference
// one zmm holds 8 doubles, so one load brings x_j for all 8 LPs , then one FMA does all of them
// i think the compiler already vectorizes the template version above in a similar way, so this is probably not needed,
// and the template one is more portable actually.
void spmm_8_avx512(const CSR& A, const double* X, double* Y) {
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num();
        int e = (t == T - 1) ? A.m : row_begin(A, t + 1, T);

        for (int i = row_begin(A, t, T); i < e; i++) {
            __m512d acc = _mm512_setzero_pd();

            for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
                // zmm1 = [a, a, a, a, a, a, a, a]
                __m512d a = _mm512_set1_pd(A.val[k]);

                // zmm2 = [x0, x1, x2, x3, x4, x5, x6, x7]
                __m512d xj = _mm512_loadu_pd(X + (long)A.col_idx[k] * 8);

                // FMA
                // acc = acc + (a * xj)
                acc = _mm512_fmadd_pd(a, xj, acc);
            }
            _mm512_storeu_pd(Y + (long)i * 8, acc);
        }
    }
}
*/

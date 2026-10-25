#pragma once
#include <vector>

struct CSR {
    int m = 0, n = 0;
    std::vector<int> row_ptr, col_idx;
    std::vector<double> val;
    long nnz() const { return (long)val.size(); }
};


inline CSR transpose(const CSR& A) {
    CSR T;
    T.m = A.n; T.n = A.m;
    T.row_ptr.assign(A.n + 1, 0);
    T.col_idx.resize(A.nnz());
    T.val.resize(A.nnz());
    for (int j : A.col_idx) T.row_ptr[j + 1]++;
    for (int j = 0; j < A.n; j++) T.row_ptr[j + 1] += T.row_ptr[j];
    std::vector<int> pos(T.row_ptr.begin(), T.row_ptr.end() - 1);
    for (int i = 0; i < A.m; i++)
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            int p = pos[A.col_idx[k]]++;
            T.col_idx[p] = i;
            T.val[p] = A.val[k];
        }
    return T;
}

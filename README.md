# batched-pdlp

CPU batched PDHG for N LPs that share the same constraint matrix A (only c, bounds differ).

- `src/spmm.hpp`: CSR SpMM. X and Y are row-major with N columns, so for N=8 one row is one cache line. Rows are split across threads by nnz.
- `src/pdhg.hpp`: constant-step PDHG (eta = 0.998/||A||_2), single (`pdhg`) and batched (`pdhg_batch<N>`). Each LP stops on its own KKT check.
- `src/mps.cpp`: MPS reader. COLUMNS fill A^T directly, A is its transpose.

Build: `make` (g++ -O3 -march=native -fopenmp).

| binary | |
|---|---|
| `spmm_check file.mps` | spmm<N> vs spmv, N = 1..64 |
| `spmm_bench file.mps [reps]` | SpMM time per LP, N = 1..64 |
| `pdhg_batch file.mps [delta] [max_iter]` | batch of 8 vs 8 single solves, bitwise comparison |
| `pdhg_bench file.mps [iters] [nocopies]` | time per iteration per LP: single, batch N=8, parallel copies |

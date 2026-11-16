# batched-pdlp

PDHG on the CPU for N linear programs that share the same constraint matrix A and differ only in
their objective vectors. All N LPs are solved together: each iteration multiplies A by an n x N block
(SpMM), so A is read once per iteration for all N instead of once per LP.

The LP form is min c'x subject to l <= Ax <= u, xl <= x <= xu. The method follows cuPDLPx:
reflected Halpern PDHG with adaptive restarts, PID update of the primal weight, and diagonal
scaling (geometric mean, Ruiz, Pock-Chambolle). Each LP has its own step sizes, restarts and
termination; finished LPs are removed from the batch.

## Build

```
mkdir -p build
g++ -O3 -march=native -fopenmp -std=c++17 -Isrc bench/solve.cpp src/mps.cpp -o build/solve
```

## Run

```
OMP_NUM_THREADS=4 build/solve many afiro.mps 0 4 4
```

Solves 4 copies of afiro (delta 0 means the same objective) as one batch of 4. The output is one CSV
line per LP (status, iterations, primal and dual objective), then the total time.

With delta > 0, LP 0 keeps c and LP l gets c_j (1 + delta u), u uniform in [-1, 1].

## Modes

- `many file.mps delta nlp chunk [key=val ...]`: solves nlp LPs in batches of chunk. chunk 1 solves them one by one.
- `single file.mps [key=val ...]`: solves one LP.
- `batch file.mps delta`: solves 8 LPs as one batch, then each alone, and checks that the results are bitwise equal. Build with `-ffp-contract=off` for this.
- `kkt file.mps delta nlp eps sol[:lp] ...`: checks solution files against the KKT conditions of the unscaled LP.
- `mps file.mps delta lp out.mps [nlp]`: writes LP number lp as an MPS file.
- `sb file.mps nb`: experimental, strong branching children of a MIP root as batches.

Keys: `eps` (default 1e-4, relative), `maxit`, `freq` (check period, default 200). `many` also takes
`sol=dir` (write the solutions) and `par=1` (run the batches side by side, one thread each).

## Layout

```
bench/solve.cpp   command-line driver
src/pdhg.hpp      batched PDHG: scaling, iteration, restarts, KKT check
src/spmm.hpp      sparse matrix times a vector (SpMV) or an n x N block (SpMM)
src/csr.hpp       CSR matrix and transpose
src/mps.hpp/.cpp  MPS reader, LP struct
```

## Limitations

- Minimization only. OBJSENSE MAX is not read, so a max problem is silently solved as min.
- No infeasibility or unboundedness detection. Such LPs run until maxit.
- No presolve. Plain .mps only, no .gz.
- Needs g++ with OpenMP (Linux or WSL).

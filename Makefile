CXX      = g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra
BUILD    = build

all: $(BUILD)/mps_info $(BUILD)/spmm_check $(BUILD)/spmm_bench $(BUILD)/pdhg_solve $(BUILD)/pdhg_batch $(BUILD)/pdhg_bench $(BUILD)/spmm_bench_novec

$(BUILD)/mps_info: src/mps_info.cpp src/mps.cpp src/mps.hpp src/csr.hpp src/spmm.hpp src/pdhg.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) src/mps_info.cpp src/mps.cpp -o $@

$(BUILD)/spmm_check: src/spmm_check.cpp src/mps.cpp src/mps.hpp src/csr.hpp src/spmm.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) src/spmm_check.cpp src/mps.cpp -o $@

$(BUILD)/spmm_bench: src/spmm_bench.cpp src/mps.cpp src/mps.hpp src/csr.hpp src/spmm.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) src/spmm_bench.cpp src/mps.cpp -o $@

# auto vectorization off
$(BUILD)/spmm_bench_novec: src/spmm_bench.cpp src/mps.cpp src/mps.hpp src/csr.hpp src/spmm.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize -fno-tree-slp-vectorize src/spmm_bench.cpp src/mps.cpp -o $@

$(BUILD)/pdhg_solve: src/pdhg_solve.cpp src/mps.cpp src/mps.hpp src/csr.hpp src/spmm.hpp src/pdhg.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) src/pdhg_solve.cpp src/mps.cpp -o $@

$(BUILD)/pdhg_bench: src/pdhg_bench.cpp src/mps.cpp src/mps.hpp src/csr.hpp src/spmm.hpp src/pdhg.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) src/pdhg_bench.cpp src/mps.cpp -o $@

# No FMA 
$(BUILD)/pdhg_batch: src/pdhg_batch.cpp src/mps.cpp src/mps.hpp src/csr.hpp src/spmm.hpp src/pdhg.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -ffp-contract=off src/pdhg_batch.cpp src/mps.cpp -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean

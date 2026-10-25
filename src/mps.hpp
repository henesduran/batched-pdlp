#pragma once
#include <string>
#include <vector>
#include "csr.hpp"


struct LP {
    CSR A, At;
    std::vector<double> c, l, u, xl, xu;
    double obj_off = 0.0;
};

LP read_mps(const std::string& path);

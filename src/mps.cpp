#include "mps.hpp"
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

static const double INF = std::numeric_limits<double>::infinity();

LP read_mps(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);

    LP lp;
    CSR& At = lp.At;
    std::unordered_map<std::string, int> row_id, col_id;
    std::vector<char> type;
    std::vector<double> rhs, rng;
    std::vector<bool> has_rng;
    std::string obj_row, sec, line, cur_col;
    int m = 0;

    auto row = [&](const std::string& name) {
        auto it = row_id.find(name);
        if (it == row_id.end()) throw std::runtime_error("unknown row " + name);
        return it->second;
    };

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '*') continue;
        std::istringstream ss(line);
        if (!isspace((unsigned char)line[0])) { ss >> sec; continue; }

        if (sec == "ROWS") {
            std::string t, name;
            ss >> t >> name;
            if (t != "N") { row_id[name] = m++; type.push_back(t[0]); }
            else if (obj_row.empty()) obj_row = name;
            else row_id[name] = -1;
        } else if (sec == "COLUMNS") {
            if (line.find("'MARKER'") != std::string::npos) continue;
            std::string col, r;
            double v;
            ss >> col;
            if (col != cur_col) {
                cur_col = col;
                col_id[col] = (int)lp.c.size();
                At.row_ptr.push_back((int)At.val.size());
                lp.c.push_back(0.0);
            }
            while (ss >> r >> v) {
                if (r == obj_row) { lp.c.back() = v; continue; }
                int i = row(r);
                if (i < 0 || v == 0.0) continue;
                At.col_idx.push_back(i);
                At.val.push_back(v);
            }
        } else if (sec == "RHS" || sec == "RANGES") {
            if (rhs.empty()) { rhs.assign(m, 0.0); rng.assign(m, 0.0); has_rng.assign(m, false); }
            std::string name, r;
            double v;
            ss >> name;
            while (ss >> r >> v) {
                if (r == obj_row) { if (sec == "RHS") lp.obj_off = -v; continue; }
                int i = row(r);
                if (i < 0) continue;
                if (sec == "RHS") rhs[i] = v;
                else { rng[i] = v; has_rng[i] = true; }
            }
        } else if (sec == "BOUNDS") {
            if (lp.xl.empty()) { lp.xl.assign(lp.c.size(), 0.0); lp.xu.assign(lp.c.size(), INF); }
            std::string t, name, col;
            double v = 0.0;
            ss >> t >> name >> col >> v;
            auto it = col_id.find(col);
            if (it == col_id.end()) throw std::runtime_error("unknown column " + col);
            int j = it->second;
            if (t == "UP" || t == "UI") lp.xu[j] = v;
            else if (t == "LO" || t == "LI") lp.xl[j] = v;
            else if (t == "FX") lp.xl[j] = lp.xu[j] = v;
            else if (t == "FR") { lp.xl[j] = -INF; lp.xu[j] = INF; }
            else if (t == "MI") lp.xl[j] = -INF;
            else if (t == "PL") lp.xu[j] = INF;
            else if (t == "BV") { lp.xl[j] = 0.0; lp.xu[j] = 1.0; }
            else throw std::runtime_error("unknown bound type " + t);
        }
    }
    At.row_ptr.push_back((int)At.val.size());
    At.m = (int)lp.c.size();
    At.n = m;
    lp.A = transpose(At);
    
    lp.At = transpose(lp.A);

    if (rhs.empty()) { rhs.assign(m, 0.0); rng.assign(m, 0.0); has_rng.assign(m, false); }
    if (lp.xl.empty()) { lp.xl.assign(lp.c.size(), 0.0); lp.xu.assign(lp.c.size(), INF); }
    lp.l.resize(m);
    lp.u.resize(m);
    for (int i = 0; i < m; i++) {
        double b = rhs[i], r = rng[i];
        switch (type[i]) {
            case 'E':
                lp.l[i] = b; lp.u[i] = b;
                if (has_rng[i]) { if (r > 0) lp.u[i] = b + r; else lp.l[i] = b + r; }
                break;
            case 'L': lp.l[i] = has_rng[i] ? b - std::fabs(r) : -INF; lp.u[i] = b; break;
            case 'G': lp.l[i] = b; lp.u[i] = has_rng[i] ? b + std::fabs(r) : INF; break;
            default: throw std::runtime_error(std::string("unknown row type ") + type[i]);
        }
    }
    return lp;
}

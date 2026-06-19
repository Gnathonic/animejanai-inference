// src/rife_cpu.h — pure-CPU RIFE math, no HIP/MIGraphX deps (unit-testable).
#pragma once
#include <string>
#include <cstdint>
#include <cstddef>

namespace rife_cpu {

inline std::string model_name(int code, bool ensemble) {
    std::string s = std::to_string(code);
    if (s.size() < 2) return "";                                   // invalid
    std::string dec = (s.size() == 2) ? s.substr(1, 1) : s.substr(1, 2);
    std::string name = "rife_v" + s.substr(0, 1) + "." + dec;      // substr, NOT s[0] (UB)
    if (s.size() == 4 && s.back() == '1') name += "_lite";
    if (ensemble)                         name += "_ensemble";
    return name;
}

struct Geom { int w, h, pw, ph, pad_l, pad_t; };
inline Geom geometry(int w, int h) {
    Geom g; g.w = w; g.h = h;
    g.pw = (w + 63) / 64 * 64;
    g.ph = (h + 63) / 64 * 64;
    g.pad_l = ((g.pw - w) / 2) & ~1;           // centered, rounded down to even
    g.pad_t = ((g.ph - h) / 2) & ~1;
    return g;
}

// (functions added in A2, A3, A6, A7)
}  // namespace rife_cpu

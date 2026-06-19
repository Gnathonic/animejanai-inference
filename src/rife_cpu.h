// src/rife_cpu.h — pure-CPU RIFE math, no HIP/MIGraphX deps (unit-testable).
#pragma once
#include <string>
#include <cstdint>
#include <cstddef>
#include <cmath>

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

inline bool scene_detect(const uint8_t* ya, ptrdiff_t sa, const uint8_t* yb, ptrdiff_t sb,
                         int w, int h, int pw, int ph, double norm, double threshold) {
    double sum = 0.0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* ra = ya + (ptrdiff_t)y * sa;
        const uint8_t* rb = yb + (ptrdiff_t)y * sb;
        for (int x = 0; x < w; ++x) sum += std::abs((int)ra[x] - (int)rb[x]) * norm;
    }
    return sum / ((double)pw * ph) > threshold;
}

inline bool scene_detect(const uint16_t* ya, ptrdiff_t sa, const uint16_t* yb, ptrdiff_t sb,
                         int w, int h, int pw, int ph, double norm, double threshold) {
    double sum = 0.0;
    for (int y = 0; y < h; ++y) {
        const uint16_t* ra = (const uint16_t*)((const uint8_t*)ya + (ptrdiff_t)y * sa);
        const uint16_t* rb = (const uint16_t*)((const uint8_t*)yb + (ptrdiff_t)y * sb);
        for (int x = 0; x < w; ++x) sum += std::abs((int)ra[x] - (int)rb[x]) * norm;
    }
    return sum / ((double)pw * ph) > threshold;
}

// (functions added in A2, A3, A4, A6, A7)
}  // namespace rife_cpu

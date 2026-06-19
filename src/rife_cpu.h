// src/rife_cpu.h — pure-CPU RIFE math, no HIP/MIGraphX deps (unit-testable).
#pragma once
#include <string>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include "aji.h"   // C ABI types only (aji_frame/aji_format/aji_range); no HIP/MIGraphX

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

// fill_consts — write the four CONSTANT mesh/multiplier planes (channels 7-10)
// of the RIFE v1 input layout into the fp32 staging tensor. Done ONCE at setup;
// only ch0-6 change per frame. Dims are the PADDED pw/ph (vsmlrt get_rife_input
// v1; kernels.cu k_rife_consts). `t` must point at the full 11-channel tensor
// (element offset 7*pw*ph is computed here). Channels 0-6 are left untouched.
//   ch7  meshX = 2*x/(pw-1) - 1
//   ch8  meshY = 2*y/(ph-1) - 1
//   ch9  mulW  = 2/(pw-1)   (constant)
//   ch10 mulH  = 2/(ph-1)   (constant)
inline void fill_consts(float* t, int pw, int ph) {
    const size_t plane = (size_t)pw * ph;
    float* mx = t + 7*plane; float* my = t + 8*plane;
    float* mw = t + 9*plane; float* mh = t + 10*plane;
    const float kw = 2.0f / (pw - 1), kh = 2.0f / (ph - 1);
    for (int y = 0; y < ph; ++y) for (int x = 0; x < pw; ++x) {
        size_t i = (size_t)y * pw + x;
        mx[i] = 2.0f * x / (pw - 1) - 1.0f;
        my[i] = 2.0f * y / (ph - 1) - 1.0f;
        mw[i] = kw;
        mh[i] = kh;
    }
}

// RIFE CPU color (implemented in rife_cpu.cpp; pulls resample.h).
//
// yuv420_to_rgb_planes — BT.709 (forced, regardless of f.matrix) + bilinear
// chroma UPSAMPLE of an NV12/P010 frame into the 11-channel fp32 tensor at
// channels 3*frame_index .. 3*frame_index+2 (planar R,G,B; plane = pw*ph). The
// source w*h window is written CENTERED into the padded pw*ph at (pad_l,pad_t);
// the whole RGB plane is first pre-filled with the BT.709 conversion of studio
// black so GridSample warps near the edge read black, not garbage. `range`
// follows the source (matrix does not).
void yuv420_to_rgb_planes(const aji_frame& f, const Geom& g, float* tensor,
                          int frame_index, aji_range range);

// rgb_planes_to_yuv420 — inverse: crop the centered g.w x g.h window from a
// 3-channel RGB tensor (channels 0..2, planar, plane = pw*ph), BT.709 RGB->YUV
// + bilinear chroma DOWNSAMPLE, quantize, and write into the NV12/P010 `out`
// frame. `range` follows the source.
void rgb_planes_to_yuv420(const float* tensor3, const Geom& g, aji_frame& out,
                          aji_range range);

}  // namespace rife_cpu

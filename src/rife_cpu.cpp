// src/rife_cpu.cpp — RIFE CPU color (BT.709 + bilinear, 11-channel tensor I/O).
//
// Adapted from aji_rocm.cpp's gpu_pre/gpu_post CPU color templates, but for RIFE:
//   1. Filter      = AJI_FILTER_BILINEAR   (not Spline36)
//   2. Matrix      = AJI_MATRIX_BT709 forced (range still follows the source)
//   3. Destination = the 11-channel fp32 tensor (channels 3*frame_index..+2),
//                    centered into the padded pw*ph at (pad_l,pad_t)
//   4. Borders     = BT.709 conversion of studio black (so edge GridSample warps
//                    read black, not garbage)
//
// Pulls resample.h + aji.h (CPU color + ABI types); free of HIP/MIGraphX. Uses
// local std::vector scratch and creates the bilinear plans per call (perf caching
// is a later task). Mirror of aji_dml.cpp:2132-2135 plan creation (BILINEAR), NOT
// gpu_pre which bakes Spline36.
#include "rife_cpu.h"
#include "resample.h"

#include <vector>
#include <cmath>

using aji_resample::weights;

namespace {

// Mirror boundary clamp (zimg semantics), identical to gpu_pre/gpu_post.
inline int mirr(int i, int n) {
    i = i < 0 ? -i - 1 : i;
    i = i >= n ? 2 * n - 1 - i : i;
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

}  // namespace

namespace rife_cpu {

// NV12/P010 frame -> centered RGB window in the 11-ch tensor, BT.709 + bilinear,
// borders pre-filled with BT.709(studio black).
void yuv420_to_rgb_planes(const aji_frame& f, const Geom& g, float* tensor,
                          int frame_index, aji_range range) {
    const int W = g.w, H = g.h;
    const int cw = W >> 1, ch = H >> 1;
    const int format = f.format;
    const size_t plane = (size_t)g.pw * g.ph;

    // BT.709 hardcoded for RIFE color regardless of f.matrix; range follows source.
    aji_csp csp = aji_resample::make_csp(format, AJI_MATRIX_BT709, range);
    const float kg = 1.0f - csp.kr - csp.kb;

    // Studio black in this format/range -> RGB, used to pre-fill the padded plane.
    // NV12 8-bit Y=16,C=128; P010 16-bit Y=16*256,C=128*256. Running the same matrix
    // math on the raw black YUV is range-robust (limited -> (0,0,0); full -> a small
    // grey), and keeps the border consistent with the interior conversion.
    const bool t16 = (format == AJI_FMT_P010);
    const float blkY_raw = t16 ? 16.0f * 256.0f : 16.0f;
    const float blkC_raw = t16 ? 128.0f * 256.0f : 128.0f;
    {
        const float Yb = (blkY_raw - csp.yoff) / csp.yscale;
        const float Ub = (blkC_raw - csp.coff) / csp.cscale;
        const float Vb = (blkC_raw - csp.coff) / csp.cscale;
        const float Rblk = Yb + 2.0f * (1.0f - csp.kr) * Vb;
        const float Bblk = Yb + 2.0f * (1.0f - csp.kb) * Ub;
        const float Gblk = Yb - (2.0f * csp.kb * (1.0f - csp.kb) * Ub +
                                 2.0f * csp.kr * (1.0f - csp.kr) * Vb) / kg;
        float* Rp = tensor + (size_t)(3 * frame_index + 0) * plane;
        float* Gp = tensor + (size_t)(3 * frame_index + 1) * plane;
        float* Bp = tensor + (size_t)(3 * frame_index + 2) * plane;
        for (size_t i = 0; i < plane; i++) { Rp[i] = Rblk; Gp[i] = Gblk; Bp[i] = Bblk; }
    }

    // ---- read host YUV planes into fp32 raw-container values (mirror aji_rocm) ----
    std::vector<float> Yf((size_t)W * H), Uf((size_t)cw * ch), Vf((size_t)cw * ch);
    ptrdiff_t ys  = f.stride[0] ? f.stride[0] : (ptrdiff_t)W * (t16 ? 2 : 1);
    ptrdiff_t uvs = f.stride[1] ? f.stride[1] : (ptrdiff_t)cw * 2 * (t16 ? 2 : 1);
    if (t16) {
        const uint8_t* yb = (const uint8_t*)f.plane[0];
        const uint8_t* uvb = (const uint8_t*)f.plane[1];
        for (int y = 0; y < H; y++) {
            const uint16_t* row = (const uint16_t*)(yb + (ptrdiff_t)y * ys);
            for (int x = 0; x < W; x++) Yf[(size_t)y * W + x] = (float)row[x];
        }
        for (int y = 0; y < ch; y++) {
            const uint16_t* row = (const uint16_t*)(uvb + (ptrdiff_t)y * uvs);
            for (int x = 0; x < cw; x++) { Uf[(size_t)y * cw + x] = (float)row[2*x]; Vf[(size_t)y * cw + x] = (float)row[2*x+1]; }
        }
    } else {
        const uint8_t* yb = (const uint8_t*)f.plane[0];
        const uint8_t* uvb = (const uint8_t*)f.plane[1];
        for (int y = 0; y < H; y++) {
            const uint8_t* row = yb + (ptrdiff_t)y * ys;
            for (int x = 0; x < W; x++) Yf[(size_t)y * W + x] = (float)row[x];
        }
        for (int y = 0; y < ch; y++) {
            const uint8_t* row = uvb + (ptrdiff_t)y * uvs;
            for (int x = 0; x < cw; x++) { Uf[(size_t)y * cw + x] = (float)row[2*x]; Vf[(size_t)y * cw + x] = (float)row[2*x+1]; }
        }
    }

    // ---- bilinear chroma UPSAMPLE (cw x ch) -> (W x H), siting LEFT (NV12/P010 default) ----
    double sx, sy; aji_resample::chroma_shifts(f.siting ? f.siting : AJI_SITING_LEFT, true, &sx, &sy);
    weights ph_w = aji_resample::compute(cw, W, sx, AJI_FILTER_BILINEAR);
    weights pv_w = aji_resample::compute(ch, H, sy, AJI_FILTER_BILINEAR);

    std::vector<float> t0u((size_t)W * ch), t0v((size_t)W * ch);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < W; x++) {
            float u = 0, v = 0; int s0 = ph_w.start[x];
            for (int j = 0; j < ph_w.taps; j++) { float w = ph_w.wt[(size_t)x * ph_w.taps + j]; int s = mirr(s0 + j, cw);
                u += w * Uf[(size_t)y * cw + s]; v += w * Vf[(size_t)y * cw + s]; }
            t0u[(size_t)y * W + x] = u; t0v[(size_t)y * W + x] = v;
        }
    std::vector<float> t1u((size_t)W * H), t1v((size_t)W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float u = 0, v = 0; int s0 = pv_w.start[y];
            for (int j = 0; j < pv_w.taps; j++) { float w = pv_w.wt[(size_t)y * pv_w.taps + j]; int s = mirr(s0 + j, ch);
                u += w * t0u[(size_t)s * W + x]; v += w * t0v[(size_t)s * W + x]; }
            t1u[(size_t)y * W + x] = u; t1v[(size_t)y * W + x] = v;
        }

    // ---- matrix + write the centered window into the padded tensor planes ----
    float* Rp = tensor + (size_t)(3 * frame_index + 0) * plane;
    float* Gp = tensor + (size_t)(3 * frame_index + 1) * plane;
    float* Bp = tensor + (size_t)(3 * frame_index + 2) * plane;
    for (int y = 0; y < H; y++) {
        size_t di = (size_t)(y + g.pad_t) * g.pw + g.pad_l;
        for (int x = 0; x < W; x++, di++) {
            size_t si = (size_t)y * W + x;
            float Y = (Yf[si] - csp.yoff) / csp.yscale;
            float U = (t1u[si] - csp.coff) / csp.cscale;
            float V = (t1v[si] - csp.coff) / csp.cscale;
            Rp[di] = Y + 2.0f * (1.0f - csp.kr) * V;
            Bp[di] = Y + 2.0f * (1.0f - csp.kb) * U;
            Gp[di] = Y - (2.0f * csp.kb * (1.0f - csp.kb) * U +
                          2.0f * csp.kr * (1.0f - csp.kr) * V) / kg;
        }
    }
}

// Centered RGB window in a 3-channel tensor -> NV12/P010 out, BT.709 RGB->YUV +
// bilinear chroma DOWNSAMPLE. Inverse of yuv420_to_rgb_planes.
void rgb_planes_to_yuv420(const float* tensor3, const Geom& g, aji_frame& out,
                          aji_range range) {
    const int W = g.w, H = g.h;
    const int cw = W >> 1, ch = H >> 1;
    const int format = out.format;
    const size_t plane = (size_t)g.pw * g.ph;
    const bool t16 = (format == AJI_FMT_P010);

    aji_csp csp = aji_resample::make_csp(format, AJI_MATRIX_BT709, range);
    const float kg = 1.0f - csp.kr - csp.kb;
    const float qdiv = t16 ? 64.0f : 1.0f;
    const float qmax = t16 ? 1023.0f : 255.0f;
    auto quant = [](float v, float qd, float qm){ float r = rintf(v / qd); r = r < 0 ? 0 : (r > qm ? qm : r); return r * qd; };

    // RGB->Y / Un / Vn at full res, reading the CENTERED window from the padded tensor.
    const float* R = tensor3 + (size_t)0 * plane;
    const float* G = tensor3 + (size_t)1 * plane;
    const float* B = tensor3 + (size_t)2 * plane;
    std::vector<float> Yout((size_t)W * H), Un((size_t)W * H), Vn((size_t)W * H);
    for (int y = 0; y < H; y++) {
        size_t si = (size_t)(y + g.pad_t) * g.pw + g.pad_l;
        for (int x = 0; x < W; x++, si++) {
            size_t di = (size_t)y * W + x;
            float r = R[si], gg = G[si], b = B[si];
            float Y = csp.kr * r + kg * gg + csp.kb * b;
            Yout[di] = quant(Y * csp.yscale + csp.yoff, qdiv, qmax);
            Un[di] = (b - Y) / (2.0f * (1.0f - csp.kb));
            Vn[di] = (r - Y) / (2.0f * (1.0f - csp.kr));
        }
    }

    // bilinear chroma DOWNSAMPLE, siting FORCED LEFT (matches gpu_post zimg RGB->YUV).
    double sx, sy; aji_resample::chroma_shifts(AJI_SITING_LEFT, false, &sx, &sy);
    weights ph_w = aji_resample::compute(W, cw, sx, AJI_FILTER_BILINEAR);
    weights pv_w = aji_resample::compute(H, ch, sy, AJI_FILTER_BILINEAR);

    std::vector<float> hu((size_t)cw * H), hv((size_t)cw * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < cw; x++) {
            float u = 0, v = 0; int s0 = ph_w.start[x];
            for (int j = 0; j < ph_w.taps; j++) { float w = ph_w.wt[(size_t)x * ph_w.taps + j]; int s = mirr(s0 + j, W);
                u += w * Un[(size_t)y * W + s]; v += w * Vn[(size_t)y * W + s]; }
            hu[(size_t)y * cw + x] = u; hv[(size_t)y * cw + x] = v;
        }
    std::vector<float> uvout((size_t)cw * ch * 2);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            float u = 0, v = 0; int s0 = pv_w.start[y];
            for (int j = 0; j < pv_w.taps; j++) { float w = pv_w.wt[(size_t)y * pv_w.taps + j]; int s = mirr(s0 + j, H);
                u += w * hu[(size_t)s * cw + x]; v += w * hv[(size_t)s * cw + x]; }
            uvout[((size_t)y * cw + x) * 2]     = quant(u * csp.cscale + csp.coff, qdiv, qmax);
            uvout[((size_t)y * cw + x) * 2 + 1] = quant(v * csp.cscale + csp.coff, qdiv, qmax);
        }

    // ---- store into the out frame (mirror gpu_post) ----
    const float* yf = Yout.data();
    const float* uvf = uvout.data();
    if (t16) {
        uint16_t* yp = (uint16_t*)out.plane[0];
        uint16_t* uvp = (uint16_t*)out.plane[1];
        ptrdiff_t ys = out.stride[0] ? out.stride[0] : (ptrdiff_t)W * 2;
        ptrdiff_t uvs = out.stride[1] ? out.stride[1] : (ptrdiff_t)cw * 2 * 2;
        for (int y = 0; y < H; y++) {
            uint16_t* row = (uint16_t*)((char*)yp + (ptrdiff_t)y * ys);
            for (int x = 0; x < W; x++) row[x] = (uint16_t)yf[(size_t)y * W + x];
        }
        for (int y = 0; y < ch; y++) {
            uint16_t* row = (uint16_t*)((char*)uvp + (ptrdiff_t)y * uvs);
            for (int x = 0; x < cw * 2; x++) row[x] = (uint16_t)uvf[(size_t)y * cw * 2 + x];
        }
    } else {
        uint8_t* yp = (uint8_t*)out.plane[0];
        uint8_t* uvp = (uint8_t*)out.plane[1];
        ptrdiff_t ys = out.stride[0] ? out.stride[0] : (ptrdiff_t)W;
        ptrdiff_t uvs = out.stride[1] ? out.stride[1] : (ptrdiff_t)cw * 2;
        for (int y = 0; y < H; y++) {
            uint8_t* row = yp + (ptrdiff_t)y * ys;
            for (int x = 0; x < W; x++) row[x] = (uint8_t)yf[(size_t)y * W + x];
        }
        for (int y = 0; y < ch; y++) {
            uint8_t* row = uvp + (ptrdiff_t)y * uvs;
            for (int x = 0; x < cw * 2; x++) row[x] = (uint8_t)uvf[(size_t)y * cw * 2 + x];
        }
    }
}

}  // namespace rife_cpu

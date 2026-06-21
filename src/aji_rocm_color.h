// Host-callable launchers for the GPU-resident color kernels (Phase B). The device
// kernels are compiled for the present GPU with hipRTC at runtime (see
// src/aji_rocm_color.cpp); these launchers are called from aji_rocm.cpp. All pointers
// are DEVICE pointers unless noted. The chroma resample weights match resample.h's
// spline36 exactly (computed on the host with aji_resample::compute and uploaded), so
// the GPU color is visually identical to the CPU gpu_pre/gpu_post path.
#pragma once
#include <cstddef>
#include <cstdint>

// Colorspace + quantization constants, mirroring resample.h's aji_csp + gpu_post.
// Y = kr*R + kg*G + kb*B (kg = 1-kr-kb); store quant(Y*yscale+yoff). chroma diffs
// Un=(B-Y)/(2(1-kb)), Vn=(R-Y)/(2(1-kr)); after resample store quant(u*cscale+coff).
// quant(v) = clamp(rint(v/qdiv), 0, qmax) * qdiv. P010 => qdiv=64,qmax=1023,is_p010=1.
typedef struct {
    float kr, kb;
    float yscale, yoff, cscale, coff;
    float qdiv, qmax;
    int is_p010;
} aji_color_csp;

#ifdef __cplusplus
extern "C" {
#endif

// Build/toolchain self-test: returns 0 if a trivial kernel ran correctly.
int aji_color_selftest(void);

// OUT-color: model output RGB fp16 (device, NCHW {3,H,W}) -> device NV12/P010 planes.
// Mirrors gpu_post: per-pixel RGB->Y(quant)+Un/Vn, then chroma H-downsample (W->cw) via
// (ph_start,ph_wt), then V-downsample (H->ch) via (pv_start,pv_wt) + quant + interleave.
// yplane/uvplane are device, tight (Y: W*H, UV: cw*ch*2 elements). Runs on `stream`.
void aji_gpu_out_color(const void* rgb_fp16, int W, int H, aji_color_csp csp,
                       const int* ph_start, const float* ph_wt, int ph_taps,
                       const int* pv_start, const float* pv_wt, int pv_taps,
                       float* Un, float* Vn, float* hu, float* hv,
                       void* yplane, void* uvplane, void* stream);

// 4:4:4 16-bit planar (Phase B): pure BT.709 matrix, NO chroma resample.
// post444: model output RGB fp16 (device, NCHW {3,H,W}) -> three full-res u16 planes
//          (Y/Cb/Cr); qdiv=1, qmax=65535 (is_p010 ignored).
// pre444:  three full-res u16 planes -> RGB fp16 (RIFE 4:4:4 input).
// ys = Y byte stride, cs = Cb/Cr byte stride. All pointers DEVICE. Runs on `stream`.
void aji_gpu_post444(const void* rgb_fp16, int w, int h, aji_color_csp csp,
                     void* yplane, void* cbplane, void* crplane,
                     ptrdiff_t ys, ptrdiff_t cs, void* stream);
void aji_gpu_pre444(const void* yplane, const void* cbplane, const void* crplane,
                    ptrdiff_t ys, ptrdiff_t cs, int w, int h, aji_color_csp csp,
                    void* rgb_fp16, void* stream);

#ifdef __cplusplus
}
#endif

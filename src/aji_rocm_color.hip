// GPU-resident color kernels for aji_rocm (Phase B). Compiled with hipcc; the host
// launchers (extern "C") are called from aji_rocm.cpp. Keeps the upscaled frame on the
// GPU: the model's fp16 RGB output is consumed directly into NV12/P010 here, so the 4K
// RGB buffer never crosses PCIe and the fp16->fp32 cast + CPU out-color disappear.
//
// Parity: the chroma resample weights are resample.h's spline36 weights, computed on the
// host and uploaded; each output pixel sums the same taps in the same order in fp32, so
// the result matches the CPU gpu_post within fp rounding.
#include <hip/hip_runtime.h>
#include "aji_rocm_color.h"

// resample.h's mirror() boundary, verbatim.
__device__ __forceinline__ int aji_mirr(int i, int n) {
    i = i < 0 ? -i - 1 : i;
    i = i >= n ? 2 * n - 1 - i : i;
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}
__device__ __forceinline__ float aji_quant(float v, float qdiv, float qmax) {
    float r = rintf(v / qdiv);
    r = r < 0.0f ? 0.0f : (r > qmax ? qmax : r);
    return r * qdiv;
}

// --- toolchain self-test ---
__global__ void k_selftest(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = d[i] + 1.0f;
}
extern "C" int aji_color_selftest(void) {
    const int n = 1024;
    float* d = nullptr;
    if (hipMalloc(&d, n * sizeof(float)) != hipSuccess) return -1;
    if (hipMemset(d, 0, n * sizeof(float)) != hipSuccess) { hipFree(d); return -1; }
    k_selftest<<<(n + 255) / 256, 256>>>(d, n);
    if (hipDeviceSynchronize() != hipSuccess) { hipFree(d); return -3; }
    float h0 = -1.0f;
    hipMemcpy(&h0, d, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d);
    return (h0 == 1.0f) ? 0 : -2;
}

// --- OUT-color (RGB fp16 -> NV12/P010), 3 passes mirroring gpu_post ---

// Pass 1: per full-res pixel, RGB->Y (quant -> Y plane) + Un/Vn chroma differences.
__global__ void k_out_y_uvdiff(const _Float16* __restrict__ rgb, int W, int H,
                               aji_color_csp csp, void* yplane,
                               float* __restrict__ Un, float* __restrict__ Vn) {
    long n = (long)W * H;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float R = (float)rgb[i], G = (float)rgb[n + i], B = (float)rgb[2 * n + i];
    float kg = 1.0f - csp.kr - csp.kb;
    float Y = csp.kr * R + kg * G + csp.kb * B;
    float yval = aji_quant(Y * csp.yscale + csp.yoff, csp.qdiv, csp.qmax);
    if (csp.is_p010) ((unsigned short*)yplane)[i] = (unsigned short)yval;
    else             ((unsigned char*)yplane)[i]  = (unsigned char)yval;
    Un[i] = (B - Y) / (2.0f * (1.0f - csp.kb));
    Vn[i] = (R - Y) / (2.0f * (1.0f - csp.kr));
}

// Pass 2: horizontal chroma downsample (W -> cw), full height.
__global__ void k_out_chroma_h(const float* __restrict__ Un, const float* __restrict__ Vn,
                               int W, int H, int cw,
                               const int* __restrict__ start, const float* __restrict__ wt, int taps,
                               float* __restrict__ hu, float* __restrict__ hv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || y >= H) return;
    float u = 0.0f, v = 0.0f; int s0 = start[x];
    for (int j = 0; j < taps; j++) {
        float w = wt[(long)x * taps + j]; int s = aji_mirr(s0 + j, W);
        u += w * Un[(long)y * W + s]; v += w * Vn[(long)y * W + s];
    }
    hu[(long)y * cw + x] = u; hv[(long)y * cw + x] = v;
}

// Pass 3: vertical chroma downsample (H -> ch) + quant + interleave into the UV plane.
__global__ void k_out_chroma_v(const float* __restrict__ hu, const float* __restrict__ hv,
                               int cw, int H, int ch,
                               const int* __restrict__ start, const float* __restrict__ wt, int taps,
                               aji_color_csp csp, void* uvplane) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || y >= ch) return;
    float u = 0.0f, v = 0.0f; int s0 = start[y];
    for (int j = 0; j < taps; j++) {
        float w = wt[(long)y * taps + j]; int s = aji_mirr(s0 + j, H);
        u += w * hu[(long)s * cw + x]; v += w * hv[(long)s * cw + x];
    }
    float uval = aji_quant(u * csp.cscale + csp.coff, csp.qdiv, csp.qmax);
    float vval = aji_quant(v * csp.cscale + csp.coff, csp.qdiv, csp.qmax);
    long o = ((long)y * cw + x) * 2;
    if (csp.is_p010) { ((unsigned short*)uvplane)[o] = (unsigned short)uval; ((unsigned short*)uvplane)[o + 1] = (unsigned short)vval; }
    else             { ((unsigned char*)uvplane)[o]  = (unsigned char)uval;  ((unsigned char*)uvplane)[o + 1]  = (unsigned char)vval;  }
}

extern "C" void aji_gpu_out_color(const void* rgb_fp16, int W, int H, aji_color_csp csp,
                                  const int* ph_start, const float* ph_wt, int ph_taps,
                                  const int* pv_start, const float* pv_wt, int pv_taps,
                                  float* Un, float* Vn, float* hu, float* hv,
                                  void* yplane, void* uvplane, void* stream) {
    hipStream_t s = (hipStream_t)stream;
    const _Float16* rgb = (const _Float16*)rgb_fp16;
    const int cw = W >> 1, ch = H >> 1;
    const long n = (long)W * H;
    const int bs = 256;
    k_out_y_uvdiff<<<(unsigned)((n + bs - 1) / bs), bs, 0, s>>>(rgb, W, H, csp, yplane, Un, Vn);
    dim3 blk(16, 16);
    k_out_chroma_h<<<dim3((cw + 15) / 16, (H + 15) / 16), blk, 0, s>>>(Un, Vn, W, H, cw, ph_start, ph_wt, ph_taps, hu, hv);
    k_out_chroma_v<<<dim3((cw + 15) / 16, (ch + 15) / 16), blk, 0, s>>>(hu, hv, cw, H, ch, pv_start, pv_wt, pv_taps, csp, uvplane);
}

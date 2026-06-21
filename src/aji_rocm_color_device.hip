// Device-only color kernels, embedded and JIT-compiled per-GPU with hipRTC at
// runtime (see src/aji_rocm_color.cpp). NOT a compiled translation unit: CMake
// reads this file and bakes it into libaji_rocm.so as a string. Keep it free of
// #includes and host code so hipRTC compiles it standalone. Math is identical to
// the former static aji_rocm_color.hip — parity is guarded by rocm_color_test.

// Mirror of aji_rocm_color.h's aji_color_csp (must match field-for-field).
typedef struct {
    float kr, kb;
    float yscale, yoff, cscale, coff;
    float qdiv, qmax;
    int is_p010;
} aji_color_csp;

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

extern "C" __global__ void k_selftest(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = d[i] + 1.0f;
}

extern "C" __global__ void k_out_y_uvdiff(const _Float16* __restrict__ rgb, int W, int H,
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

extern "C" __global__ void k_out_chroma_h(const float* __restrict__ Un, const float* __restrict__ Vn,
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

extern "C" __global__ void k_out_chroma_v(const float* __restrict__ hu, const float* __restrict__ hv,
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

// ---- 4:4:4 16-bit planar (Phase B): pure BT.709 matrix, NO chroma resample ----
// RGB fp16 (NCHW {3,H,W}) <-> three full-res u16 planes (Y, Cb, Cr). post writes u16 to
// all three planes with qdiv=1/qmax=65535; pre reads three u16 planes. Mirrors the TRT
// reference k_post444/k_pre444 (kernels.cu). Strides in bytes: ys (Y), cs (Cb and Cr).
extern "C" __global__ void k_post444(const _Float16* __restrict__ src, int w, int h,
                                     aji_color_csp csp, unsigned char* yp, long ys,
                                     unsigned char* cbp, unsigned char* crp, long cs) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    long plane = (long)w * h, idx = (long)y * w + x;
    float r = (float)src[idx], g = (float)src[plane + idx], b = (float)src[2 * plane + idx];
    float kg = 1.0f - csp.kr - csp.kb;
    float Y = csp.kr * r + kg * g + csp.kb * b;
    float U = (b - Y) / (2.0f * (1.0f - csp.kb));
    float V = (r - Y) / (2.0f * (1.0f - csp.kr));
    ((unsigned short*)(yp  + (long)y * ys))[x] = (unsigned short)aji_quant(Y * csp.yscale + csp.yoff, 1.0f, 65535.0f);
    ((unsigned short*)(cbp + (long)y * cs))[x] = (unsigned short)aji_quant(U * csp.cscale + csp.coff, 1.0f, 65535.0f);
    ((unsigned short*)(crp + (long)y * cs))[x] = (unsigned short)aji_quant(V * csp.cscale + csp.coff, 1.0f, 65535.0f);
}

extern "C" __global__ void k_pre444(const unsigned char* yp, long ys,
                                    const unsigned char* cbp, const unsigned char* crp, long cs,
                                    int w, int h, aji_color_csp csp, _Float16* dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    long plane = (long)w * h, idx = (long)y * w + x;
    float Y = (((const unsigned short*)(yp  + (long)y * ys))[x] - csp.yoff) / csp.yscale;
    float U = (((const unsigned short*)(cbp + (long)y * cs))[x] - csp.coff) / csp.cscale;
    float V = (((const unsigned short*)(crp + (long)y * cs))[x] - csp.coff) / csp.cscale;
    float kg = 1.0f - csp.kr - csp.kb;
    dst[idx]             = (_Float16)(Y + 2.0f * (1.0f - csp.kr) * V);
    dst[2 * plane + idx] = (_Float16)(Y + 2.0f * (1.0f - csp.kb) * U);
    dst[plane + idx]     = (_Float16)(Y - (2.0f * csp.kb * (1.0f - csp.kb) * U +
                                           2.0f * csp.kr * (1.0f - csp.kr) * V) / kg);
}

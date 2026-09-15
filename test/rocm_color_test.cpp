// Characterization test for the GPU out-color kernels. Builds a deterministic
// synthetic fp16 RGB frame, runs aji_gpu_out_color, and compares the resulting
// NV12/P010 Y+UV bytes against a committed golden. The harness links the frozen
// color ABI, so it is identical before and after the hipRTC refactor: the golden
// is captured from the static-hipcc build, and the JIT build must reproduce it.
//   ./aji_rocm_color_test --emit   # write goldens
//   ./aji_rocm_color_test          # compare (CI/regression)
// AJI_COLOR_TOL=N relaxes the per-byte tolerance (default 0) for diagnosis only.
#include <hip/hip_runtime.h>
#include "aji_rocm_color.h"
#include "aji.h"
#include "resample.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

using aji_resample::weights;

static const int OW = 128, OH = 128;   // <=720p: GPU and CPU paths are bit-exact here

// Deterministic fp16 RGB in NCHW {3,OH,OW}, values in [0,1].
static std::vector<unsigned short> make_rgb_fp16() {
    std::vector<float> f((size_t)3 * OW * OH);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < OH; y++)
            for (int x = 0; x < OW; x++)
                f[((size_t)c * OH + y) * OW + x] =
                    0.5f + 0.5f * std::sin(0.07f * x + 0.11f * y + 1.3f * c);
    std::vector<unsigned short> h(f.size());
    // host fp32->fp16: _Float16 is a host type under -mf16c, store its 2 bytes
    for (size_t i = 0; i < f.size(); i++) { _Float16 v = (_Float16)f[i]; std::memcpy(&h[i], &v, 2); }
    return h;
}

static int run_one(int fmt, const char* golden, bool emit) {
    const int mat = AJI_MATRIX_BT709, rng = AJI_RANGE_LIMITED;
    const int cw = OW >> 1, ch = OH >> 1;
    const bool p010 = (fmt == AJI_FMT_P010);
    const int bytes = p010 ? 2 : 1;

    aji_csp rc = aji_resample::make_csp(fmt, mat, rng);
    aji_color_csp csp{};
    csp.kr = rc.kr; csp.kb = rc.kb;
    csp.yscale = rc.yscale; csp.yoff = rc.yoff; csp.cscale = rc.cscale; csp.coff = rc.coff;
    csp.is_p010 = p010 ? 1 : 0;
    csp.qdiv = p010 ? 64.0f : 1.0f;
    csp.qmax = p010 ? 1023.0f : 255.0f;

    double sx, sy; aji_resample::chroma_shifts(AJI_SITING_LEFT, false, &sx, &sy);
    weights ph = aji_resample::compute(OW, cw, sx, AJI_FILTER_SPLINE36);
    weights pv = aji_resample::compute(OH, ch, sy, AJI_FILTER_SPLINE36);

    auto rgb = make_rgb_fp16();
    void *drgb=nullptr,*yplane=nullptr,*uvplane=nullptr;
    int *dphs=nullptr,*dpvs=nullptr; float *dphw=nullptr,*dpvw=nullptr;
    float *Un=nullptr,*Vn=nullptr,*hu=nullptr,*hv=nullptr;
    hipMalloc(&drgb, rgb.size()*2); hipMemcpy(drgb, rgb.data(), rgb.size()*2, hipMemcpyHostToDevice);
    auto upi=[&](int*&d,const std::vector<int>&v){hipMalloc(&d,v.size()*4);hipMemcpy(d,v.data(),v.size()*4,hipMemcpyHostToDevice);};
    auto upf=[&](float*&d,const std::vector<float>&v){hipMalloc(&d,v.size()*4);hipMemcpy(d,v.data(),v.size()*4,hipMemcpyHostToDevice);};
    upi(dphs,ph.start); upf(dphw,ph.wt); upi(dpvs,pv.start); upf(dpvw,pv.wt);
    hipMalloc(&Un,(size_t)OW*OH*4); hipMalloc(&Vn,(size_t)OW*OH*4);
    hipMalloc(&hu,(size_t)cw*OH*4); hipMalloc(&hv,(size_t)cw*OH*4);
    hipMalloc(&yplane,(size_t)OW*OH*bytes); hipMalloc(&uvplane,(size_t)cw*ch*2*bytes);

    aji_gpu_out_color(drgb, OW, OH, csp, dphs, dphw, ph.taps, dpvs, dpvw, pv.taps,
                      Un, Vn, hu, hv, yplane, uvplane, nullptr);
    if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "sync failed\n"); return 1; }

    const size_t ysz = (size_t)OW*OH*bytes, uvsz = (size_t)cw*ch*2*bytes;
    std::vector<unsigned char> got(ysz + uvsz);
    hipMemcpy(got.data(), yplane, ysz, hipMemcpyDeviceToHost);
    hipMemcpy(got.data()+ysz, uvplane, uvsz, hipMemcpyDeviceToHost);

    int rc_ret = 0;
    if (emit) {
        FILE* f = fopen(golden, "wb"); fwrite(got.data(), 1, got.size(), f); fclose(f);
        fprintf(stderr, "[emit] %s (%zu bytes)\n", golden, got.size());
    } else {
        FILE* f = fopen(golden, "rb");
        if (!f) { fprintf(stderr, "missing golden %s\n", golden); return 2; }
        std::vector<unsigned char> exp(got.size());
        size_t n = fread(exp.data(), 1, exp.size(), f); fclose(f);
        const int tol = getenv("AJI_COLOR_TOL") ? atoi(getenv("AJI_COLOR_TOL")) : 0;
        long bad = 0, maxd = 0;
        if (n != got.size()) { fprintf(stderr, "size mismatch %s\n", golden); return 3; }
        for (size_t i = 0; i < got.size(); i++) { int d = abs((int)got[i]-(int)exp[i]); if (d>maxd) maxd=d; if (d>tol) bad++; }
        fprintf(stderr, "[check] %s maxdiff=%ld baddiff=%ld\n", golden, maxd, bad);
        if (bad) rc_ret = 4;
    }
    for (void* p : {drgb,(void*)dphs,(void*)dphw,(void*)dpvs,(void*)dpvw,(void*)Un,(void*)Vn,(void*)hu,(void*)hv,yplane,uvplane}) hipFree(p);
    return rc_ret;
}

// 4:4:4 (Phase B): pre444/post444 are pure-matrix inverses (no chroma resample), so a
// RGB fp16 -> YUV444 u16 -> RGB fp16 round-trip should recover the input within fp16
// precision (the only loss is the u16 chroma quant in the middle, ~2e-5 in RGB).
static int run_444_roundtrip() {
    const int W = OW, H = OH;
    aji_csp rc = aji_resample::make_csp(AJI_FMT_YUV444P16, AJI_MATRIX_BT709, AJI_RANGE_LIMITED);
    aji_color_csp csp{};
    csp.kr = rc.kr; csp.kb = rc.kb;
    csp.yscale = rc.yscale; csp.yoff = rc.yoff; csp.cscale = rc.cscale; csp.coff = rc.coff;
    csp.qdiv = 1.0f; csp.qmax = 65535.0f; csp.is_p010 = 0;

    auto rgbh = make_rgb_fp16();                 // fp16 RGB NCHW {3,H,W}, [0,1]
    const size_t plane = (size_t)W * H;
    void *drgb=nullptr,*dY=nullptr,*dCb=nullptr,*dCr=nullptr,*drgb2=nullptr;
    hipMalloc(&drgb, 3*plane*2); hipMemcpy(drgb, rgbh.data(), 3*plane*2, hipMemcpyHostToDevice);
    hipMalloc(&dY, plane*2); hipMalloc(&dCb, plane*2); hipMalloc(&dCr, plane*2);
    hipMalloc(&drgb2, 3*plane*2);
    const ptrdiff_t st = (ptrdiff_t)W * 2;       // tight u16 planes
    aji_gpu_post444(drgb, W, H, csp, dY, dCb, dCr, st, st, nullptr);   // RGB -> YUV444 u16
    aji_gpu_pre444(dY, dCb, dCr, st, st, W, H, csp, drgb2, nullptr);   // YUV444 -> RGB
    if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "444 sync failed\n"); return 1; }

    std::vector<unsigned short> rgbo(3*plane);
    hipMemcpy(rgbo.data(), drgb2, 3*plane*2, hipMemcpyDeviceToHost);
    const double tol = 0.02;
    double maxd = 0; long bad = 0;
    for (size_t i = 0; i < rgbo.size(); i++) {
        _Float16 a, b; std::memcpy(&a, &rgbh[i], 2); std::memcpy(&b, &rgbo[i], 2);
        double d = std::fabs((double)(float)a - (double)(float)b);
        if (d > maxd) maxd = d; if (d > tol) bad++;
    }
    fprintf(stderr, "[check] 444 roundtrip RGB maxdiff=%.5f baddiff=%ld (tol %.3f)\n", maxd, bad, tol);
    for (void* p : {drgb,dY,dCb,dCr,drgb2}) hipFree(p);
    return bad ? 5 : 0;
}

int main(int argc, char** argv) {
    bool emit = argc > 1 && std::strcmp(argv[1], "--emit") == 0;
    if (aji_color_selftest() != 0) { fprintf(stderr, "selftest FAILED\n"); return 10; }
    fprintf(stderr, "selftest OK\n");
    int r = 0;
    r |= run_one(AJI_FMT_NV12, "test/fixtures/rocm_color_nv12_128.bin", emit);
    r |= run_one(AJI_FMT_P010, "test/fixtures/rocm_color_p010_128.bin", emit);
    r |= run_444_roundtrip();
    fprintf(stderr, r ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return r;
}

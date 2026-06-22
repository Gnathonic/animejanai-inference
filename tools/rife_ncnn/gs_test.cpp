// Validate gridsample_vulkan: run the de-batched RIFE ncnn model on CPU vs Vulkan
// and compare against the onnx reference. Build against the source libncnn.so.
#include "net.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>

static std::vector<float> readbin(const char* p, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    size_t got = fread(v.data(), 4, n, f); fclose(f);
    if (got != n) { fprintf(stderr, "short read %s: %zu/%zu\n", p, got, n); exit(1); }
    return v;
}

static void psnr(const std::vector<float>& a, const std::vector<float>& b, const char* tag) {
    double mse = 0, mx = 0, lo = 1e30, hi = -1e30;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - b[i]; mse += d * d; if (fabs(d) > mx) mx = fabs(d);
        lo = std::min(lo, (double)b[i]); hi = std::max(hi, (double)b[i]);
    }
    mse /= a.size();
    double rng = std::max(hi - lo, 1e-6);
    double p = mse < 1e-12 ? 99.0 : 20 * log10(rng) - 10 * log10(mse);
    printf("  %-32s maxdiff=%.5f  PSNR=%.1f dB\n", tag, mx, p);
}

// run model, return (3,oh,ow) channel-major
static std::vector<float> run(const char* param, const char* bin, bool vulkan,
                              const float* in, int W, int H, int* oc, int* oh, int* ow) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = vulkan;
    net.opt.use_fp16_packed = vulkan; net.opt.use_fp16_storage = vulkan; net.opt.use_fp16_arithmetic = vulkan;
    if (net.load_param(param)) { fprintf(stderr, "load_param fail\n"); exit(1); }
    if (net.load_model(bin)) { fprintf(stderr, "load_model fail\n"); exit(1); }
    ncnn::Mat inm(W, H, 11);
    for (int c = 0; c < 11; c++)
        memcpy(inm.channel(c), in + (size_t)c * W * H, (size_t)W * H * 4);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", inm);
    ncnn::Mat out;
    ex.extract("out0", out);
    *oc = out.c; *oh = out.h; *ow = out.w;
    std::vector<float> r((size_t)out.c * out.h * out.w);
    for (int c = 0; c < out.c; c++)
        memcpy(r.data() + (size_t)c * out.h * out.w, out.channel(c), (size_t)out.h * out.w * 4);
    return r;
}

static void inspect(const char* param, const char* bin) {
    ncnn::Net net; net.opt.use_vulkan_compute = true;
    net.load_param(param); net.load_model(bin);
    const std::vector<ncnn::Layer*>& L = net.layers();
    int gs = 0, gs_gpu = 0, nonvk = 0;
    for (size_t i = 0; i < L.size(); i++) {
        if (!L[i]->support_vulkan) nonvk++;
        if (L[i]->type == "GridSample") { gs++; if (L[i]->support_vulkan) gs_gpu++; }
    }
    printf("layers=%zu  non-vulkan(CPU-fallback)=%d  GridSample=%d (GPU-native=%d)\n",
           L.size(), nonvk, gs, gs_gpu);
}

static double bench(const char* param, const char* bin, bool vulkan, const float* in, int W, int H, int iters) {
    ncnn::Net net; net.opt.use_vulkan_compute = vulkan;
    net.opt.use_fp16_packed = vulkan; net.opt.use_fp16_storage = vulkan; net.opt.use_fp16_arithmetic = vulkan;
    net.load_param(param); net.load_model(bin);
    ncnn::Mat inm(W, H, 11);
    for (int c = 0; c < 11; c++) memcpy(inm.channel(c), in + (size_t)c * W * H, (size_t)W * H * 4);
    for (int w = 0; w < 3; w++) { ncnn::Extractor ex = net.create_extractor(); ex.input("in0", inm); ncnn::Mat o; ex.extract("out0", o); }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) { ncnn::Extractor ex = net.create_extractor(); ex.input("in0", inm); ncnn::Mat o; ex.extract("out0", o); }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / iters;
}

int main(int argc, char** argv) {
    const char* param = "rife_v4.14_db.ncnn.param";
    const char* bin = "rife_v4.14_db.ncnn.bin";
    inspect(param, bin);

    // case 1: 256x256 vs onnx-fp16 and onnx-fp32 refs
    {
        int W = 256, H = 256;
        auto in = readbin("in_256.bin", (size_t)11 * W * H);
        auto ref16 = readbin("ref16_256.bin", (size_t)3 * W * H);
        auto ref32 = readbin("ref32_256.bin", (size_t)3 * W * H);
        int oc, oh, ow;
        auto cpu = run(param, bin, false, in.data(), W, H, &oc, &oh, &ow);
        printf("[256x256] cpu out = %dx%dx%d\n", oc, oh, ow);
        auto vk = run(param, bin, true, in.data(), W, H, &oc, &oh, &ow);
        printf("[256x256] vulkan out = %dx%dx%d\n", oc, oh, ow);
        psnr(cpu, ref32, "cpu     vs onnx-fp32");
        psnr(vk, ref32, "vulkan  vs onnx-fp32");
        psnr(vk, ref16, "vulkan  vs onnx-fp16 (TRT/ROCm ref)");
        psnr(vk, cpu, "vulkan  vs cpu (both ncnn)");
    }
    // case 2: 640x384 non-square vs onnx-fp32
    {
        int W = 640, H = 384;
        auto in = readbin("in_640x384.bin", (size_t)11 * W * H);
        auto ref32 = readbin("ref32_640x384.bin", (size_t)3 * W * H);
        int oc, oh, ow;
        auto vk = run(param, bin, true, in.data(), W, H, &oc, &oh, &ow);
        printf("[640x384] vulkan out = %dx%dx%d\n", oc, oh, ow);
        psnr(vk, ref32, "vulkan  vs onnx-fp32 @640x384");
    }
    // timing
    {
        auto in = readbin("in_256.bin", (size_t)11 * 256 * 256);
        printf("[timing 256x256] cpu=%.2f ms  vulkan(native GridSample)=%.2f ms/frame\n",
               bench(param, bin, false, in.data(), 256, 256, 20),
               bench(param, bin, true, in.data(), 256, 256, 50));
    }
    return 0;
}

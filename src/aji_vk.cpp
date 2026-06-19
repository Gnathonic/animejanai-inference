// aji_vk — Vulkan + ncnn backend for the AnimeJaNai inference shim (libaji_vk.so).
//
// Full GPU color path with BOTH modes of the aji.h ABI:
//   - direct mode  (engine_path=<model>.param): a single fixed upscaler.
//   - conf mode    (conf_path + model_dir + slot): faithful resolution-conditional
//     model CHAINS with stacking, ported from aji_trt.cpp's chain selection and
//     reusing aji_conf.cpp verbatim.
//
// aji_frame carries HOST YUV per aji_frame.format:
//   AJI_FMT_NV12: plane[0]=Y u8 (W*H), plane[1]=CbCr interleaved u8 (cw*ch*2).
//   AJI_FMT_P010: same but u16 (10-bit MSB-aligned, value<<6).
// aji_infer does the YUV<->RGB color on the CPU via resample.h (the zimg reference
// math, bit-exact; OpenMP over rows + reused ctx scratch), and the model chain on
// ncnn-Vulkan:
//   YUV(host) -> CPU pre -> RGB -> [model chain + CPU Spline36 resize] -> CPU post -> YUV(host).
// The chain already bridges color<->model through host ncnn::Mat, so CPU color adds
// no extra GPU round-trip.
//
// The GLSL/SPV color kernels (color_init + ColorCtx below) are retained but INACTIVE:
// they hit a subtle ncnn-Vulkan-at-scale bug on RADV (>256px frames corrupt), so the
// CPU path carries color. Fixing that + a fully GPU-resident VkMat path is the P4
// optimization (it would lift the HD/4K tail, which is GPU/model-bound today).

#include "aji.h"
#include "aji_conf.h"
#include "kernels.h"
#include "resample.h"
#include "net.h"
#include "mat.h"

#include "gpu.h"
#include "command.h"
#include "pipeline.h"
#include "option.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using aji_resample::weights;

// The GLSL/SPV color path (color_init below) is currently INACTIVE: color runs on
// the CPU (resample.h, bit-exact — the GLSL kernels corrupt at >256px on RADV). The
// kernels are retained for the future GPU-resident path. When re-enabled, point this
// at the installed .spv dir via -DAJI_VK_SPV_DIR or the AJI_VK_SPV_DIR env var; the
// default is relative so no absolute/developer path is baked into the shipped lib.
#ifndef AJI_VK_SPV_DIR
#define AJI_VK_SPV_DIR "spv"
#endif
static const char* kSpvDir = []{ const char* e = getenv("AJI_VK_SPV_DIR"); return e && e[0] ? e : AJI_VK_SPV_DIR; }();

struct NcnnModel {
    ncnn::Net net;
    std::string in_name, out_name;
    int scale = 0;
};

struct Step {
    enum Kind { RESIZE, MODEL } kind;
    int out_w = 0, out_h = 0;
    int model_idx = -1;   // MODEL
};

// Persistent GPU color context: the VulkanDevice plus the pre/post/resize
// Pipelines, created once from the SPV files.
struct ColorCtx {
    const ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::Option opt;
    // pre
    ncnn::Pipeline* p_uvh = nullptr;   // cs_uv_h
    ncnn::Pipeline* p_v = nullptr;     // cs_v_f32
    ncnn::Pipeline* p_comb = nullptr;  // cs_pre_combine
    // post
    ncnn::Pipeline* p_mat = nullptr;   // cs_post_matrix
    ncnn::Pipeline* p_h = nullptr;     // cs_h_f32
    ncnn::Pipeline* p_uvv = nullptr;   // cs_post_uv_v (interleaved CbCr store)
    // resize
    ncnn::Pipeline* p_rsh = nullptr;   // cs_rs_h
    ncnn::Pipeline* p_rsv = nullptr;   // cs_rs_v
    bool ready = false;
};

struct aji_ctx {
    bool conf_mode = false;

    // direct mode
    std::string engine_path;

    // conf mode
    AjiConf conf;
    std::string model_dir;
    int slot = 1;

    // active plan (both modes)
    std::vector<std::unique_ptr<NcnnModel>> models;
    std::vector<Step> steps;
    int chain_scale = 0;          // product of model scales (for aji_scale_factor)
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    bool active = false;

    ColorCtx color;

    // Per-frame CPU-color scratch, reused across frames. Allocating these (~20MB
    // at 960x720) every frame sends each large block through mmap/munmap, so every
    // frame re-faults its pages — a periodic cost that shows up as playback
    // stutter. One ctx processes frames sequentially, so sharing is safe; the
    // OpenMP regions only ever write distinct indices. resize() keeps capacity,
    // so after the first frame of a stream there is no reallocation.
    struct Scratch {
        std::vector<float> Yf, Uf, Vf;                  // host YUV -> float
        std::vector<float> t0u, t0v, t1u, t1v;          // pre chroma upsample
        std::vector<float> rt0;                         // resize H intermediate
        std::vector<float> Yout, Un, Vn, hu, hv, uvout; // post
    } scratch;

    // CPU-color OpenMP width. Capped: on small frames the per-region spawn/sync
    // overhead of many threads outweighs the work (measured: 8 ~= peak, 32 < 1).
    int nthreads = 1;

    std::string err, log;
    aji_log_fn logfn = nullptr;
    void* log_opaque = nullptr;
};

static void logmsg(aji_ctx* c, int level, const char* m) {
    if (c && c->logfn) c->logfn(c->log_opaque, level, m);
}

static int round_even(double x) {
    int r = (int)(x + 0.5);
    return r;
}

static std::string bin_for(const std::string& param) {
    auto pos = param.rfind(".param");
    if (pos != std::string::npos) return param.substr(0, pos) + ".bin";
    return param + ".bin";
}

// Load a classic-ncnn model and probe its spatial scale with a small dummy run.
static std::unique_ptr<NcnnModel> load_model(aji_ctx* c, const std::string& param) {
    auto m = std::make_unique<NcnnModel>();
    m->net.opt.use_vulkan_compute = true;
    m->net.opt.use_fp16_packed = true;
    m->net.opt.use_fp16_storage = true;
    m->net.opt.use_fp16_arithmetic = true;
    // ncnn has no TensorRT-style per-layer engine build; the closest wins are
    // enabling the matrix-core (cooperative_matrix) GEMM path and Winograd 3x3 (the
    // SPAN models are all 3x3 convs). Gated by device support inside ncnn, so safe to
    // request unconditionally; AJI_VK_NO_CM=1 disables coopmat for A/B comparison.
    m->net.opt.use_cooperative_matrix = getenv("AJI_VK_NO_CM") == nullptr;
    m->net.opt.use_winograd_convolution = true;
    m->net.opt.use_sgemm_convolution = true;
    m->net.opt.use_subgroup_ops = true;
    m->net.opt.use_shader_local_memory = true;
    if (m->net.load_param(param.c_str())) { c->err = "load_param failed: " + param; return nullptr; }
    std::string bin = bin_for(param);
    if (m->net.load_model(bin.c_str())) { c->err = "load_model failed: " + bin; return nullptr; }
    const auto& ins = m->net.input_names();
    const auto& outs = m->net.output_names();
    if (ins.empty() || outs.empty()) { c->err = "model has no input/output blobs: " + param; return nullptr; }
    m->in_name = ins[0];
    m->out_name = outs[0];
    const int S = 32;
    ncnn::Mat dummy(S, S, 3); dummy.fill(0.5f);
    ncnn::Extractor ex = m->net.create_extractor();
    ncnn::Mat o;
    if (ex.input(m->in_name.c_str(), dummy) != 0 || ex.extract(m->out_name.c_str(), o) != 0 || o.w % S != 0) {
        c->err = "scale probe failed: " + param; return nullptr;
    }
    m->scale = o.w / S;
    return m;
}

// fit (sw,sh) into a (bw,bh) box preserving aspect (mirrors aji_trt fit_box).
static void fit_box(int sw, int sh, double bw, double bh, int* nw, int* nh) {
    double s = std::min(bw / sw, bh / sh);
    *nw = round_even(sw * s);
    *nh = round_even(sh * s);
}

// ------------------------- GPU color helpers -------------------------------

static std::vector<uint32_t> load_spv(const std::string& path, bool* ok) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { *ok = false; return {}; }
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> v(sz / 4);
    f.read((char*)v.data(), sz);
    *ok = (bool)f;
    return v;
}

// Lazily build the persistent VulkanDevice + Pipelines from the SPV files.
static bool color_init(aji_ctx* c) {
    ColorCtx& cc = c->color;
    if (cc.ready) return true;

    ncnn::create_gpu_instance();
    cc.vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    if (!cc.vkdev) { c->err = "color_init: no vulkan device"; return false; }
    cc.opt.blob_vkallocator = cc.vkdev->acquire_blob_allocator();
    cc.opt.workspace_vkallocator = cc.opt.blob_vkallocator;
    cc.opt.staging_vkallocator = cc.vkdev->acquire_staging_allocator();
    cc.opt.use_fp16_packed = false;
    cc.opt.use_fp16_storage = false;
    cc.opt.use_fp16_arithmetic = false;

    auto mkpipe = [&](const char* name) -> ncnn::Pipeline* {
        bool ok = false;
        std::string path = std::string(kSpvDir) + "/" + name + ".spv";
        auto spv = load_spv(path, &ok);
        if (!ok || spv.empty()) { c->err = "color_init: failed to load " + path; return nullptr; }
        ncnn::Pipeline* p = new ncnn::Pipeline(cc.vkdev);
        p->set_local_size_xyz(32, 8, 1);
        if (p->create(spv.data(), spv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
            c->err = "color_init: pipeline create failed for " + std::string(name);
            delete p; return nullptr;
        }
        return p;
    };

    cc.p_uvh  = mkpipe("cs_uv_h");
    cc.p_v    = mkpipe("cs_v_f32");
    cc.p_comb = mkpipe("cs_pre_combine");
    cc.p_mat  = mkpipe("cs_post_matrix");
    cc.p_h    = mkpipe("cs_h_f32");
    cc.p_uvv  = mkpipe("cs_post_uv_v");
    cc.p_rsh  = mkpipe("cs_rs_h");
    cc.p_rsv  = mkpipe("cs_rs_v");
    if (!cc.p_uvh || !cc.p_v || !cc.p_comb || !cc.p_mat || !cc.p_h || !cc.p_uvv || !cc.p_rsh || !cc.p_rsv)
        return false;

    cc.ready = true;
    return true;
}

static void color_free(ColorCtx& cc) {
    delete cc.p_uvh; delete cc.p_v; delete cc.p_comb;
    delete cc.p_mat; delete cc.p_h; delete cc.p_uvv;
    delete cc.p_rsh; delete cc.p_rsv;
    if (cc.vkdev) {
        if (cc.opt.blob_vkallocator) cc.vkdev->reclaim_blob_allocator(cc.opt.blob_vkallocator);
        if (cc.opt.staging_vkallocator) cc.vkdev->reclaim_staging_allocator(cc.opt.staging_vkallocator);
    }
    cc = ColorCtx();
}

static void up_buf(ncnn::VkCompute& cmd, ColorCtx& cc, std::vector<float>& host,
                   int w, int h, ncnn::VkMat& vk) {
    ncnn::Mat m(w * h, (void*)host.data(), (size_t)4u);
    cmd.record_upload(m, vk, cc.opt);
}

// GLSL pre: host YUV planes (Y full-res, U/V half-res, all as fp32 raw container
// values) -> RGB fp32 NCHW ncnn::Mat (W x H x 3). Mirrors kernel_test/p010_test.
static int gpu_pre(aji_ctx* c, int W, int H, int format, int matrix, int range, int siting,
                   std::vector<float>& Yf, std::vector<float>& Uf, std::vector<float>& Vf,
                   ncnn::Mat& rgb_out) {
    // CPU implementation using resample.h (the zimg reference math; bit-exact).
    // The GLSL kernels are validated at 256 but hit a subtle ncnn-Vulkan bug at
    // larger frames; since the chain already bridges through host ncnn::Mat, CPU
    // color adds no extra round-trip. GPU-resident color is the P4 optimization.
    (void)c;
    const int cw = W >> 1, ch = H >> 1;
    aji_csp csp = aji_resample::make_csp(format, matrix, range);
    double sx, sy; aji_resample::chroma_shifts(siting, true, &sx, &sy);
    weights ph = aji_resample::compute(cw, W, sx, AJI_FILTER_SPLINE36);
    weights pv = aji_resample::compute(ch, H, sy, AJI_FILTER_SPLINE36);
    auto mirr = [](int i, int n){ i = i < 0 ? -i - 1 : i; i = i >= n ? 2*n - 1 - i : i; return i < 0 ? 0 : (i >= n ? n - 1 : i); };

    // The outer row loops are independent (each writes a distinct output row and
    // reads only inputs/weights); parallelizing over rows leaves the inner tap
    // reduction order untouched, so the result stays bit-exact to the serial run.
    // Scratch lives in the ctx and is reused across frames (no per-frame mmap).
    const int NT = c->nthreads;
    std::vector<float>& t0u = c->scratch.t0u; t0u.resize((size_t)W * ch);
    std::vector<float>& t0v = c->scratch.t0v; t0v.resize((size_t)W * ch);
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < W; x++) {
            float u = 0, v = 0; int s0 = ph.start[x];
            for (int j = 0; j < ph.taps; j++) { float w = ph.wt[(size_t)x * ph.taps + j]; int s = mirr(s0 + j, cw);
                u += w * Uf[(size_t)y * cw + s]; v += w * Vf[(size_t)y * cw + s]; }
            t0u[(size_t)y * W + x] = u; t0v[(size_t)y * W + x] = v;
        }
    std::vector<float>& t1u = c->scratch.t1u; t1u.resize((size_t)W * H);
    std::vector<float>& t1v = c->scratch.t1v; t1v.resize((size_t)W * H);
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float u = 0, v = 0; int s0 = pv.start[y];
            for (int j = 0; j < pv.taps; j++) { float w = pv.wt[(size_t)y * pv.taps + j]; int s = mirr(s0 + j, ch);
                u += w * t0u[(size_t)s * W + x]; v += w * t0v[(size_t)s * W + x]; }
            t1u[(size_t)y * W + x] = u; t1v[(size_t)y * W + x] = v;
        }
    rgb_out.create(W, H, 3);
    float* Rp = rgb_out.channel(0); float* Gp = rgb_out.channel(1); float* Bp = rgb_out.channel(2);
    const float kg = 1.0f - csp.kr - csp.kb;
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (long i = 0; i < (long)W * H; i++) {
        float Y = (Yf[i] - csp.yoff) / csp.yscale;
        float U = (t1u[i] - csp.coff) / csp.cscale;
        float V = (t1v[i] - csp.coff) / csp.cscale;
        Rp[i] = Y + 2.0f * (1.0f - csp.kr) * V;
        Bp[i] = Y + 2.0f * (1.0f - csp.kb) * U;
        Gp[i] = Y - (2.0f * csp.kb * (1.0f - csp.kb) * U + 2.0f * csp.kr * (1.0f - csp.kr) * V) / kg;
    }
    return AJI_OK;
}

// GLSL Spline36 resize: RGB fp32 NCHW (SW x SH x 3) -> RGB fp32 NCHW (DW x DH x 3).
static int gpu_resize(aji_ctx* c, const ncnn::Mat& src, int DW, int DH, ncnn::Mat& dst) {
    const int SW = src.w, SH = src.h;
    const int NT = c->nthreads;
    weights ph = aji_resample::compute(SW, DW, 0.0, AJI_FILTER_SPLINE36);
    weights pv = aji_resample::compute(SH, DH, 0.0, AJI_FILTER_SPLINE36);
    auto mirr = [](int i, int n){ i = i < 0 ? -i - 1 : i; i = i >= n ? 2*n - 1 - i : i; return i < 0 ? 0 : (i >= n ? n - 1 : i); };

    dst.create(DW, DH, 3);
    std::vector<float>& t0 = c->scratch.rt0; t0.resize((size_t)DW * SH);
    for (int k = 0; k < 3; k++) {
        const float* s = src.channel(k);
        #pragma omp parallel for schedule(static) num_threads(NT)
        for (int y = 0; y < SH; y++)
            for (int x = 0; x < DW; x++) {
                float a = 0; int s0 = ph.start[x];
                for (int j = 0; j < ph.taps; j++) a += ph.wt[(size_t)x * ph.taps + j] * s[(size_t)y * SW + mirr(s0 + j, SW)];
                t0[(size_t)y * DW + x] = a;
            }
        float* d = dst.channel(k);
        #pragma omp parallel for schedule(static) num_threads(NT)
        for (int y = 0; y < DH; y++)
            for (int x = 0; x < DW; x++) {
                float a = 0; int s0 = pv.start[y];
                for (int j = 0; j < pv.taps; j++) a += pv.wt[(size_t)y * pv.taps + j] * t0[(size_t)mirr(s0 + j, SH) * DW + x];
                d[(size_t)y * DW + x] = a;
            }
    }
    return AJI_OK;
}

// GLSL post: RGB fp32 NCHW (W x H x 3) -> host YUV (NV12 8-bit or P010 16-bit).
// siting is FORCED LEFT for the chroma downsample (zimg semantics on RGB->YUV).
static int gpu_post(aji_ctx* c, const ncnn::Mat& rgb, int format, int matrix, int range,
                    const aji_frame* out) {
    const int NT = c->nthreads;
    const int W = rgb.w, H = rgb.h;
    const int cw = W >> 1, ch = H >> 1;

    aji_csp csp = aji_resample::make_csp(format, matrix, range);
    double sx, sy; aji_resample::chroma_shifts(AJI_SITING_LEFT, false, &sx, &sy);  // FORCE LEFT
    weights ph = aji_resample::compute(W, cw, sx, AJI_FILTER_SPLINE36);
    weights pv = aji_resample::compute(H, ch, sy, AJI_FILTER_SPLINE36);
    const float qdiv = (format == AJI_FMT_P010) ? 64.0f : 1.0f;
    const float qmax = (format == AJI_FMT_P010) ? 1023.0f : 255.0f;
    auto mirr = [](int i, int n){ i = i < 0 ? -i - 1 : i; i = i >= n ? 2*n - 1 - i : i; return i < 0 ? 0 : (i >= n ? n - 1 : i); };
    auto quant = [](float v, float qd, float qm){ float r = rintf(v / qd); r = r < 0 ? 0 : (r > qm ? qm : r); return r * qd; };

    const float* R = rgb.channel(0); const float* G = rgb.channel(1); const float* B = rgb.channel(2);
    const float kg = 1.0f - csp.kr - csp.kb;
    std::vector<float>& Yout = c->scratch.Yout; Yout.resize((size_t)W * H);
    std::vector<float>& Un = c->scratch.Un; Un.resize((size_t)W * H);
    std::vector<float>& Vn = c->scratch.Vn; Vn.resize((size_t)W * H);
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (long i = 0; i < (long)W * H; i++) {
        float Y = csp.kr * R[i] + kg * G[i] + csp.kb * B[i];
        Yout[i] = quant(Y * csp.yscale + csp.yoff, qdiv, qmax);
        Un[i] = (B[i] - Y) / (2.0f * (1.0f - csp.kb));
        Vn[i] = (R[i] - Y) / (2.0f * (1.0f - csp.kr));
    }
    std::vector<float>& hu = c->scratch.hu; hu.resize((size_t)cw * H);
    std::vector<float>& hv = c->scratch.hv; hv.resize((size_t)cw * H);
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (int y = 0; y < H; y++)
        for (int x = 0; x < cw; x++) {
            float u = 0, v = 0; int s0 = ph.start[x];
            for (int j = 0; j < ph.taps; j++) { float w = ph.wt[(size_t)x * ph.taps + j]; int s = mirr(s0 + j, W);
                u += w * Un[(size_t)y * W + s]; v += w * Vn[(size_t)y * W + s]; }
            hu[(size_t)y * cw + x] = u; hv[(size_t)y * cw + x] = v;
        }
    std::vector<float>& uvout = c->scratch.uvout; uvout.resize((size_t)cw * ch * 2);
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            float u = 0, v = 0; int s0 = pv.start[y];
            for (int j = 0; j < pv.taps; j++) { float w = pv.wt[(size_t)y * pv.taps + j]; int s = mirr(s0 + j, H);
                u += w * hu[(size_t)s * cw + x]; v += w * hv[(size_t)s * cw + x]; }
            uvout[((size_t)y * cw + x) * 2]     = quant(u * csp.cscale + csp.coff, qdiv, qmax);
            uvout[((size_t)y * cw + x) * 2 + 1] = quant(v * csp.cscale + csp.coff, qdiv, qmax);
        }
    const float* yf = Yout.data();
    const float* uvf = uvout.data();
    if (format == AJI_FMT_P010) {
        uint16_t* yp = (uint16_t*)out->plane[0];
        uint16_t* uvp = (uint16_t*)out->plane[1];
        ptrdiff_t ys = out->stride[0] ? out->stride[0] : (ptrdiff_t)W * 2;
        ptrdiff_t uvs = out->stride[1] ? out->stride[1] : (ptrdiff_t)cw * 2 * 2;
        for (int y = 0; y < H; y++) {
            uint16_t* row = (uint16_t*)((char*)yp + (ptrdiff_t)y * ys);
            for (int x = 0; x < W; x++) row[x] = (uint16_t)yf[(size_t)y * W + x];
        }
        for (int y = 0; y < ch; y++) {
            uint16_t* row = (uint16_t*)((char*)uvp + (ptrdiff_t)y * uvs);
            for (int x = 0; x < cw * 2; x++) row[x] = (uint16_t)uvf[(size_t)y * cw * 2 + x];
        }
    } else {
        uint8_t* yp = (uint8_t*)out->plane[0];
        uint8_t* uvp = (uint8_t*)out->plane[1];
        ptrdiff_t ys = out->stride[0] ? out->stride[0] : (ptrdiff_t)W;
        ptrdiff_t uvs = out->stride[1] ? out->stride[1] : (ptrdiff_t)cw * 2;
        for (int y = 0; y < H; y++) {
            uint8_t* row = yp + (ptrdiff_t)y * ys;
            for (int x = 0; x < W; x++) row[x] = (uint8_t)yf[(size_t)y * W + x];
        }
        for (int y = 0; y < ch; y++) {
            uint8_t* row = uvp + (ptrdiff_t)y * uvs;
            for (int x = 0; x < cw * 2; x++) row[x] = (uint8_t)uvf[(size_t)y * cw * 2 + x];
        }
    }
    return AJI_OK;
}

// ------------------------------- plan --------------------------------------

// Build the step plan for the active chain (conf mode) or single model (direct).
static int build_plan(aji_ctx* c, int w, int h, double fps) {
    c->models.clear();
    c->steps.clear();
    c->chain_scale = 0;
    int cw = w, ch = h;

    if (!c->conf_mode) {
        auto m = load_model(c, c->engine_path);
        if (!m) return AJI_ERR_ENGINE;
        int sc = m->scale;
        c->models.push_back(std::move(m));
        c->steps.push_back({Step::MODEL, w * sc, h * sc, 0});
        cw = w * sc; ch = h * sc;
        c->chain_scale = sc;
        c->log = "step 1: aji_vk " + std::to_string(sc) + "x (ncnn-Vulkan)";
    } else {
        auto sit = c->conf.slots.find(c->slot);
        const AjiChainConf* chain = nullptr;
        if (sit != c->conf.slots.end()) {
            double px = (double)w * h;
            for (const auto& chn : sit->second.chains) {
                if (chn.min_px <= px && px <= chn.max_px && chn.min_fps <= fps && fps <= chn.max_fps) {
                    chain = &chn; break;
                }
            }
        } else {
            // The requested slot isn't in the conf at all (typo'd slot=, wrong
            // conf path falling back to built-ins, ...). Passthrough is the
            // faithful fallback, but warn loudly so it isn't mistaken for the
            // upscaler silently doing nothing.
            char w2[160];
            snprintf(w2, sizeof w2, "slot %d not found in conf (%zu slots loaded: "
                     "e.g. %d); passing through unscaled", c->slot, c->conf.slots.size(),
                     c->conf.slots.empty() ? -1 : c->conf.slots.begin()->first);
            logmsg(c, 2, w2);
        }
        if (!chain) { c->log = "No chains activated"; c->out_w = w; c->out_h = h; c->active = false; return 0; }

        std::string steps_log;
        for (const auto& m : chain->models) {
            double factor = m.resize_factor_before_upscale;
            if (m.resize_height_before_upscale != 0) factor = 100;
            if (factor != 100) {
                int nw = round_even(cw * factor / 100.0), nh = round_even(ch * factor / 100.0);
                if (nw >= 2 && nh >= 2 && (nw != cw || nh != ch)) {
                    c->steps.push_back({Step::RESIZE, nw, nh, -1}); cw = nw; ch = nh;
                    steps_log += "resize " + std::to_string(factor) + "% -> " + std::to_string(cw) + "x" + std::to_string(ch) + "; ";
                }
            }
            if (m.resize_height_before_upscale != 0 && (int)m.resize_height_before_upscale != ch) {
                int nw, nh; fit_box(cw, ch, m.resize_height_before_upscale * 16.0 / 9.0, m.resize_height_before_upscale, &nw, &nh);
                c->steps.push_back({Step::RESIZE, nw, nh, -1}); cw = nw; ch = nh;
                steps_log += "resize h" + std::to_string((int)m.resize_height_before_upscale) + " -> " + std::to_string(cw) + "x" + std::to_string(ch) + "; ";
            } else if (ch > 1080) {
                int nw, nh; fit_box(cw, ch, 1920, 1080, &nw, &nh);
                c->steps.push_back({Step::RESIZE, nw, nh, -1}); cw = nw; ch = nh;
                steps_log += "resize >1080 -> " + std::to_string(cw) + "x" + std::to_string(ch) + "; ";
            }
            if (m.name.empty()) continue;
            std::string param = c->model_dir + "/" + m.name + ".param";
            auto nm = load_model(c, param);
            if (!nm) { logmsg(c, 2, c->err.c_str()); return AJI_ERR_ENGINE; }
            int sc = nm->scale;
            c->models.push_back(std::move(nm));
            cw *= sc; ch *= sc;
            c->steps.push_back({Step::MODEL, cw, ch, (int)c->models.size() - 1});
            c->chain_scale = c->chain_scale ? c->chain_scale * sc : sc;
            steps_log += "model " + m.name + " " + std::to_string(sc) + "x -> " + std::to_string(cw) + "x" + std::to_string(ch) + "; ";
        }
        if (chain->rife) steps_log += "(RIFE requested — not yet supported in aji_vk); ";
        char hdr[160];
        snprintf(hdr, sizeof hdr, "slot %d chain %d: %dx%d -> %dx%d  [", c->slot, chain->index, w, h, cw, ch);
        c->log = std::string(hdr) + steps_log + "]";
    }

    c->in_w = w; c->in_h = h; c->out_w = cw; c->out_h = ch;
    c->active = !c->steps.empty();
    return c->active ? 1 : 0;
}

extern "C" {

AJI_EXPORT aji_ctx* aji_create(const aji_create_params* p) {
    if (!p || p->api_version < 7) return nullptr;
    aji_ctx* c = new aji_ctx();
    c->logfn = p->log;
    c->log_opaque = p->log_opaque;
#ifdef _OPENMP
    c->nthreads = std::max(1, std::min(8, omp_get_max_threads()));
#endif

    if (p->conf_path && p->conf_path[0]) {
        c->conf_mode = true;
        c->model_dir = p->model_dir ? p->model_dir : ".";
        c->slot = p->slot ? p->slot : 1;
        std::string err;
        if (!aji_conf_load(p->conf_path, &c->conf, &err)) {
            c->err = "aji_conf_load failed: " + err; logmsg(c, 2, c->err.c_str()); return c;
        }
        char msg[160];
        snprintf(msg, sizeof msg, "aji_vk conf mode: %zu slots, slot=%d, model_dir=%s",
                 c->conf.slots.size(), c->slot, c->model_dir.c_str());
        logmsg(c, 1, msg);
    } else if (p->engine_path && p->engine_path[0]) {
        c->engine_path = p->engine_path;
        logmsg(c, 1, "aji_vk direct mode");
    } else {
        c->err = "aji_vk needs conf_path (conf mode) or engine_path (direct mode)";
        logmsg(c, 2, c->err.c_str());
    }
    return c;
}

AJI_EXPORT int aji_set_slot(aji_ctx* c, int slot) { if (c) c->slot = slot; return AJI_OK; }

AJI_EXPORT int aji_configure(aji_ctx* c, int w, int h, double fps, int* out_w, int* out_h) {
    if (!c) return AJI_ERR;
    int r = build_plan(c, w, h, fps);
    if (r < 0) { return r; }
    if (out_w) *out_w = c->out_w;
    if (out_h) *out_h = c->out_h;
    return r;  // 1 active, 0 passthrough
}

AJI_EXPORT int aji_infer(aji_ctx* c, const aji_frame* in, const aji_frame* out, void* /*stream*/) {
    if (!c || !c->active) return AJI_ERR;
    if (!in || !out) return AJI_ERR;
    if (in->width != c->in_w || in->height != c->in_h) return AJI_ERR_SHAPE;

    const int W = in->width, H = in->height;
    const int cw = W >> 1, ch = H >> 1;
    const int format = in->format ? in->format : AJI_FMT_NV12;
    const int matrix = in->matrix ? in->matrix : AJI_MATRIX_BT709;
    const int range  = in->range  ? in->range  : AJI_RANGE_LIMITED;
    const int siting = in->siting ? in->siting : AJI_SITING_LEFT;
    const int out_format = out->format ? out->format : format;
    const int out_matrix = out->matrix ? out->matrix : matrix;
    const int out_range  = out->range  ? out->range  : range;

    if (format != AJI_FMT_NV12 && format != AJI_FMT_P010) { c->err = "aji_infer: unsupported input format"; return AJI_ERR_FORMAT; }
    if (out_format != AJI_FMT_NV12 && out_format != AJI_FMT_P010) { c->err = "aji_infer: unsupported output format"; return AJI_ERR_FORMAT; }

    // Color is done on CPU (resample.h, bit-exact reference) — no GPU color init
    // needed. (The GLSL kernels in color_init are kept for the future P4
    // GPU-resident path but are not on the critical path and corrupt at scale.)

    // ---- read host YUV planes into fp32 raw-container values ----
    std::vector<float>& Yf = c->scratch.Yf; Yf.resize((size_t)W * H);
    std::vector<float>& Uf = c->scratch.Uf; Uf.resize((size_t)cw * ch);
    std::vector<float>& Vf = c->scratch.Vf; Vf.resize((size_t)cw * ch);
    ptrdiff_t ys = in->stride[0] ? in->stride[0] : (ptrdiff_t)W * (format == AJI_FMT_P010 ? 2 : 1);
    ptrdiff_t uvs = in->stride[1] ? in->stride[1] : (ptrdiff_t)cw * 2 * (format == AJI_FMT_P010 ? 2 : 1);
    if (format == AJI_FMT_P010) {
        const uint8_t* yb = (const uint8_t*)in->plane[0];
        const uint8_t* uvb = (const uint8_t*)in->plane[1];
        for (int y = 0; y < H; y++) {
            const uint16_t* row = (const uint16_t*)(yb + (ptrdiff_t)y * ys);
            for (int x = 0; x < W; x++) Yf[(size_t)y * W + x] = (float)row[x];
        }
        for (int y = 0; y < ch; y++) {
            const uint16_t* row = (const uint16_t*)(uvb + (ptrdiff_t)y * uvs);
            for (int x = 0; x < cw; x++) { Uf[(size_t)y * cw + x] = (float)row[2*x]; Vf[(size_t)y * cw + x] = (float)row[2*x+1]; }
        }
    } else {
        const uint8_t* yb = (const uint8_t*)in->plane[0];
        const uint8_t* uvb = (const uint8_t*)in->plane[1];
        for (int y = 0; y < H; y++) {
            const uint8_t* row = yb + (ptrdiff_t)y * ys;
            for (int x = 0; x < W; x++) Yf[(size_t)y * W + x] = (float)row[x];
        }
        for (int y = 0; y < ch; y++) {
            const uint8_t* row = uvb + (ptrdiff_t)y * uvs;
            for (int x = 0; x < cw; x++) { Uf[(size_t)y * cw + x] = (float)row[2*x]; Vf[(size_t)y * cw + x] = (float)row[2*x+1]; }
        }
    }

    // ---- GLSL pre: YUV -> RGB fp32 NCHW ----
    ncnn::Mat cur;
    if (gpu_pre(c, W, H, format, matrix, range, siting, Yf, Uf, Vf, cur) != AJI_OK) {
        logmsg(c, 2, c->err.c_str()); return AJI_ERR;
    }

    // ---- model chain + resize steps (RGB ncnn::Mat throughput) ----
    for (const Step& st : c->steps) {
        if (st.kind == Step::RESIZE) {
            ncnn::Mat dst;
            if (gpu_resize(c, cur, st.out_w, st.out_h, dst) != AJI_OK) { logmsg(c, 2, c->err.c_str()); return AJI_ERR; }
            cur = dst;
        } else {
            NcnnModel* nm = c->models[st.model_idx].get();
            ncnn::Extractor ex = nm->net.create_extractor();
            if (ex.input(nm->in_name.c_str(), cur) != 0) { c->err = "infer: input failed"; return AJI_ERR; }
            ncnn::Mat o;
            if (ex.extract(nm->out_name.c_str(), o) != 0) { c->err = "infer: extract failed"; return AJI_ERR; }
            cur = o;
        }
    }
    if (cur.w != c->out_w || cur.h != c->out_h || cur.c < 3) { c->err = "infer: unexpected output dims"; return AJI_ERR_SHAPE; }

    // GLSL post wants a contiguous 3-channel NCHW Mat; if the model produced a
    // packed/elempack mat, flatten to a fresh fp32 NCHW Mat.
    ncnn::Mat rgb;
    if (cur.elempack != 1 || cur.c != 3) {
        ncnn::Mat tmp;
        ncnn::convert_packing(cur, tmp, 1);
        rgb.create(cur.w, cur.h, 3);
        for (int k = 0; k < 3; k++) memcpy(rgb.channel(k), tmp.channel(k), (size_t)cur.w * cur.h * sizeof(float));
    } else {
        rgb = cur;
    }

    // ---- GLSL post: RGB -> host YUV ----
    if (gpu_post(c, rgb, out_format, out_matrix, out_range, out) != AJI_OK) {
        logmsg(c, 2, c->err.c_str()); return AJI_ERR;
    }
    return AJI_OK;
}

AJI_EXPORT uint64_t aji_flush(aji_ctx*, void*) { return 1; }
AJI_EXPORT int aji_done(aji_ctx*, uint64_t) { return 1; }
AJI_EXPORT int aji_wait(aji_ctx*, uint64_t) { return AJI_OK; }

AJI_EXPORT const char* aji_current_log(aji_ctx* c) { return c ? c->log.c_str() : ""; }
AJI_EXPORT int aji_scale_factor(aji_ctx* c) { return c ? c->chain_scale : 0; }
AJI_EXPORT int aji_rife_factor(aji_ctx*, int*, int*) { return 0; }
AJI_EXPORT int aji_rife_before_upscale(aji_ctx*) { return 1; }  // moot: aji_vk has no RIFE
AJI_EXPORT int aji_poll(aji_ctx*) { return 0; }

AJI_EXPORT int aji_infer_rife(aji_ctx* c, const aji_frame*, const aji_frame*, double,
                              const aji_frame*, void*) {
    if (c) c->err = "RIFE not supported in aji_vk yet (P7)";
    return AJI_ERR;
}

AJI_EXPORT const char* aji_last_error(aji_ctx* c) { return c ? c->err.c_str() : "null ctx"; }

AJI_EXPORT void aji_destroy(aji_ctx** c) {
    if (c && *c) { color_free((*c)->color); delete *c; *c = nullptr; }
}

} // extern "C"

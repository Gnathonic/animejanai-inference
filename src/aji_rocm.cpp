// aji_rocm — AMD ROCm/MIGraphX backend for the AnimeJaNai inference shim (libaji_rocm.so).
//
// Same ABI + structure as the (retired) ncnn-Vulkan backend, but the model runs on
// MIGraphX (AMD's TensorRT-equivalent graph compiler) instead of ncnn. Measured ~7-9x
// faster than ncnn-Vulkan on these SPAN models (e.g. HD Balanced 1080p->4K 20.5ms vs
// 115ms on a 9070 XT), TensorRT-class.
//
// Both modes of the aji.h ABI:
//   - direct mode  (engine_path=<model>.onnx): a single fixed upscaler.
//   - conf mode    (conf_path + model_dir + slot): resolution-conditional model CHAINS
//     with stacking, reusing aji_conf.cpp verbatim (same selection as aji_trt/aji_dml).
//
// aji_frame carries HOST YUV per aji_frame.format:
//   AJI_FMT_NV12: plane[0]=Y u8 (W*H), plane[1]=CbCr interleaved u8 (cw*ch*2).
//   AJI_FMT_P010: same but u16 (10-bit MSB-aligned, value<<6).
// aji_infer does the YUV<->RGB color on the CPU via resample.h (the zimg reference
// math, bit-exact; OpenMP over rows + reused ctx scratch), then the model chain on
// MIGraphX:
//   YUV(host) -> CPU pre -> RGB fp32 -> [fp16 cast -> MIGraphX -> fp16 out -> fp32]
//   -> [CPU Spline36 resize] -> CPU post -> YUV(host).
// (Moving color onto the GPU is a later optimization; with the model now ~10x faster,
//  CPU color is the next bottleneck at high res.)
//
// The model is the SAME fp16 .onnx the TensorRT/DirectML backends ship (NOT ncnn
// .param). MIGraphX compiles a per-(model,input-shape) engine; it is cached to a .mxr
// next to the model (or pre-compiled offline with migraphx-driver) so the ~17s compile
// happens once, not per launch.

#include "aji.h"
#include "aji_conf.h"
#include "resample.h"

#include <migraphx/migraphx.hpp>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>   // AJI_ROCM_TIMING per-stage instrumentation
#ifdef _OPENMP
#include <omp.h>
#endif

using aji_resample::weights;

// IEEE fp16 (binary-compatible with migraphx_shape_half_type / uint16 half bits). The
// SPAN .onnx declare fp16 input/output, so the model boundary converts fp32 RGB <-> fp16.
using half_t = _Float16;

// Minimal NCHW fp32 image buffer (replaces ncnn::Mat in the CPU color path): channel k
// is the contiguous w*h plane at offset k*w*h, i.e. the whole buffer is [R,G,B] planar.
struct RgbMat {
    int w = 0, h = 0, c = 0;
    std::vector<float> buf;
    // resize (NOT assign): every element is overwritten downstream, so zeroing is wasted;
    // and when this buffer is ctx-pooled and reused at a constant resolution, resize to the
    // same size is a no-op (no per-frame 100MB malloc/memset churn).
    void create(int W, int H, int C) { w = W; h = H; c = C; buf.resize((size_t)W * H * C); }
    float* channel(int k) { return buf.data() + (size_t)k * w * h; }
    const float* channel(int k) const { return buf.data() + (size_t)k * w * h; }
};

// A MIGraphX program compiled for one model at one input shape (keyed by path,in_w,in_h),
// with persistent device buffers so each frame is a bare eval + two hipMemcpys — no
// per-eval allocation (the offload_copy path's ~60ms/frame tax at 4K).
struct MgxModel {
    migraphx::program prog;
    // Every program parameter (input + "main:#output_0" [+ scratch]) gets one device
    // buffer, allocated once. Rebuilt into program_parameters each frame.
    struct Param { std::string name; void* dev = nullptr; size_t bytes = 0;
                   std::vector<std::size_t> dims; bool is_input = false; };
    std::vector<Param> params;
    std::string in_name = "input";
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    size_t out_bytes = 0;
    int scale = 0;
    ~MgxModel() { for (auto& p : params) if (p.dev) hipFree(p.dev); }
};

struct Step {
    enum Kind { RESIZE, MODEL } kind;
    int out_w = 0, out_h = 0;
    int model_idx = -1;   // MODEL
};

// Pinned (page-locked) host fp16 buffer for the model in/out copies — DMA hipMemcpy
// (~25GB/s) instead of pageable (~3GB/s), which is ~17ms/frame at 4K. Grows on demand.
struct PinnedHalf {
    half_t* ptr = nullptr; size_t cap = 0;
    half_t* get(size_t n) {
        if (n > cap) { if (ptr) hipHostFree(ptr);
            if (hipHostMalloc((void**)&ptr, n * sizeof(half_t)) != hipSuccess) ptr = nullptr;
            cap = ptr ? n : 0; }
        return ptr;
    }
    ~PinnedHalf() { if (ptr) hipHostFree(ptr); }
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
    std::vector<std::unique_ptr<MgxModel>> models;
    std::vector<Step> steps;
    int chain_scale = 0;          // product of model scales (for aji_scale_factor)
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    bool active = false;

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
        PinnedHalf mdl_in;                              // fp32 RGB -> fp16 model input (pinned)
        PinnedHalf mdl_out;                             // fp16 model output (dev->host, pinned)
        RgbMat rgbA, rgbB;                              // ping-pong RGB buffers (reused per frame)
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

// Compile (or load a cached) MIGraphX fp16 engine for one .onnx at a FIXED input shape.
// MIGraphX builds a static per-shape engine, so a model used at several resolutions in a
// chain gets one engine per resolution; each is cached to a .mxr next to the .onnx so the
// ~17s compile happens once (offline migraphx-driver pre-compilation drops it entirely).
static std::unique_ptr<MgxModel> load_model(aji_ctx* c, const std::string& onnx_path,
                                            int in_w, int in_h) {
    auto m = std::make_unique<MgxModel>();
    m->in_w = in_w; m->in_h = in_h;
    m->in_name = "input";   // the SPAN dynamo export's input blob name

    const std::string cache = onnx_path + "." + std::to_string(in_w) + "x" +
                              std::to_string(in_h) + ".dev.fp16.mxr";
    try {
        std::ifstream probe(cache, std::ios::binary);
        if (probe.good()) {
            m->prog = migraphx::load(cache.c_str());
        } else {
            migraphx::onnx_options oo;
            oo.set_input_parameter_shape(m->in_name, {1, 3, (size_t)in_h, (size_t)in_w});
            m->prog = migraphx::parse_onnx(onnx_path.c_str(), oo);
            migraphx::quantize_fp16(m->prog);
            migraphx::compile_options co; co.set_offload_copy(false);  // device-resident
            m->prog.compile(migraphx::target("gpu"), co);
            migraphx::save(m->prog, cache.c_str());
        }
    } catch (const std::exception& e) {
        c->err = "migraphx compile failed (" + onnx_path + "): " + e.what();
        return nullptr;
    }

    // One device buffer per program parameter (input + main:#output_0 [+ scratch]),
    // allocated once. The output is the 4D non-input param; scale = out_h / in_h.
    try {
        auto ps = m->prog.get_parameter_shapes();
        for (auto&& name : ps.names()) {
            auto sh = ps[name];
            MgxModel::Param p;
            p.name = name;
            p.bytes = sh.bytes();
            p.dims = sh.lengths();
            p.is_input = (p.name == m->in_name);
            if (hipMalloc(&p.dev, p.bytes) != hipSuccess || !p.dev) {
                c->err = "hipMalloc failed for param " + p.name; return nullptr;
            }
            if (!p.is_input && p.dims.size() == 4) {
                m->out_h = (int)p.dims[2]; m->out_w = (int)p.dims[3]; m->out_bytes = p.bytes;
            }
            m->params.push_back(std::move(p));
        }
        if (m->out_h <= 0 || m->out_h % in_h != 0) { c->err = "model output shape not a scale of input"; return nullptr; }
        m->scale = m->out_h / in_h;
    } catch (const std::exception& e) {
        c->err = "migraphx param setup failed: " + std::string(e.what());
        return nullptr;
    }
    return m;
}

// fit (sw,sh) into a (bw,bh) box preserving aspect (mirrors aji_trt fit_box).
static void fit_box(int sw, int sh, double bw, double bh, int* nw, int* nh) {
    double s = std::min(bw / sw, bh / sh);
    *nw = round_even(sw * s);
    *nh = round_even(sh * s);
}

// ------------------------- CPU color (resample.h) --------------------------

// GLSL pre: host YUV planes (Y full-res, U/V half-res, all as fp32 raw container
// values) -> RGB fp32 NCHW RgbMat (W x H x 3). Mirrors kernel_test/p010_test.
static int gpu_pre(aji_ctx* c, int W, int H, int format, int matrix, int range, int siting,
                   std::vector<float>& Yf, std::vector<float>& Uf, std::vector<float>& Vf,
                   RgbMat& rgb_out) {
    // CPU implementation using resample.h (the zimg reference math; bit-exact).
    // The GLSL kernels are validated at 256 but hit a subtle ncnn-Vulkan bug at
    // larger frames; since the chain already bridges through host RgbMat, CPU
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
static int gpu_resize(aji_ctx* c, const RgbMat& src, int DW, int DH, RgbMat& dst) {
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
static int gpu_post(aji_ctx* c, const RgbMat& rgb, int format, int matrix, int range,
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
        auto m = load_model(c, c->engine_path, cw, ch);
        if (!m) return AJI_ERR_ENGINE;
        int sc = m->scale;
        c->models.push_back(std::move(m));
        c->steps.push_back({Step::MODEL, w * sc, h * sc, 0});
        cw = w * sc; ch = h * sc;
        c->chain_scale = sc;
        c->log = "step 1: aji_rocm " + std::to_string(sc) + "x (MIGraphX)";
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
            std::string onnx = c->model_dir + "/" + m.name + ".onnx";
            auto nm = load_model(c, onnx, cw, ch);
            if (!nm) { logmsg(c, 2, c->err.c_str()); return AJI_ERR_ENGINE; }
            int sc = nm->scale;
            c->models.push_back(std::move(nm));
            cw *= sc; ch *= sc;
            c->steps.push_back({Step::MODEL, cw, ch, (int)c->models.size() - 1});
            c->chain_scale = c->chain_scale ? c->chain_scale * sc : sc;
            steps_log += "model " + m.name + " " + std::to_string(sc) + "x -> " + std::to_string(cw) + "x" + std::to_string(ch) + "; ";
        }
        if (chain->rife) steps_log += "(RIFE requested — not yet supported in aji_rocm); ";
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
        snprintf(msg, sizeof msg, "aji_rocm conf mode: %zu slots, slot=%d, model_dir=%s",
                 c->conf.slots.size(), c->slot, c->model_dir.c_str());
        logmsg(c, 1, msg);
    } else if (p->engine_path && p->engine_path[0]) {
        c->engine_path = p->engine_path;
        logmsg(c, 1, "aji_rocm direct mode");
    } else {
        c->err = "aji_rocm needs conf_path (conf mode) or engine_path (direct mode)";
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

    // TEMP: env-gated per-stage timing (AJI_ROCM_TIMING=1).
    static const bool kT = getenv("AJI_ROCM_TIMING") != nullptr;
    static double tPre=0, tModel=0, tPost=0; static long tN=0;
    auto _now = []{ return std::chrono::steady_clock::now(); };
    auto _ms  = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };
    auto _t0 = _now();

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

    // ---- CPU pre: YUV -> RGB fp32 NCHW (into a pooled ping-pong buffer) ----
    RgbMat* cur = &c->scratch.rgbA;
    RgbMat* alt = &c->scratch.rgbB;
    if (gpu_pre(c, W, H, format, matrix, range, siting, Yf, Uf, Vf, *cur) != AJI_OK) {
        logmsg(c, 2, c->err.c_str()); return AJI_ERR;
    }
    auto _t1 = _now();   // end of input color

    // ---- model chain + resize steps (RGB fp32 NCHW; ping-pong the two pooled buffers
    // so nothing reallocates per frame) ----
    for (const Step& st : c->steps) {
        if (st.kind == Step::RESIZE) {
            if (gpu_resize(c, *cur, st.out_w, st.out_h, *alt) != AJI_OK) { logmsg(c, 2, c->err.c_str()); return AJI_ERR; }
            std::swap(cur, alt);
        } else {
            MgxModel* nm = c->models[st.model_idx].get();
            const int IW = cur->w, IH = cur->h;
            const size_t n = (size_t)3 * IW * IH;
            // fp32 RGB NCHW -> fp16 (the model input dtype); pinned + -mf16c vectorized.
            half_t* ib = c->scratch.mdl_in.get(n);
            if (!ib) { c->err = "pinned host alloc failed (in)"; return AJI_ERR; }
            const float* cb = cur->buf.data();
            #pragma omp parallel for schedule(static) num_threads(c->nthreads)
            for (long i = 0; i < (long)n; i++) ib[i] = (half_t)cb[i];

            // Device-resident eval: copy fp16 input to its persistent device buffer, bind
            // every param's device buffer, run (writes into main:#output_0), copy back.
            try {
                migraphx::program_parameters pp;
                for (auto& p : nm->params) {
                    if (p.is_input) {
                        if (hipMemcpy(p.dev, ib, p.bytes, hipMemcpyHostToDevice) != hipSuccess) {
                            c->err = "hipMemcpy H2D failed"; return AJI_ERR;
                        }
                    }
                    migraphx::shape sh(migraphx_shape_half_type, p.dims);
                    pp.add(p.name.c_str(), migraphx::argument(sh, p.dev));
                }
                auto _ge = std::chrono::steady_clock::now();
                auto outs = nm->prog.eval(pp);
                hipDeviceSynchronize();   // eval is async on the GPU stream; wait before D2H
                if (getenv("AJI_ROCM_TIMING")) { static double te=0; static long ne=0;
                    te += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_ge).count();
                    if (++ne % 50 == 0) fprintf(stderr, "[AJI_ROCM_TIMING/eval] eval+sync=%.1fms\n", te/ne); }
                auto osh = outs[0].get_shape().lengths();   // {1,3,OH,OW}
                if (osh.size() < 4) { c->err = "infer: bad output rank"; return AJI_ERR_SHAPE; }
                const int OW = (int)osh[3], OH = (int)osh[2];
                half_t* od = c->scratch.mdl_out.get((size_t)3 * OW * OH);
                if (!od) { c->err = "pinned host alloc failed (out)"; return AJI_ERR; }
                if (hipMemcpy(od, outs[0].data(), nm->out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
                    c->err = "hipMemcpy D2H failed"; return AJI_ERR;
                }
                alt->create(OW, OH, 3);
                float* ob = alt->buf.data(); const long on = (long)3 * OW * OH;
                #pragma omp parallel for schedule(static) num_threads(c->nthreads)
                for (long i = 0; i < on; i++) ob[i] = (float)od[i];
                std::swap(cur, alt);
            } catch (const std::exception& e) {
                c->err = std::string("infer eval failed: ") + e.what(); return AJI_ERR;
            }
        }
    }
    if (cur->w != c->out_w || cur->h != c->out_h || cur->c < 3) { c->err = "infer: unexpected output dims"; return AJI_ERR_SHAPE; }
    auto _t2 = _now();   // end of model chain

    // ---- CPU post: RGB fp32 NCHW -> host YUV ----
    if (gpu_post(c, *cur, out_format, out_matrix, out_range, out) != AJI_OK) {
        logmsg(c, 2, c->err.c_str()); return AJI_ERR;
    }
    if (kT) {
        tPre += _ms(_t0,_t1); tModel += _ms(_t1,_t2); tPost += _ms(_t2,_now()); tN++;
        if (tN % 50 == 0) {
            double tot = tPre+tModel+tPost;
            fprintf(stderr, "[AJI_ROCM_TIMING] n=%ld  in-color=%.1fms (%.0f%%)  "
                "model=%.1fms (%.0f%%)  out-color=%.1fms (%.0f%%)  total=%.1fms => %.1f fps\n",
                tN, tPre/tN, 100*tPre/tot, tModel/tN, 100*tModel/tot, tPost/tN, 100*tPost/tot,
                tot/tN, 1000.0/(tot/tN));
        }
    }
    return AJI_OK;
}

AJI_EXPORT uint64_t aji_flush(aji_ctx*, void*) { return 1; }
AJI_EXPORT int aji_done(aji_ctx*, uint64_t) { return 1; }
AJI_EXPORT int aji_wait(aji_ctx*, uint64_t) { return AJI_OK; }

AJI_EXPORT const char* aji_current_log(aji_ctx* c) { return c ? c->log.c_str() : ""; }
AJI_EXPORT int aji_scale_factor(aji_ctx* c) { return c ? c->chain_scale : 0; }
AJI_EXPORT int aji_rife_factor(aji_ctx*, int*, int*) { return 0; }
AJI_EXPORT int aji_rife_before_upscale(aji_ctx*) { return 1; }  // moot: aji_rocm has no RIFE
AJI_EXPORT int aji_poll(aji_ctx*) { return 0; }

AJI_EXPORT int aji_infer_rife(aji_ctx* c, const aji_frame*, const aji_frame*, double,
                              const aji_frame*, void*) {
    if (c) c->err = "RIFE not supported in aji_rocm yet (P7)";
    return AJI_ERR;
}

AJI_EXPORT const char* aji_last_error(aji_ctx* c) { return c ? c->err.c_str() : "null ctx"; }

AJI_EXPORT void aji_destroy(aji_ctx** c) {
    if (c && *c) { delete *c; *c = nullptr; }
}

} // extern "C"

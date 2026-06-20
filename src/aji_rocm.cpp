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
#include "aji_rocm_color.h"   // GPU-resident color kernels (Phase B)
#include "rife_cpu.h"

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
#include <atomic>   // async engine compile (async_build)
#include <thread>
#include <mutex>    // async inference pipeline (P4 overlap)
#include <condition_variable>
#include <deque>
#include <set>
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

// One background MIGraphX compile. MIGraphX builds a static per-resolution
// engine in ~140s; blocking aji_configure on that would freeze the player, so
// an uncached model compiles on a worker thread while the chain runs
// passthrough, and aji_poll() reports completion (the filter then reconfigures
// and the now-cached .mxr loads in ~1s). The worker captures a shared_ptr to
// this — NOT the aji_ctx — so it never touches ctx state and aji_destroy can
// detach it without a use-after-free or a shutdown hang. Mirrors aji_trt's
// async build, so the existing engine_monitor.lua OSD drives both backends.
struct BuildState {
    std::string onnx, key;     // .onnx path and its .mxr cache path (the failed-set key)
    int w = 0, h = 0;          // input shape the engine is compiled for
    std::atomic<int> done{0};  // 0 running, 1 finished (success or failure)
    std::atomic<bool> ok{false};
    std::atomic<bool> reported{false};  // aji_poll has returned 1 for this build
    std::string err;           // set by the worker; read by main only after done==1
};

// GPU-resident out-color state (Phase B): device weight tables + scratch + output planes,
// allocated once per (out W,H,format,matrix,range). The model's fp16 RGB output is colored
// to NV12/P010 on the GPU here, so the 4K RGB buffer never leaves the device.
struct GpuColor {
    bool ready = false;
    int W = 0, H = 0, fmt = 0, mat = 0, rng = 0;   // what `ready` was built for
    aji_color_csp csp{};
    int*   ph_start = nullptr; float* ph_wt = nullptr; int ph_taps = 0;  // W->cw downsample
    int*   pv_start = nullptr; float* pv_wt = nullptr; int pv_taps = 0;  // H->ch downsample
    float* Un = nullptr; float* Vn = nullptr;      // device scratch, W*H
    float* hu = nullptr; float* hv = nullptr;      // device scratch, cw*H
    void*  yplane = nullptr; void* uvplane = nullptr;  // device output (tight)
    void free() {
        for (void* p : {(void*)ph_start,(void*)ph_wt,(void*)pv_start,(void*)pv_wt,
                        (void*)Un,(void*)Vn,(void*)hu,(void*)hv,yplane,uvplane})
            if (p) hipFree(p);
        ph_start=nullptr; ph_wt=nullptr; pv_start=nullptr; pv_wt=nullptr;
        Un=Vn=hu=hv=nullptr; yplane=uvplane=nullptr; ready=false;
    }
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

    // CPU-color OpenMP width. nthreads is the small-frame cap (spawn/sync overhead of
    // many threads outweighs the work on small frames: 8 ~= peak, 32 < 1); nthreads_max
    // is the box's width, used for LARGE frames (4K post is ~8M px and scales well past
    // 8 cores). color_nt() picks per-call by pixel count.
    int nthreads = 1;
    int nthreads_max = 1;

    std::string err, log;
    aji_log_fn logfn = nullptr;
    void* log_opaque = nullptr;

    // async engine compile (filter passes async_build=1; CLI/benchmark leaves it 0).
    bool async_build = false;
    std::shared_ptr<BuildState> build;       // current/last background compile, or null
    std::thread build_thread;
    std::set<std::string> failed_builds;     // .mxr keys that failed; don't re-kick (avoids a loop)

    // Async inference pipeline (P4 overlap). aji_infer = submit: CPU in-color (gpu_pre, on
    // the filter thread) -> enqueue; a worker thread runs the GPU model chain; aji_wait =
    // D2H + CPU out-color (gpu_post, on the filter thread). With the filter pipelining at
    // depth>1, out-color(N) overlaps eval(N+1). Per-slot rgb_in/rgb_out; the chain scratch
    // (rgbA/rgbB/rt0/pinned/device) is worker-only-serial so it stays shared; gpu_pre and
    // gpu_post scratch are filter-thread-only and disjoint from the worker -> no color race.
    enum { SLOT_FREE = 0, SLOT_SUBMITTED = 1, SLOT_DONE = 2 };
    struct InferSlot {
        RgbMat rgb_in, rgb_out;     // in-color output / chain output
        aji_frame out{};            // where out-color writes (planes valid until wait, via filter outq ref)
        int ofmt = 0, omat = 0, orng = 0;
        uint64_t ticket = 0;
        int state = SLOT_FREE;
        int err = AJI_OK;
        bool gpu_colored = false;   // worker already wrote out (GPU out-color); wait skips gpu_post
        std::string errmsg;
    };
    static const int kRing = 4;     // in-flight depth cap (filter depth is small; submit back-pressures if full)
    InferSlot slots[kRing];
    std::thread infer_worker;
    std::mutex infer_mtx;
    std::condition_variable infer_cv_worker, infer_cv_main;
    std::deque<int> infer_pending;  // slot indices awaiting the worker
    uint64_t infer_last_ticket = 0;
    bool infer_worker_started = false;
    bool infer_stop = false;

    // GPU-resident out-color (Phase B). Eligible = a single MODEL step, no RESIZE (the
    // 1080p->4K case); gated by AJI_ROCM_GPU_COLOR. gc holds the device state.
    bool gpu_color_eligible = false;
    GpuColor gc;

    // RIFE interpolation state. Populated by setup_rife (A8); used by aji_infer_rife (A9).
    struct RifeState {
        std::unique_ptr<MgxModel> model;
        rife_cpu::Geom g{};
        int    num = 1, den = 1;
        double scd_threshold = 0.150;
        bool   before_upscale = true, enabled = false, loaded = false;
        void*  dev_in = nullptr; void* dev_out = nullptr;     // 11*plane*2 / 3*plane*2 bytes
        PinnedHalf pin_in, pin_out;                            // dedicated pinned host fp16
        std::vector<float> assembly;                           // 11*plane fp32 (consts + per-frame)
        // dedicated color scratch (resample plans + temp planes) added in A5 wiring

        // Free the device tensors (the MgxModel and the pinned/assembly buffers free
        // themselves via their own dtors). Called on reconfigure and from ~RifeState.
        void free_dev() {
            if (dev_in)  { hipFree(dev_in);  dev_in  = nullptr; }
            if (dev_out) { hipFree(dev_out); dev_out = nullptr; }
        }
        // Reconfigure-safe reset: drop the model + device buffers and disable, so stale
        // RIFE state never leaks across an aji_configure that changes/removes RIFE.
        void reset() {
            free_dev();
            model.reset();
            assembly.clear();
            enabled = loaded = false;
        }
        ~RifeState() { free_dev(); }
    };
    RifeState rife;
    std::string rife_model_dir;
    std::mutex gpu_eval_mtx;
};

static void logmsg(aji_ctx* c, int level, const char* m) {
    if (c && c->logfn) c->logfn(c->log_opaque, level, m);
}

static int round_even(double x) {
    int r = (int)(x + 0.5);
    return r;
}

// OpenMP width for a CPU-color/cast pass. The resample/color is memory-bandwidth-bound,
// so threads past ~8 just contend on the memory bus and SLOW it down (measured: scaling a
// 4K post pass to 32 threads regressed wall fps 14->12). So the historical cap holds even
// for 4K; nthreads_max is kept only for diagnostics. The real lever for color is moving it
// onto the GPU (P4), not more CPU threads.
static inline int color_nt(const aji_ctx* c, long /*npix*/) { return c->nthreads; }

// The cache file for one engine: <onnx>.<W>x<H>.c<channels>.dev.mlir.fp16.mxr next to the
// model. The channel count is folded into the key so an 11-ch RIFE engine at the same
// resolution can't collide with a 3-ch upscaler engine. The ".mlir" tag invalidates the
// older MLIR-disabled engines, which had a non-deterministic MIOpen-fallback conv artifact
// (evenly-spaced column static) at 4K.
static std::string mxr_cache_path(const std::string& onnx, int w, int h, int channels = 3) {
    return onnx + "." + std::to_string(w) + "x" + std::to_string(h)
           + ".c" + std::to_string(channels) + ".dev.mlir.fp16.mxr";
}
static bool mxr_cached(const std::string& onnx, int w, int h, int channels = 3) {
    std::ifstream probe(mxr_cache_path(onnx, w, h, channels), std::ios::binary);
    return probe.good();
}

// Parse + fp16-quantize + GPU-compile one .onnx at a FIXED input shape and save the
// engine. Touches NO ctx state, so it is safe to run on a worker thread. Writes to a
// temp file then atomically renames into place, so an interrupted compile (player quit
// mid-build) never leaves a half-written .mxr that would later fail to load.
// Returns false + *errout on error.
static bool compile_mxr(const std::string& onnx_path, int in_w, int in_h, std::string* errout,
                        int in_channels = 3) {
    try {
        // MLIR (rocMLIR) is the DEFAULT conv codegen on RDNA and is REQUIRED for correctness:
        // disabling it falls back to a MIOpen conv solver that reads uninitialized workspace at
        // 4K and yields non-deterministic, evenly-spaced column static (confirmed: MLIR-off
        // differs every run, MLIR-on is bit-identical). It costs ~70s more compile (one-time,
        // behind the progress bar) and ZERO inference fps (measured 32.4 vs 32.5ms). So leave
        // MLIR on (the 2.15 default); MIGRAPHX_DISABLE_MLIR=1 in the env still forces it off for
        // experiments. (Dynamic-shape compile is still impossible: the SPAN reflect-pad preamble
        // has non-constant pads MIGraphX can't parse dynamically.)
        migraphx::onnx_options oo;
        oo.set_input_parameter_shape("input", {1, (size_t)in_channels, (size_t)in_h, (size_t)in_w});
        auto prog = migraphx::parse_onnx(onnx_path.c_str(), oo);
        migraphx::quantize_fp16(prog);
        migraphx::compile_options co; co.set_offload_copy(false);  // device-resident
        prog.compile(migraphx::target("gpu"), co);
        const std::string cache = mxr_cache_path(onnx_path, in_w, in_h, in_channels);
        const std::string tmp = cache + ".tmp." + std::to_string(in_w) + "x" + std::to_string(in_h);
        migraphx::save(prog, tmp.c_str());
        if (std::rename(tmp.c_str(), cache.c_str()) != 0) {
            std::remove(tmp.c_str());
            throw std::runtime_error("could not rename engine into place");
        }
        return true;
    } catch (const std::exception& e) {
        if (errout) *errout = std::string("migraphx compile failed (") + onnx_path + "): " + e.what();
        return false;
    }
}

// Kick a background compile of one engine; the configure that called this returns
// passthrough and aji_poll() reports when to reconfigure. The worker captures only a
// shared_ptr to the BuildState (never the ctx), so it outlives the ctx safely.
static void start_async_build(aji_ctx* c, const std::string& onnx, int w, int h,
                              int in_channels = 3) {
    if (c->build_thread.joinable()) c->build_thread.join();  // reap a previous finished worker
    auto bs = std::make_shared<BuildState>();
    bs->onnx = onnx; bs->key = mxr_cache_path(onnx, w, h, in_channels); bs->w = w; bs->h = h;
    c->build = bs;
    c->build_thread = std::thread([bs, in_channels]() {
        std::string e;
        bool ok = compile_mxr(bs->onnx, bs->w, bs->h, &e, in_channels);
        bs->err = e;
        bs->ok.store(ok);
        bs->done.store(1);   // release: main reads ok/err only after seeing done==1
    });
}

// Load an engine for one .onnx at a FIXED input shape into an MgxModel. The .mxr must
// already be on disk (the async path defers an uncached model before reaching here; the
// sync path compiles it inline first). MIGraphX builds a static per-shape engine, so a
// model used at several resolutions in a chain gets one engine per resolution.
// in_channels == 11 selects the RIFE path: output is the 3-channel non-input 4D param,
// scale is fixed at 1, and both shapes are hard-asserted.
static std::unique_ptr<MgxModel> load_model(aji_ctx* c, const std::string& onnx_path,
                                            int in_w, int in_h, int in_channels = 3) {
    auto m = std::make_unique<MgxModel>();
    m->in_w = in_w; m->in_h = in_h;
    m->in_name = "input";   // the SPAN dynamo export's input blob name; same for RIFE

    const std::string cache = mxr_cache_path(onnx_path, in_w, in_h, in_channels);
    try {
        if (!mxr_cached(onnx_path, in_w, in_h, in_channels)) {
            // sync (CLI/benchmark) path: async_build defers before getting here
            std::string e;
            if (!compile_mxr(onnx_path, in_w, in_h, &e, in_channels)) { c->err = e; return nullptr; }
        }
        m->prog = migraphx::load(cache.c_str());
    } catch (const std::exception& e) {
        c->err = "migraphx engine load failed (" + cache + "): " + e.what();
        return nullptr;
    }

    // One device buffer per program parameter (input + main:#output_0 [+ scratch]),
    // allocated once. For upscale (in_channels==3): the output is the last 4D non-input
    // param; scale = out_h / in_h. For RIFE (in_channels==11): the output is the non-input
    // 4D param whose second dimension is 3; shapes are hard-asserted.
    const bool is_rife = (in_channels == 11);
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
            // Zero every device buffer once. With offload_copy=false MIGraphX gives us the
            // "scratch" workspace param to own; some conv solvers read padding/edge regions
            // of it that they never write, so an uninitialized (hipMalloc) buffer yields
            // non-deterministic garbage in evenly-spaced column bands. Zeroing makes those
            // reads a deterministic 0 (the correct pad value) and kills the artifact.
            hipMemset(p.dev, 0, p.bytes);
            if (getenv("AJI_ROCM_GPU_COLOR_DEBUG"))
                fprintf(stderr, "[param] %s  bytes=%zu  is_input=%d\n", p.name.c_str(), p.bytes, (int)p.is_input);
            if (!is_rife) {
                // Upscale path: last-wins 4D non-input param is the output.
                if (!p.is_input && p.dims.size() == 4) {
                    m->out_h = (int)p.dims[2]; m->out_w = (int)p.dims[3]; m->out_bytes = p.bytes;
                }
            } else {
                // RIFE path: output is the FIRST non-input 4D param with dims[1]==3.
                // Guard with out_h==0 so a later scratch param with the same channel
                // count can't overwrite the real output latch.
                if (!p.is_input && p.dims.size() == 4 && p.dims[1] == 3 && m->out_h == 0) {
                    m->out_h = (int)p.dims[2]; m->out_w = (int)p.dims[3]; m->out_bytes = p.bytes;
                }
            }
            m->params.push_back(std::move(p));
        }

        if (!is_rife) {
            // Upscale: output must be an integer scale of the input.
            if (m->out_h <= 0 || m->out_h % in_h != 0) { c->err = "model output shape not a scale of input"; return nullptr; }
            m->scale = m->out_h / in_h;
        } else {
            // RIFE: assert input {1,11,in_h,in_w} and output {1,3,in_h,in_w}.
            // Silently miscompiled dynamic-channel graphs abort at Concat, so fail loudly here.
            bool input_ok = false, output_ok = false;
            for (auto& p : m->params) {
                if (p.is_input && p.dims.size() == 4 &&
                    p.dims[0] == 1 && p.dims[1] == 11 &&
                    p.dims[2] == (size_t)in_h && p.dims[3] == (size_t)in_w)
                    input_ok = true;
                if (!p.is_input && p.dims.size() == 4 &&
                    p.dims[0] == 1 && p.dims[1] == 3 &&
                    p.dims[2] == (size_t)in_h && p.dims[3] == (size_t)in_w)
                    output_ok = true;
            }
            if (!input_ok) {
                c->err = "RIFE engine: input param is not {1,11," + std::to_string(in_h)
                         + "," + std::to_string(in_w) + "} — wrong model or shape";
                return nullptr;
            }
            if (!output_ok) {
                c->err = "RIFE engine: output param is not {1,3," + std::to_string(in_h)
                         + "," + std::to_string(in_w) + "} — wrong model or shape";
                return nullptr;
            }
            m->scale = 1;  // RIFE is temporal interpolation, not spatial upscale
        }
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
    const int NT = color_nt(c, (long)W * H);
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
    const int NT = color_nt(c, (long)DW * DH);
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
    const int W = rgb.w, H = rgb.h;
    const int NT = color_nt(c, (long)W * H);
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

// Ensure the engine for one model step is ready. In async_build mode an uncached engine
// is compiled on a worker thread and this returns 0 (the chain runs passthrough until
// aji_poll() triggers a reconfigure); a build that already failed is not re-kicked. On
// success the model is pushed into c->models and its scale (>0) returned. Returns -1 on a
// hard load error. On defer/failure it sets a marker in c->log for engine_monitor.lua.
// Returns: >0 scale (loaded), 0 deferred to a background build, -1 error.
// in_channels defaults to 3 (upscale path); pass 11 for RIFE so the cache key and
// compile both use the correct channel count.
static int ensure_model(aji_ctx* c, const std::string& onnx, const std::string& name,
                        int in_w, int in_h, int in_channels = 3) {
    if (c->async_build && !mxr_cached(onnx, in_w, in_h, in_channels)) {
        const std::string key = mxr_cache_path(onnx, in_w, in_h, in_channels);
        char res[32]; snprintf(res, sizeof res, "%dx%d", in_w, in_h);
        if (c->failed_builds.count(key)) {
            c->log = "MIGraphX engine build FAILED for " + name + " for " + res +
                     "; playing without this chain";
            return 0;
        }
        // One build at a time: a multi-engine chain cascades through repeated
        // poll() -> reconfigure cycles, building the next uncached engine each time.
        bool building = c->build && c->build->done.load() == 0;
        if (!building) start_async_build(c, onnx, in_w, in_h, in_channels);
        c->log = "Building MIGraphX engine for " + name + " for " + res +
                 " (first play at this resolution)";
        return 0;
    }
    auto m = load_model(c, onnx, in_w, in_h, in_channels);
    if (!m) { logmsg(c, 2, c->err.c_str()); return -1; }
    int sc = m->scale;
    c->models.push_back(std::move(m));
    return sc;
}

// Configure RIFE interpolation for the active chain (called from build_plan, replacing
// the old "not supported" drop). Mirrors ensure_model's async-build cascade, but the
// compiled engine lands in c->rife.model (NOT c->models — it is driven by aji_infer_rife,
// not the upscale step loop) and there is no spatial scale. The RIFE .mxr rides the SAME
// single-build slot as the upscale engines, so a cold-cache upscale+RIFE chain cascades
// one compile per poll->reconfigure cycle.
//   LOADED   — engine on disk + loaded; rife.enabled && rife.loaded; interpolation live.
//   DEFERRED — engine compiling on the background worker; passthrough (no interp) until
//              cached; build_plan must return 0 (c->log carries the build marker).
//   DISABLED — RIFE off (no model dir, invalid model code, or a load/shape error); the
//              chain continues without interpolation. Never hard-errors the chain.
enum RifeSetup { RIFE_LOADED, RIFE_DEFERRED, RIFE_DISABLED };
static RifeSetup setup_rife(aji_ctx* c, const AjiChainConf* chain,
                            int src_w, int src_h, int cw, int ch,
                            double /*fps*/, std::string& steps_log) {
    // Reconfigure-safe: drop any prior RIFE engine/buffers before reconfiguring.
    c->rife.reset();

    // 1. Geometry — RIFE compiles at the SOURCE shape when before_upscale (interpolate
    // then upscale every frame), else at the chain-output shape.
    int rw = chain->rife_before_upscale ? src_w : cw;
    int rh = chain->rife_before_upscale ? src_h : ch;
    c->rife.g = rife_cpu::geometry(rw, rh);
    const int pw = c->rife.g.pw, ph = c->rife.g.ph;

    // 2. Model name. Invalid code or no model dir -> disable (chain continues, no interp).
    std::string name = rife_cpu::model_name(chain->rife_model, chain->rife_ensemble);
    if (name.empty() || c->rife_model_dir.empty()) {
        steps_log += "(RIFE disabled: " +
                     std::string(name.empty() ? "invalid rife_model code" : "no rife_model_dir") +
                     "); ";
        c->rife.enabled = false;
        return RIFE_DISABLED;
    }
    const std::string onnx = c->rife_model_dir + "/" + name + ".onnx";

    // 3. Async-build cascade (mirror ensure_model): an uncached RIFE engine compiles on
    // the single background build slot; the chain runs passthrough (no interp) meanwhile.
    if (c->async_build && !mxr_cached(onnx, pw, ph, 11)) {
        const std::string key = mxr_cache_path(onnx, pw, ph, 11);
        char res[32]; snprintf(res, sizeof res, "%dx%d", pw, ph);
        if (c->failed_builds.count(key)) {
            c->log = "MIGraphX engine build FAILED for " + name + " for " + res +
                     "; playing without interpolation";
            // A failed RIFE build degrades to no-interp passthrough, not a deferred
            // reconfigure: treat as DISABLED so build_plan keeps building the chain.
            c->rife.enabled = false;
            return RIFE_DISABLED;
        }
        bool building = c->build && c->build->done.load() == 0;
        if (!building) start_async_build(c, onnx, pw, ph, 11);
        c->log = "Building MIGraphX engine for " + name + " for " + res +
                 " (first play at this resolution)";
        // before_upscale is set now (not just at LOAD) so aji_rife_before_upscale reports
        // the right value even while deferred; factor still returns 0 until loaded.
        c->rife.before_upscale = chain->rife_before_upscale;
        c->rife.enabled = true; c->rife.loaded = false;
        return RIFE_DEFERRED;
    }

    // 4. Cached (or sync build inline): load the RIFE engine. load_model hard-asserts the
    // {1,11,ph,pw} input / {1,3,ph,pw} output shapes for channels==11.
    auto m = load_model(c, onnx, pw, ph, 11);
    if (!m) {
        logmsg(c, 2, c->err.c_str());
        steps_log += "(RIFE disabled: engine load failed); ";
        c->rife.enabled = false;
        return RIFE_DISABLED;
    }
    c->rife.model = std::move(m);

    // 5. Dedicated device tensors (fp16) + pinned host staging. free_dev was already
    // called by reset() above, so no double-allocation.
    const size_t plane = (size_t)pw * ph;
    if (hipMalloc(&c->rife.dev_in,  11 * plane * 2) != hipSuccess || !c->rife.dev_in ||
        hipMalloc(&c->rife.dev_out,  3 * plane * 2) != hipSuccess || !c->rife.dev_out) {
        c->err = "RIFE: hipMalloc failed for device tensors";
        logmsg(c, 2, c->err.c_str());
        c->rife.reset();
        steps_log += "(RIFE disabled: device alloc failed); ";
        return RIFE_DISABLED;
    }
    c->rife.pin_in.get(11 * plane);
    c->rife.pin_out.get(3 * plane);

    // 6. Host const template: ch7-10 (mesh/multiplier) filled once; A9 reuses this per
    // frame and writes ch0-6 (frames + timestep) on top.
    c->rife.assembly.assign(11 * plane, 0.f);
    rife_cpu::fill_consts(c->rife.assembly.data(), pw, ph);

    // 7. Carry chain config into the runtime state and mark live.
    c->rife.num = chain->rife_factor_num;
    c->rife.den = chain->rife_factor_den;
    c->rife.scd_threshold = chain->rife_scd_threshold;
    c->rife.before_upscale = chain->rife_before_upscale;
    c->rife.enabled = true;
    c->rife.loaded = true;

    steps_log += "RIFE " + name + " " + std::to_string(chain->rife_factor_num) + "/" +
                 std::to_string(chain->rife_factor_den) + " interp at " +
                 std::to_string(rw) + "x" + std::to_string(rh) + " (padded " +
                 std::to_string(pw) + "x" + std::to_string(ph) + ", " +
                 (chain->rife_before_upscale ? "pre-upscale" : "post-upscale") + "); ";
    return RIFE_LOADED;
}

// Build the step plan for the active chain (conf mode) or single model (direct).
static int build_plan(aji_ctx* c, int w, int h, double fps) {
    c->models.clear();
    c->steps.clear();
    c->chain_scale = 0;
    c->gpu_color_eligible = false;
    // Drop any prior RIFE state up front so a reconfigure that removes RIFE (direct mode,
    // no chain, or a chain without rife=) doesn't leak a stale engine/buffers. setup_rife
    // re-enables it (and re-resets) when the new chain requests RIFE.
    c->rife.reset();
    int cw = w, ch = h;

    if (!c->conf_mode) {
        int sc = ensure_model(c, c->engine_path, "engine", cw, ch);
        if (sc < 0) return AJI_ERR_ENGINE;
        if (sc == 0) {  // compiling in the background (or build failed) -> passthrough
            c->in_w = w; c->in_h = h; c->out_w = w; c->out_h = h; c->active = false;
            return 0;
        }
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
            int sc = ensure_model(c, onnx, m.name, cw, ch);
            if (sc < 0) return AJI_ERR_ENGINE;
            if (sc == 0) {  // compiling in the background (or build failed) -> passthrough
                c->steps.clear(); c->models.clear();
                c->in_w = w; c->in_h = h; c->out_w = w; c->out_h = h; c->active = false;
                return 0;   // c->log already carries the build marker
            }
            cw *= sc; ch *= sc;
            c->steps.push_back({Step::MODEL, cw, ch, (int)c->models.size() - 1});
            c->chain_scale = c->chain_scale ? c->chain_scale * sc : sc;
            steps_log += "model " + m.name + " " + std::to_string(sc) + "x -> " + std::to_string(cw) + "x" + std::to_string(ch) + "; ";
        }
        // RIFE runs BEFORE the c->active computation below so a RIFE-only chain (zero
        // upscale models, steps empty -> active=false) still configures interpolation.
        if (chain->rife) {
            RifeSetup rs = setup_rife(c, chain, w, h, cw, ch, fps, steps_log);
            if (rs == RIFE_DEFERRED) {
                // RIFE engine compiling in the background (c->log carries the build
                // marker). Mirror the upscale defer at :719-720: passthrough, no active,
                // until aji_poll() triggers a reconfigure that finds the cached .mxr.
                c->steps.clear(); c->models.clear();
                c->in_w = w; c->in_h = h; c->out_w = w; c->out_h = h; c->active = false;
                return 0;
            }
            // LOADED or DISABLED: fall through and finish the plan normally.
        }
        char hdr[160];
        snprintf(hdr, sizeof hdr, "slot %d chain %d: %dx%d -> %dx%d  [", c->slot, chain->index, w, h, cw, ch);
        c->log = std::string(hdr) + steps_log + "]";
    }

    c->in_w = w; c->in_h = h; c->out_w = cw; c->out_h = ch;
    c->active = !c->steps.empty();
    // GPU-resident out-color applies to a single-model no-resize chain (the 1080p->4K case);
    // multi-model/resize chains keep the CPU path. On by default (bit-exact vs the CPU path,
    // verified maxdiff=0 at 720p; ~1.5x at 4K); opt out with AJI_ROCM_GPU_COLOR=0.
    static const bool gpu_color_off = []{ const char* e = getenv("AJI_ROCM_GPU_COLOR"); return e && e[0] == '0'; }();
    c->gpu_color_eligible = !gpu_color_off && c->active &&
                            c->steps.size() == 1 && c->steps[0].kind == Step::MODEL;
    return c->active ? 1 : 0;
}

// -------------------------- async inference pipeline -----------------------

// Run the model+resize chain on the WORKER thread: rgb_in (RGB fp32 NCHW) -> rgb_out
// (c->out_w x c->out_h). Ping-pongs between rgb_in (the slot's own buffer, free to reuse
// after the first step reads it) and the worker-serial scratch rgbA; the final result is
// O(1)-swapped into rgb_out. Uses only worker-owned scratch (rgbA, rt0 via gpu_resize,
// mdl_in/mdl_out, device buffers) — never the filter-thread color scratch. Returns AJI_OK
// or an error code (+ *errmsg). Mirrors the old synchronous chain body exactly.
static int run_chain(aji_ctx* c, RgbMat& rgb_in, RgbMat& rgb_out, std::string* errmsg) {
    RgbMat* cur = &rgb_in;
    RgbMat* alt = &c->scratch.rgbA;
    for (const Step& st : c->steps) {
        if (st.kind == Step::RESIZE) {
            if (gpu_resize(c, *cur, st.out_w, st.out_h, *alt) != AJI_OK) { if (errmsg) *errmsg = c->err; return AJI_ERR; }
            std::swap(cur, alt);
        } else {
            MgxModel* nm = c->models[st.model_idx].get();
            const int IW = cur->w, IH = cur->h;
            const size_t n = (size_t)3 * IW * IH;
            half_t* ib = c->scratch.mdl_in.get(n);
            if (!ib) { if (errmsg) *errmsg = "pinned host alloc failed (in)"; return AJI_ERR; }
            const float* cb = cur->buf.data();
            #pragma omp parallel for schedule(static) num_threads(color_nt(c, (long)n))
            for (long i = 0; i < (long)n; i++) ib[i] = (half_t)cb[i];
            try {
                migraphx::program_parameters pp;
                for (auto& p : nm->params) {
                    if (p.is_input) {
                        if (hipMemcpy(p.dev, ib, p.bytes, hipMemcpyHostToDevice) != hipSuccess) {
                            if (errmsg) *errmsg = "hipMemcpy H2D failed"; return AJI_ERR;
                        }
                    }
                    migraphx::shape sh(migraphx_shape_half_type, p.dims);
                    pp.add(p.name.c_str(), migraphx::argument(sh, p.dev));
                }
                auto _ge = std::chrono::steady_clock::now();
                migraphx::argument outs_arg;
                {
                    std::lock_guard<std::mutex> lk(c->gpu_eval_mtx);
                    auto outs = nm->prog.eval(pp);   // run the inference graph (MIGraphX API, not code-eval)
                    hipDeviceSynchronize();           // eval is async on the GPU stream; wait before D2H
                    outs_arg = outs[0];
                }
                if (getenv("AJI_ROCM_TIMING")) { static double te=0; static long ne=0;
                    te += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_ge).count();
                    if (++ne % 50 == 0) fprintf(stderr, "[AJI_ROCM_TIMING/eval] eval+sync=%.1fms\n", te/ne); }
                auto osh = outs_arg.get_shape().lengths();
                if (osh.size() < 4) { if (errmsg) *errmsg = "bad output rank"; return AJI_ERR_SHAPE; }
                const int OW = (int)osh[3], OH = (int)osh[2];
                half_t* od = c->scratch.mdl_out.get((size_t)3 * OW * OH);
                if (!od) { if (errmsg) *errmsg = "pinned host alloc failed (out)"; return AJI_ERR; }
                if (hipMemcpy(od, outs_arg.data(), nm->out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
                    if (errmsg) *errmsg = "hipMemcpy D2H failed"; return AJI_ERR;
                }
                alt->create(OW, OH, 3);
                float* ob = alt->buf.data(); const long on = (long)3 * OW * OH;
                #pragma omp parallel for schedule(static) num_threads(color_nt(c, on))
                for (long i = 0; i < on; i++) ob[i] = (float)od[i];
                std::swap(cur, alt);
            } catch (const std::exception& e) {
                if (errmsg) *errmsg = std::string("infer eval failed: ") + e.what(); return AJI_ERR;
            }
        }
    }
    if (cur->w != c->out_w || cur->h != c->out_h || cur->c < 3) { if (errmsg) *errmsg = "unexpected output dims"; return AJI_ERR_SHAPE; }
    std::swap(rgb_out.buf, cur->buf);           // O(1) hand-off (cur is scratch; keeps an allocation)
    rgb_out.w = cur->w; rgb_out.h = cur->h; rgb_out.c = cur->c;
    return AJI_OK;
}

// Allocate + upload the device-side out-color state (weights, scratch, output planes) for
// one (W,H,format,matrix,range). Reused across frames; rebuilt if the stream changes. The
// spline36 weights are resample.h's, computed on the host -> the GPU color matches gpu_post.
static bool gpu_color_ensure(aji_ctx* c, int W, int H, int fmt, int mat, int rng) {
    GpuColor& gc = c->gc;
    if (gc.ready && gc.W == W && gc.H == H && gc.fmt == fmt && gc.mat == mat && gc.rng == rng) return true;
    gc.free();
    const int cw = W >> 1, ch = H >> 1;
    aji_csp csp = aji_resample::make_csp(fmt, mat, rng);
    gc.csp.kr = csp.kr; gc.csp.kb = csp.kb;
    gc.csp.yscale = csp.yscale; gc.csp.yoff = csp.yoff;
    gc.csp.cscale = csp.cscale; gc.csp.coff = csp.coff;
    gc.csp.is_p010 = (fmt == AJI_FMT_P010) ? 1 : 0;
    gc.csp.qdiv = gc.csp.is_p010 ? 64.0f : 1.0f;
    gc.csp.qmax = gc.csp.is_p010 ? 1023.0f : 255.0f;
    double sx, sy; aji_resample::chroma_shifts(AJI_SITING_LEFT, false, &sx, &sy);  // FORCE LEFT, like gpu_post
    weights ph = aji_resample::compute(W, cw, sx, AJI_FILTER_SPLINE36);
    weights pv = aji_resample::compute(H, ch, sy, AJI_FILTER_SPLINE36);
    gc.ph_taps = ph.taps; gc.pv_taps = pv.taps;
    bool ok = true;
    auto up_i = [&](int*& d, const std::vector<int>& v) {
        ok = ok && hipMalloc(&d, v.size()*sizeof(int)) == hipSuccess
                && hipMemcpy(d, v.data(), v.size()*sizeof(int), hipMemcpyHostToDevice) == hipSuccess; };
    auto up_f = [&](float*& d, const std::vector<float>& v) {
        ok = ok && hipMalloc(&d, v.size()*sizeof(float)) == hipSuccess
                && hipMemcpy(d, v.data(), v.size()*sizeof(float), hipMemcpyHostToDevice) == hipSuccess; };
    up_i(gc.ph_start, ph.start); up_f(gc.ph_wt, ph.wt);
    up_i(gc.pv_start, pv.start); up_f(gc.pv_wt, pv.wt);
    const int bytes = gc.csp.is_p010 ? 2 : 1;
    ok = ok && hipMalloc(&gc.Un, (size_t)W*H*sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&gc.Vn, (size_t)W*H*sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&gc.hu, (size_t)cw*H*sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&gc.hv, (size_t)cw*H*sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&gc.yplane, (size_t)W*H*bytes) == hipSuccess;
    ok = ok && hipMalloc(&gc.uvplane, (size_t)cw*ch*2*bytes) == hipSuccess;
    if (!ok) { gc.free(); return false; }
    gc.W = W; gc.H = H; gc.fmt = fmt; gc.mat = mat; gc.rng = rng; gc.ready = true;
    return true;
}

// GPU-resident path for a single-model no-resize chain: CPU-colored RGB input -> model ->
// GPU out-color (RGB fp16 device -> NV12/P010 device) -> small YUV D2H into `out`. The 4K
// RGB output never crosses PCIe and the fp16->fp32 cast + CPU gpu_post are gone. Runs on
// the worker thread (the model output device buffer is consumed before the next eval).
static int run_chain_gpu(aji_ctx* c, RgbMat& rgb_in, const aji_frame* out,
                         int ofmt, int omat, int orng, std::string* errmsg) {
    MgxModel* nm = c->models[0].get();
    const int IW = rgb_in.w, IH = rgb_in.h;
    const size_t n = (size_t)3 * IW * IH;
    half_t* ib = c->scratch.mdl_in.get(n);
    if (!ib) { if (errmsg) *errmsg = "pinned host alloc failed (in)"; return AJI_ERR; }
    const float* cb = rgb_in.buf.data();
    #pragma omp parallel for schedule(static) num_threads(color_nt(c, (long)n))
    for (long i = 0; i < (long)n; i++) ib[i] = (half_t)cb[i];
    try {
        migraphx::program_parameters pp;
        for (auto& p : nm->params) {
            if (p.is_input) {
                if (hipMemcpy(p.dev, ib, p.bytes, hipMemcpyHostToDevice) != hipSuccess) { if (errmsg) *errmsg = "hipMemcpy H2D failed"; return AJI_ERR; }
            }
            migraphx::shape sh(migraphx_shape_half_type, p.dims);
            pp.add(p.name.c_str(), migraphx::argument(sh, p.dev));
        }
        auto _ge = std::chrono::steady_clock::now();
        int OW = 0, OH = 0;
        {
            std::lock_guard<std::mutex> lk(c->gpu_eval_mtx);
            // eval→gpu-out-color→final sync all on the default stream: one lock covers both.
            auto outs = nm->prog.eval(pp);   // inference graph (MIGraphX), not code-eval
            if (hipDeviceSynchronize() != hipSuccess) { if (errmsg) *errmsg = "eval sync failed"; return AJI_ERR; }
            if (getenv("AJI_ROCM_TIMING")) { static double te=0; static long ne=0;
                te += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_ge).count();
                if (++ne % 50 == 0) fprintf(stderr, "[AJI_ROCM_TIMING/eval] eval+sync=%.1fms\n", te/ne); }
            auto oshape = outs[0].get_shape();
            auto osh = oshape.lengths();
            if (osh.size() < 4) { if (errmsg) *errmsg = "bad output rank"; return AJI_ERR_SHAPE; }
            OW = (int)osh[3]; OH = (int)osh[2];
            if (OW != c->out_w || OH != c->out_h) { if (errmsg) *errmsg = "gpu-color out dims mismatch"; return AJI_ERR_SHAPE; }
            if (getenv("AJI_ROCM_GPU_COLOR_DEBUG")) {
                static bool once = false;
                if (!once) { once = true;
                    auto st = oshape.strides();
                    fprintf(stderr, "[gpucolor] out lengths={%zu,%zu,%zu,%zu} strides={",
                            osh[0],osh[1],osh[2],osh[3]);
                    for (auto s : st) fprintf(stderr, "%zu,", s);
                    fprintf(stderr, "} bytes=%zu  packedWHx3=%zu\n", oshape.bytes(), (size_t)3*OW*OH*2);
                }
            }
            if (!gpu_color_ensure(c, OW, OH, ofmt, omat, orng)) { if (errmsg) *errmsg = "gpu color setup failed"; return AJI_ERR; }
            GpuColor& gc = c->gc;
            if (const char* dp = getenv("AJI_ROCM_DUMP")) {   // debug: dump model in/out fp16 to isolate non-determinism
                std::string pin = std::string(dp) + ".in", pout = std::string(dp) + ".out";
                FILE* fi = fopen(pin.c_str(), "wb"); if (fi) { fwrite(ib, sizeof(half_t), n, fi); fclose(fi); }
                std::vector<half_t> ob((size_t)3 * OW * OH);
                hipMemcpy(ob.data(), outs[0].data(), ob.size() * sizeof(half_t), hipMemcpyDeviceToHost);
                FILE* fo = fopen(pout.c_str(), "wb"); if (fo) { fwrite(ob.data(), sizeof(half_t), ob.size(), fo); fclose(fo); }
            }
            auto _gc = std::chrono::steady_clock::now();
            aji_gpu_out_color(outs[0].data(), OW, OH, gc.csp,
                gc.ph_start, gc.ph_wt, gc.ph_taps, gc.pv_start, gc.pv_wt, gc.pv_taps,
                gc.Un, gc.Vn, gc.hu, gc.hv, gc.yplane, gc.uvplane, nullptr);
            if (hipDeviceSynchronize() != hipSuccess) { if (errmsg) *errmsg = "gpu out-color failed"; return AJI_ERR; }
            if (getenv("AJI_ROCM_TIMING")) { static double tg=0; static long ng=0;
                tg += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_gc).count();
                if (++ng % 50 == 0) fprintf(stderr, "[AJI_ROCM_TIMING/gpucolor] out-color+D2H=%.1fms\n", tg/ng); }
        }  // gpu_eval_mtx released here; D2H copies below use c->gc.yplane/c->gc.uvplane (stable)
        {
            GpuColor& gc = c->gc;
            const int cw = OW >> 1, ch = OH >> 1;
            const int bytes = gc.csp.is_p010 ? 2 : 1;
            ptrdiff_t ys  = out->stride[0] ? out->stride[0] : (ptrdiff_t)OW * bytes;
            ptrdiff_t uvs = out->stride[1] ? out->stride[1] : (ptrdiff_t)cw * 2 * bytes;
            if (hipMemcpy2D(out->plane[0], ys, gc.yplane, (size_t)OW*bytes, (size_t)OW*bytes, OH, hipMemcpyDeviceToHost) != hipSuccess) { if (errmsg) *errmsg = "D2H Y failed"; return AJI_ERR; }
            if (hipMemcpy2D(out->plane[1], uvs, gc.uvplane, (size_t)cw*2*bytes, (size_t)cw*2*bytes, ch, hipMemcpyDeviceToHost) != hipSuccess) { if (errmsg) *errmsg = "D2H UV failed"; return AJI_ERR; }
        }
    } catch (const std::exception& e) {
        if (errmsg) *errmsg = std::string("gpu chain failed: ") + e.what(); return AJI_ERR;
    }
    return AJI_OK;
}

// The worker: pull submitted slots in order, run their chain on the GPU, mark DONE.
static void infer_worker_loop(aji_ctx* c) {
    for (;;) {
        int idx;
        {
            std::unique_lock<std::mutex> lk(c->infer_mtx);
            c->infer_cv_worker.wait(lk, [c]{ return c->infer_stop || !c->infer_pending.empty(); });
            if (c->infer_stop && c->infer_pending.empty()) return;
            idx = c->infer_pending.front();
            c->infer_pending.pop_front();
        }
        std::string em;
        int e;
        bool gpu_colored = false;
        if (c->gpu_color_eligible) {
            aji_ctx::InferSlot& s = c->slots[idx];
            e = run_chain_gpu(c, s.rgb_in, &s.out, s.ofmt, s.omat, s.orng, &em);
            gpu_colored = (e == AJI_OK);   // worker already wrote out; wait skips gpu_post
        } else {
            e = run_chain(c, c->slots[idx].rgb_in, c->slots[idx].rgb_out, &em);
        }
        {
            std::lock_guard<std::mutex> lk(c->infer_mtx);
            c->slots[idx].err = e;
            c->slots[idx].errmsg = std::move(em);
            c->slots[idx].gpu_colored = gpu_colored;
            c->slots[idx].state = aji_ctx::SLOT_DONE;
        }
        c->infer_cv_main.notify_all();
    }
}

static void infer_worker_start(aji_ctx* c) {
    if (c->infer_worker_started) return;
    c->infer_stop = false;
    c->infer_worker = std::thread(infer_worker_loop, c);
    c->infer_worker_started = true;
}

static void infer_worker_stop(aji_ctx* c) {
    if (!c->infer_worker_started) return;
    { std::lock_guard<std::mutex> lk(c->infer_mtx); c->infer_stop = true; }
    c->infer_cv_worker.notify_all();
    if (c->infer_worker.joinable()) c->infer_worker.join();
    c->infer_worker_started = false;
}

extern "C" {

AJI_EXPORT aji_ctx* aji_create(const aji_create_params* p) {
    if (!p || p->api_version < 7) return nullptr;
    aji_ctx* c = new aji_ctx();
    c->logfn = p->log;
    c->log_opaque = p->log_opaque;
    c->async_build = p->async_build != 0;  // filter sets 1: compile off-thread, show OSD
#ifdef _OPENMP
    c->nthreads = std::max(1, std::min(8, omp_get_max_threads()));
    c->nthreads_max = std::max(1, omp_get_max_threads());
#endif

    if (p->rife_model_dir) c->rife_model_dir = p->rife_model_dir;

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
    // The filter drains in-flight frames before reconfiguring, but stop the worker anyway
    // so build_plan can clear/rebuild the model list with no chance of a concurrent chain.
    infer_worker_stop(c);
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

    static const bool kT = getenv("AJI_ROCM_TIMING") != nullptr;
    auto _now = []{ return std::chrono::steady_clock::now(); };
    auto _ms  = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };

    // SUBMIT (filter thread): claim a free in-flight slot (back-pressure if all are busy),
    // do CPU in-color into it, then hand it to the worker thread. The worker runs the GPU
    // model chain while this thread returns to color the next frame; aji_wait() collects the
    // result with CPU out-color. Color stays on CPU (resample.h, bit-exact reference); the
    // overlap is what raises 4K fps, not GPU color (that is the later P4 phase B).
    infer_worker_start(c);
    int slot_idx = -1;
    {
        std::unique_lock<std::mutex> lk(c->infer_mtx);
        c->infer_cv_main.wait(lk, [c, &slot_idx]{
            for (int i = 0; i < aji_ctx::kRing; i++)
                if (c->slots[i].state == aji_ctx::SLOT_FREE) { slot_idx = i; return true; }
            return false;
        });
        aji_ctx::InferSlot& sl = c->slots[slot_idx];
        sl.ticket = ++c->infer_last_ticket;
        sl.err = AJI_OK; sl.errmsg.clear();
        sl.state = aji_ctx::SLOT_SUBMITTED;   // reserved; not yet in the worker queue
    }
    aji_ctx::InferSlot& slot = c->slots[slot_idx];
    auto _t0 = _now();

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

    // ---- CPU in-color into the slot's input buffer ----
    if (gpu_pre(c, W, H, format, matrix, range, siting, Yf, Uf, Vf, slot.rgb_in) != AJI_OK) {
        { std::lock_guard<std::mutex> lk(c->infer_mtx); slot.state = aji_ctx::SLOT_FREE; slot.ticket = 0; }
        c->infer_cv_main.notify_all();
        logmsg(c, 2, c->err.c_str()); return AJI_ERR;
    }
    if (kT) { static double tp=0; static long np=0; tp += _ms(_t0, _now());
        if (++np % 50 == 0) fprintf(stderr, "[AJI_ROCM_TIMING/pre] in-color=%.1fms\n", tp/np); }

    // ---- hand the slot to the worker; its GPU chain overlaps the next frame's color ----
    slot.out = *out; slot.ofmt = out_format; slot.omat = out_matrix; slot.orng = out_range;
    {
        std::lock_guard<std::mutex> lk(c->infer_mtx);
        c->infer_pending.push_back(slot_idx);
    }
    c->infer_cv_worker.notify_one();
    return AJI_OK;
}

// Async completion (aji_flush/done/wait). The filter submits up to `depth` frames, then
// collects them in submission order. flush returns the last-submitted ticket; done is a
// non-blocking check; wait blocks for the worker's GPU chain, then does CPU out-color —
// which overlaps the worker's NEXT eval, the win this whole path exists for.
AJI_EXPORT uint64_t aji_flush(aji_ctx* c, void* /*stream*/) {
    if (!c) return 0;
    std::lock_guard<std::mutex> lk(c->infer_mtx);
    return c->infer_last_ticket;
}

AJI_EXPORT int aji_done(aji_ctx* c, uint64_t ticket) {
    if (!c || ticket == 0) return 1;
    std::lock_guard<std::mutex> lk(c->infer_mtx);
    for (int i = 0; i < aji_ctx::kRing; i++)
        if (c->slots[i].ticket == ticket) return c->slots[i].state == aji_ctx::SLOT_DONE ? 1 : 0;
    return 1;   // unknown ticket: already collected
}

AJI_EXPORT int aji_wait(aji_ctx* c, uint64_t ticket) {
    if (!c || ticket == 0) return AJI_OK;
    static const bool kT = getenv("AJI_ROCM_TIMING") != nullptr;
    int idx = -1;
    {
        std::unique_lock<std::mutex> lk(c->infer_mtx);
        c->infer_cv_main.wait(lk, [c, ticket, &idx]{
            for (int i = 0; i < aji_ctx::kRing; i++)
                if (c->slots[i].ticket == ticket) {
                    if (c->slots[i].state == aji_ctx::SLOT_DONE) { idx = i; return true; }
                    return false;   // still in flight -> keep waiting
                }
            idx = -1; return true;  // ticket not present: already collected
        });
    }
    if (idx < 0) return AJI_OK;
    aji_ctx::InferSlot& s = c->slots[idx];
    int rc = AJI_OK;
    if (s.err != AJI_OK) { c->err = s.errmsg; logmsg(c, 2, c->err.c_str()); rc = s.err; }
    else if (s.gpu_colored) {
        // the worker did GPU out-color + D2H straight into s.out — nothing to do here
    }
    else {
        auto _t = std::chrono::steady_clock::now();
        if (gpu_post(c, s.rgb_out, s.ofmt, s.omat, s.orng, &s.out) != AJI_OK) {
            logmsg(c, 2, c->err.c_str()); rc = AJI_ERR;
        } else if (kT) { static double tq=0; static long nq=0;
            tq += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_t).count();
            if (++nq % 50 == 0) fprintf(stderr, "[AJI_ROCM_TIMING/post] out-color=%.1fms\n", tq/nq); }
    }
    {
        std::lock_guard<std::mutex> lk(c->infer_mtx);
        s.state = aji_ctx::SLOT_FREE; s.ticket = 0;
        // Also free any still-occupied slot with an EARLIER ticket. The filter drains the
        // pipeline on seek/flush (drain_ring) by waiting only the NEWEST ticket — tickets
        // complete in submission order, so that covers all GPU work, but our slots are freed
        // per aji_wait. Without this the un-waited older slots leak, and after a few seeks all
        // kRing slots are gone so aji_infer blocks forever for a free slot (the seek deadlock).
        // The FIFO worker finished every lower-ticket frame before this one, so they are done.
        for (int i = 0; i < aji_ctx::kRing; i++)
            if (c->slots[i].ticket != 0 && c->slots[i].ticket < ticket) {
                c->slots[i].state = aji_ctx::SLOT_FREE;
                c->slots[i].ticket = 0;
            }
    }
    c->infer_cv_main.notify_all();
    return rc;
}

AJI_EXPORT const char* aji_current_log(aji_ctx* c) { return c ? c->log.c_str() : ""; }
AJI_EXPORT int aji_scale_factor(aji_ctx* c) { return c ? c->chain_scale : 0; }
// 0 (no interpolation) until the RIFE .mxr is actually loaded — the filter only calls
// aji_infer_rife when this returns 1, so it must stay 0 while the engine compiles.
AJI_EXPORT int aji_rife_factor(aji_ctx* c, int* num, int* den) {
    if (!c || !c->rife.enabled || !c->rife.loaded) return 0;
    if (num) *num = c->rife.num;
    if (den) *den = c->rife.den;
    return 1;
}
AJI_EXPORT int aji_rife_before_upscale(aji_ctx* c) {
    return (c && c->rife.enabled && c->rife.before_upscale) ? 1 : 0;
}
// Returns 1 exactly once when a background engine compile has finished (success or
// failure); the filter then re-runs aji_configure, which finds the now-cached engine
// (or, on failure, the failed-set marker -> passthrough). 0 if no build, still running,
// or already reported.
AJI_EXPORT int aji_poll(aji_ctx* c) {
    if (!c || !c->build) return 0;
    if (c->build->done.load() != 1) return 0;            // no build finished
    if (c->build->reported.exchange(true)) return 0;     // already told the caller once
    if (c->build_thread.joinable()) c->build_thread.join();  // reap the finished worker
    if (!c->build->ok.load())
        c->failed_builds.insert(c->build->key);          // don't re-kick a failing build
    return 1;
}

// Per-frame RIFE interpolation (4:2:0, SYNCHRONOUS). Produces the frame at timestep
// `t in (0,1)` between A and B. The `stream` arg is NULL on the sw path and ignored —
// aji_rocm always syncs on its own eval before returning. Returns:
//   AJI_OK         — `out` written (interpolated frame).
//   AJI_SCENE      — scene change detected; `out` left UNTOUCHED (caller duplicates A).
//   AJI_ERR_FORMAT / AJI_ERR_SHAPE / AJI_ERR — invalid input or eval failure.
// Color is forced BT.709 inside rife_cpu (range follows the source); the eval binding
// mirrors run_chain (bind the MODEL's params[].dev; the RifeState dev_in/dev_out are
// Phase-B scaffolding and are left untouched here).
AJI_EXPORT int aji_infer_rife(aji_ctx* c, const aji_frame* a, const aji_frame* b,
                              double t, const aji_frame* out, void* /*stream*/) {
    if (!c || !c->rife.loaded) { if (c) c->err = "rife not loaded"; return AJI_ERR; }
    auto& R = c->rife;

    // 1. Validate: formats equal and 4:2:0; dims equal the RIFE geometry's source w x h.
    if (a->format != b->format || a->format != out->format) {
        c->err = "rife fmt mismatch"; return AJI_ERR_FORMAT;
    }
    if (a->format != AJI_FMT_NV12 && a->format != AJI_FMT_P010) {
        c->err = "rife needs nv12/p010"; return AJI_ERR_FORMAT;
    }
    if (a->width != R.g.w || a->height != R.g.h ||
        b->width != R.g.w || b->height != R.g.h ||
        out->width != R.g.w || out->height != R.g.h) {
        c->err = "rife dim mismatch"; return AJI_ERR_SHAPE;
    }

    const int pw = R.g.pw, ph = R.g.ph;
    const size_t plane = (size_t)pw * ph;

    // --- env-gated per-phase profiling (AJI_RIFE_PROFILE=1) -------------------
    // All timing state is function-local static; completely inert unless the env var is set.
    // Phases: scene_detect | color_pre | tensor_cast | h2d | eval_sync | d2h | fp16_out | color_post
    static const bool s_prof = (getenv("AJI_RIFE_PROFILE") != nullptr);
    using clk = std::chrono::steady_clock;
    using dur = std::chrono::duration<double, std::milli>;
    static double s_acc[8]  = {};
    static long   s_calls   = 0;
    clk::time_point tp[9];   // 9 fence-points for 8 intervals
    if (s_prof) tp[0] = clk::now();

    // 2. Scene detect on the UNPADDED luma (plane[0]), raw container values. norm scales
    // the container range; divisor (pw*ph) is the PADDED area (in the helper). A scene
    // change skips interpolation: return AJI_SCENE with `out` untouched (caller dups A).
    const double norm = (a->format == AJI_FMT_P010) ? 1.0 / 65472.0 : 1.0 / 255.0;
    bool scene;
    if (a->format == AJI_FMT_P010) {
        scene = rife_cpu::scene_detect((const uint16_t*)a->plane[0], a->stride[0],
                                       (const uint16_t*)b->plane[0], b->stride[0],
                                       R.g.w, R.g.h, pw, ph, norm, R.scd_threshold);
    } else {
        scene = rife_cpu::scene_detect((const uint8_t*)a->plane[0], a->stride[0],
                                       (const uint8_t*)b->plane[0], b->stride[0],
                                       R.g.w, R.g.h, pw, ph, norm, R.scd_threshold);
    }
    if (s_prof) tp[1] = clk::now();
    if (scene) return AJI_SCENE;

    // 3. Assemble the 11-ch fp32 tensor. Consts (ch7-10) are already in R.assembly (set
    // once by setup_rife). A -> ch0-2, B -> ch3-5 (BT.709 + bilinear chroma upsample,
    // centered into pw x ph with black borders); ch6 <- the timestep plane.
    rife_cpu::yuv420_to_rgb_planes(*a, R.g, R.assembly.data(), 0, (aji_range)a->range);
    rife_cpu::yuv420_to_rgb_planes(*b, R.g, R.assembly.data(), 1, (aji_range)a->range);
    float* ch6 = R.assembly.data() + 6 * plane;
    for (size_t i = 0; i < plane; ++i) ch6[i] = (float)t;   // (_Float16)t below = RNE, TRT parity
    if (s_prof) tp[2] = clk::now();

    // 4. fp32 -> fp16 the full 11-ch tensor into pinned host staging.
    half_t* hin = R.pin_in.get(11 * plane);
    if (!hin) { c->err = "rife pinned host alloc failed (in)"; return AJI_ERR; }
    const float* ab = R.assembly.data();
    const long nin = (long)(11 * plane);
    #pragma omp parallel for schedule(static) num_threads(color_nt(c, nin))
    for (long i = 0; i < nin; ++i) hin[i] = (half_t)ab[i];
    if (s_prof) tp[3] = clk::now();

    // 5. H2D into the model's input param; eval + device sync under gpu_eval_mtx (the
    // upscale worker shares the device-wide sync). Bind every param's device buffer with
    // its fp16 shape, exactly like run_chain. Capture outs[0] inside the lock; D2H after.
    MgxModel* nm = R.model.get();
    migraphx::argument outs_arg;
    try {
        migraphx::program_parameters pp;
        for (auto& p : nm->params) {
            if (p.is_input) {
                if (hipMemcpy(p.dev, hin, p.bytes, hipMemcpyHostToDevice) != hipSuccess) {
                    c->err = "rife hipMemcpy H2D failed"; return AJI_ERR;
                }
            }
            migraphx::shape sh(migraphx_shape_half_type, p.dims);
            pp.add(p.name.c_str(), migraphx::argument(sh, p.dev));
        }
        if (s_prof) tp[4] = clk::now();
        {
            std::lock_guard<std::mutex> lk(c->gpu_eval_mtx);
            auto outs = nm->prog.eval(pp);     // run the RIFE graph (MIGraphX API, not code-eval)
            hipDeviceSynchronize();            // eval is async on the GPU stream; wait before D2H
            outs_arg = outs[0];
        }
    } catch (const std::exception& e) {
        c->err = std::string("rife eval failed: ") + e.what(); return AJI_ERR;
    }
    if (s_prof) tp[5] = clk::now();

    // 6. D2H the 3-ch output into pinned host staging, fp16 -> fp32, then crop the centered
    // w x h window into `out` (BT.709 RGB->YUV + bilinear chroma downsample, range follows A).
    half_t* hout = R.pin_out.get(3 * plane);
    if (!hout) { c->err = "rife pinned host alloc failed (out)"; return AJI_ERR; }
    if (hipMemcpy(hout, outs_arg.data(), nm->out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        c->err = "rife hipMemcpy D2H failed"; return AJI_ERR;
    }
    if (s_prof) tp[6] = clk::now();

    std::vector<float> rgb(3 * plane);
    const long nout = (long)(3 * plane);
    #pragma omp parallel for schedule(static) num_threads(color_nt(c, nout))
    for (long i = 0; i < nout; ++i) rgb[i] = (float)hout[i];
    if (s_prof) tp[7] = clk::now();

    rife_cpu::rgb_planes_to_yuv420(rgb.data(), R.g, *(aji_frame*)out, (aji_range)a->range);
    if (s_prof) tp[8] = clk::now();

    // Accumulate phase durations and print a breakdown every 50 calls.
    if (s_prof) {
        // Phases: [0]=scene_detect [1]=color_pre [2]=tensor_ch6+cast [3]=h2d_param_bind
        //         [4]=eval_sync    [5]=d2h       [6]=fp16_out_cast   [7]=color_post
        // Note: tp[3]->tp[4] = H2D + param bind (hipMemcpy H2D is inside the params loop);
        //       tp[4]->tp[5] = eval + hipDeviceSynchronize (GPU wall-clock).
        for (int i = 0; i < 8; ++i)
            s_acc[i] += dur(tp[i+1] - tp[i]).count();
        ++s_calls;
        if (s_calls % 50 == 0) {
            static const char* names[8] = {
                "scene_detect", "color_pre", "tensor_ch6+cast(fp32->fp16)",
                "h2d+param_bind", "eval+sync(GPU)", "d2h",
                "fp16_out_cast", "color_post"
            };
            double total = 0;
            for (int i = 0; i < 8; ++i) total += s_acc[i];
            fprintf(stderr, "[rife-prof] --- call %ld ---\n", s_calls);
            for (int i = 0; i < 8; ++i)
                fprintf(stderr, "[rife-prof] %s: %.3f ms\n",
                        names[i], s_acc[i] / s_calls);
            fprintf(stderr, "[rife-prof] TOTAL: %.3f ms  (calls=%ld)\n",
                    total / s_calls, s_calls);
        }
    }

    return AJI_OK;
}

AJI_EXPORT const char* aji_last_error(aji_ctx* c) { return c ? c->err.c_str() : "null ctx"; }

AJI_EXPORT void aji_destroy(aji_ctx** c) {
    if (c && *c) {
        infer_worker_stop(*c);   // drain + join the inference worker
        (*c)->gc.free();         // free GPU out-color device buffers
        // A compile may still be running (player quit mid-build). The worker holds its
        // own shared_ptr to the BuildState and never touches the ctx, so detaching is
        // safe — no use-after-free, and aji_destroy doesn't hang for ~140s. The temp-then-
        // rename save means an abandoned compile leaves no half-written .mxr.
        if ((*c)->build_thread.joinable()) (*c)->build_thread.detach();
        delete *c; *c = nullptr;
    }
}

} // extern "C"

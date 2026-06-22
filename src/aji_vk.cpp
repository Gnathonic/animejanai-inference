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
#include "rife_cpu.h"   // portable RIFE math (geometry/scene/consts/4:2:0 color), shared with aji_rocm
#include "net.h"
#include "mat.h"

#include "gpu.h"
#include "command.h"
#include "pipeline.h"
#include "option.h"
#include "layer.h"
#include "layer_type.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <dlfcn.h>      // dladdr: locate spv/ next to this .so for a relocatable bundle
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using aji_resample::weights;

// Where the GPU-resident color kernels (cs_*.spv) live. Resolution order, so a portable
// bundle (libaji_vk.so + spv/ co-located) just works from any CWD with no configuration:
//   1. AJI_VK_SPV_DIR env var (explicit runtime override)
//   2. "<dir of this .so>/spv" via dladdr, if it exists (the relocatable-bundle case;
//      also the dev tree, where spv/ sits next to the built lib)
//   3. compile-time -DAJI_VK_SPV_DIR (e.g. a system-install path), else "spv" (CWD-relative)
#ifndef AJI_VK_SPV_DIR
#define AJI_VK_SPV_DIR "spv"
#endif
static std::string resolve_spv_dir() {
    if (const char* e = getenv("AJI_VK_SPV_DIR")) if (e[0]) return e;
    Dl_info info;
    if (dladdr((void*)&resolve_spv_dir, &info) && info.dli_fname) {
        std::string so = info.dli_fname;
        size_t slash = so.find_last_of('/');
        if (slash != std::string::npos) {
            std::string dir = so.substr(0, slash) + "/spv";
            struct stat st;
            if (stat((dir + "/cs_post_matrix3.spv").c_str(), &st) == 0) return dir;
        }
    }
    return AJI_VK_SPV_DIR;
}
static const std::string kSpvDirStr = resolve_spv_dir();
static const char* kSpvDir = kSpvDirStr.c_str();

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

    // --- async pipeline (overlap CPU color with GPU model; mirrors aji_rocm Phase A) ---
    // aji_infer = non-blocking submit: CPU in-color (filter thread) -> rgb_in -> enqueue.
    // A single worker thread runs the model (rgb_in -> rgb_out) back-to-back, so the GPU
    // stays saturated (-> the amdgpu governor boosts clocks) instead of being starved by
    // serial CPU color. aji_wait does the CPU out-color (filter thread), overlapping the
    // worker's NEXT model. Disable with AJI_VK_SYNC=1 (blocking infer, old behavior).
    enum { SLOT_FREE = 0, SLOT_SUBMITTED, SLOT_RUNNING, SLOT_DONE };
    struct Slot {
        ncnn::Mat rgb_in, rgb_out;
        std::vector<uint8_t> rawY, rawUV;   // raw source planes (GPU in-color path; filled by aji_infer)
        int in_format = 0, in_matrix = 0, in_range = 0, in_siting = 0, in_w = 0, in_h = 0;
        aji_frame out_frame{};
        int out_format = 0, out_matrix = 0, out_range = 0;
        int state = SLOT_FREE;
        uint64_t ticket = 0;
    };
    static const int kRing = 12;
    Slot ring[kRing];
    std::mutex amx;
    std::condition_variable work_cv, done_cv, free_cv;
    std::queue<int> workq;
    std::vector<std::thread> workers;
    std::vector<Scratch> wscratch;  // per-worker out-color scratch (race-free across workers)
    int num_workers = 1;          // concurrent model workers (AJI_VK_WORKERS); >1 saturates the GPU

    // --- GPU-resident post color (the portable fix): model output stays a VkMat, the
    // RGB->YUV matrix + spline36 chroma downsample run on the GPU (bit-exact vs the CPU
    // resample.h path), only ~12-50MB of YUV is downloaded. Frees the CPU entirely. ---
    bool gpu_color = true;        // AJI_VK_CPUCOLOR=1 falls back to the CPU path
    bool gpu_incolor = false;     // AJI_VK_GPUINCOLOR=1: in-color (YUV->RGB) on the GPU too (frees the filter thread)
    struct GpShared {             // shared, read-only across workers; (re)built per output res
        ncnn::Pipeline *p_mat = nullptr, *p_h = nullptr, *p_uvv = nullptr;
        ncnn::Layer* cast = nullptr;          // ncnn-native fp16->fp32 cast for the model output
        ncnn::Option castopt;
        ncnn::VkMat vPHs, vPHw, vPVs, vPVw;   // chroma-down spline36 weights (start, taps)
        int W2 = 0, H2 = 0, fmt = 0, mat = 0, rng = 0, phtaps = 0, pvtaps = 0;
        aji_csp csp{}; bool ready = false; bool fp16 = false;  // fp16-storage post buffers (AJI_VK_FP16COLOR)
        // --- GPU IN-color (raw YUV -> chroma UPsample -> YUV->RGB); built per SOURCE res ---
        ncnn::Pipeline *ip_h = nullptr, *ip_v = nullptr, *ip_mat = nullptr;
        ncnn::VkMat ivUPHs, ivUPHw, ivUPVs, ivUPVw;   // chroma-UP spline36 weights
        int iW = 0, iH = 0, ifmt = 0, imat = 0, irng = 0, isit = 0, iphtaps = 0, ipvtaps = 0;
        aji_csp icsp{}; bool iready = false;
    } gp;
    std::mutex gpmx;              // guards lazy build of gp
    struct GpBuf {               // per-worker GPU-post scratch (3D-c1 VkMats; elempack-1)
        ncnn::VkMat vIn, Un, Vn, Yout, hu, hv, uvout, dV, dH, dUV;
        ncnn::Mat oy, ouv; bool ready = false; bool fp16 = false;  // tracks GpShared.fp16 it was built for
        // GPU in-color per-worker scratch
        ncnn::VkMat ivRawY, ivRawUV, it0u, it0v, it1u, it1v, ivRGB, ivModel, idH, idV, idM;
        bool iready = false;
    };
    std::vector<GpBuf> gpbuf;
    bool worker_started = false, worker_stop = false;
    bool sync_mode = false;
    uint64_t ticket_counter = 0;

    // --- RIFE interpolation (driven by aji_infer_rife, NOT the upscale step loop) ---
    // Mirrors aji_rocm's RifeState (Phase A: 4:2:0, CPU color via rife_cpu, GPU model on
    // ncnn-Vulkan). The model is a single de-batched rife_v4.x ncnn net (11ch in, 3ch out);
    // the warps run GPU-native via the gridsample_vulkan layer. No async build (ncnn loads
    // instantly) -> only LOADED / DISABLED.
    std::string rife_model_dir;
    struct RifeState {
        std::unique_ptr<NcnnModel> model;          // RIFE ncnn net (in_name/out_name; scale=1)
        rife_cpu::Geom g{};
        int num = 1, den = 1;
        double scd_threshold = 0.150;
        bool before_upscale = true;
        bool enabled = false, loaded = false;
        std::vector<float> assembly;               // 11*plane fp32 staging (consts ch7-10 filled once)
        void reset() { model.reset(); assembly.clear(); enabled = false; loaded = false; }
    } rife;

    std::string err, log;
    aji_log_fn logfn = nullptr;
    void* log_opaque = nullptr;
};

static void logmsg(aji_ctx* c, int level, const char* m) {
    if (c && c->logfn) c->logfn(c->log_opaque, level, m);
}

// Optional per-stage timing (AJI_VK_TIMING=1): splits in-color / model / out-color.
static double aji_now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const bool kTiming = getenv("AJI_VK_TIMING") != nullptr;

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
    m->net.opt.use_fp16_storage = true;   // model output VkMat is fp16; the GPU post reads it as fp16
    m->net.opt.use_fp16_arithmetic = true;
    // ncnn has no TensorRT-style per-layer engine build; the matrix-core
    // (cooperative_matrix) GEMM path is the big win. Gated by device support inside
    // ncnn, so safe to request unconditionally; AJI_VK_NO_CM=1 disables it for A/B.
    m->net.opt.use_cooperative_matrix = getenv("AJI_VK_NO_CM") == nullptr;
    // Winograd OFF on purpose: on RADV/gfx1201 ncnn's winograd23/43 path is memory-
    // bandwidth-bound (6x6 tiles, 2.25x intermediate storage) and ~2x SLOWER at HD/4K
    // than the implicit-GEMM cooperative-matrix path (im2col-free WMMA GEMM). Measured
    // 1080p->4K pure model: winograd 15.3 fps vs gemm-CM 28.9 (Balanced), 32 vs 63
    // (Performance); gemm-CM wins at every lower res too and is more numerically
    // accurate (no winograd transform; matches winograd within 0.3% fp16).
    // AJI_VK_WINOGRAD=1 restores winograd for A/B.
    m->net.opt.use_winograd_convolution = getenv("AJI_VK_WINOGRAD") != nullptr;
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

// Load a RIFE ncnn net (11-channel input, 3-channel output, no spatial scale). Same
// Vulkan/fp16/coopmat opts as load_model but no 3-channel scale probe (the input is 11ch).
static std::unique_ptr<NcnnModel> load_rife_model(aji_ctx* c, const std::string& param) {
    auto m = std::make_unique<NcnnModel>();
    m->net.opt.use_vulkan_compute = true;
    m->net.opt.use_fp16_packed = true;
    m->net.opt.use_fp16_storage = true;
    m->net.opt.use_fp16_arithmetic = true;
    m->net.opt.use_cooperative_matrix = getenv("AJI_VK_NO_CM") == nullptr;
    m->net.opt.use_winograd_convolution = getenv("AJI_VK_WINOGRAD") != nullptr;
    m->net.opt.use_sgemm_convolution = true;
    m->net.opt.use_subgroup_ops = true;
    m->net.opt.use_shader_local_memory = true;
    if (m->net.load_param(param.c_str())) { c->err = "rife load_param failed: " + param; return nullptr; }
    std::string bin = bin_for(param);
    if (m->net.load_model(bin.c_str())) { c->err = "rife load_model failed: " + bin; return nullptr; }
    const auto& ins = m->net.input_names();
    const auto& outs = m->net.output_names();
    if (ins.empty() || outs.empty()) { c->err = "rife model has no input/output blobs: " + param; return nullptr; }
    m->in_name = ins[0];
    // The de-batched rife_v4.x ncnn graph exposes several blobs as outputs (intermediate
    // flow/warp tensors + the final RGB "out0"); pnnx lists them in graph order, so [0] is an
    // intermediate (c=1), NOT the result. Pick the real blended output: prefer "out0", else last.
    m->out_name = outs.back();
    for (const auto& nm : outs) if (nm == "out0") { m->out_name = nm; break; }
    m->scale = 1;
    return m;
}

// Configure RIFE for the active chain (conf mode). Mirrors aji_rocm::setup_rife but without
// the MIGraphX async-build cascade (ncnn loads instantly) -> only LOADED or DISABLED. RIFE
// never hard-errors the chain: an invalid model code, missing dir, or load failure just
// disables interpolation and the chain plays on. The .param/.bin is a de-batched rife_v4.x
// converted from the SAME vsmlrt ONNX as aji_rocm/aji_trt (see tools/rife_ncnn).
static void setup_rife(aji_ctx* c, const AjiChainConf* chain,
                       int src_w, int src_h, int cw, int ch, std::string& steps_log) {
    c->rife.reset();

    // RIFE runs at the SOURCE shape when before_upscale (interpolate then upscale), else at
    // the chain-output shape.
    int rw = chain->rife_before_upscale ? src_w : cw;
    int rh = chain->rife_before_upscale ? src_h : ch;
    c->rife.g = rife_cpu::geometry(rw, rh);
    const int pw = c->rife.g.pw, ph = c->rife.g.ph;

    std::string name = rife_cpu::model_name(chain->rife_model, chain->rife_ensemble);
    if (name.empty() || c->rife_model_dir.empty()) {
        steps_log += "(RIFE disabled: " +
                     std::string(name.empty() ? "invalid rife_model code" : "no rife_model_dir") + "); ";
        return;
    }
    const std::string param = c->rife_model_dir + "/" + name + ".param";
    auto m = load_rife_model(c, param);
    if (!m) {
        logmsg(c, 2, c->err.c_str());
        steps_log += "(RIFE disabled: ncnn model load failed for " + name + "); ";
        return;
    }
    c->rife.model = std::move(m);

    // Host const template: ch7-10 (mesh/multiplier) filled once; aji_infer_rife reuses this
    // per frame and writes ch0-6 (frames + timestep) on top.
    const size_t plane = (size_t)pw * ph;
    c->rife.assembly.assign(11 * plane, 0.f);
    rife_cpu::fill_consts(c->rife.assembly.data(), pw, ph);

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
                    const aji_frame* out, aji_ctx::Scratch& scr) {
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
    std::vector<float>& Yout = scr.Yout; Yout.resize((size_t)W * H);
    std::vector<float>& Un = scr.Un; Un.resize((size_t)W * H);
    std::vector<float>& Vn = scr.Vn; Vn.resize((size_t)W * H);
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (long i = 0; i < (long)W * H; i++) {
        float Y = csp.kr * R[i] + kg * G[i] + csp.kb * B[i];
        Yout[i] = quant(Y * csp.yscale + csp.yoff, qdiv, qmax);
        Un[i] = (B[i] - Y) / (2.0f * (1.0f - csp.kb));
        Vn[i] = (R[i] - Y) / (2.0f * (1.0f - csp.kr));
    }
    std::vector<float>& hu = scr.hu; hu.resize((size_t)cw * H);
    std::vector<float>& hv = scr.hv; hv.resize((size_t)cw * H);
    #pragma omp parallel for schedule(static) num_threads(NT)
    for (int y = 0; y < H; y++)
        for (int x = 0; x < cw; x++) {
            float u = 0, v = 0; int s0 = ph.start[x];
            for (int j = 0; j < ph.taps; j++) { float w = ph.wt[(size_t)x * ph.taps + j]; int s = mirr(s0 + j, W);
                u += w * Un[(size_t)y * W + s]; v += w * Vn[(size_t)y * W + s]; }
            hu[(size_t)y * cw + x] = u; hv[(size_t)y * cw + x] = v;
        }
    std::vector<float>& uvout = scr.uvout; uvout.resize((size_t)cw * ch * 2);
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
    c->rife.reset();   // reconfigure-safe: drop any prior RIFE engine (setup_rife re-enables if the chain asks)
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
        if (chain->rife) setup_rife(c, chain, w, h, cw, ch, steps_log);
        char hdr[160];
        snprintf(hdr, sizeof hdr, "slot %d chain %d: %dx%d -> %dx%d  [", c->slot, chain->index, w, h, cw, ch);
        c->log = std::string(hdr) + steps_log + "]";
    }

    c->in_w = w; c->in_h = h; c->out_w = cw; c->out_h = ch;
    c->active = !c->steps.empty();
    return c->active ? 1 : 0;
}

// Run the active step plan (model chain + resizes) on an RGB ncnn::Mat. Used by both
// the synchronous path and the async worker. Touches only c->models/c->steps and (for
// RESIZE steps) c->scratch.rt0 — all worker-confined in async mode.
static int run_chain(aji_ctx* c, const ncnn::Mat& in_rgb, ncnn::Mat& out_rgb,
                     ncnn::VkAllocator* blob = nullptr, ncnn::VkAllocator* staging = nullptr) {
    double tm0 = kTiming ? aji_now() : 0;
    ncnn::Mat cur = in_rgb;
    for (const Step& st : c->steps) {
        if (st.kind == Step::RESIZE) {
            ncnn::Mat dst;
            if (gpu_resize(c, cur, st.out_w, st.out_h, dst) != AJI_OK) return AJI_ERR;
            cur = dst;
        } else {
            NcnnModel* nm = c->models[st.model_idx].get();
            ncnn::Extractor ex = nm->net.create_extractor();
            // per-worker allocators (multi-worker async): blob allocators are not
            // thread-safe, so each concurrent model run gets its own.
            if (blob) { ex.set_blob_vkallocator(blob); ex.set_workspace_vkallocator(blob); }
            if (staging) ex.set_staging_vkallocator(staging);
            if (ex.input(nm->in_name.c_str(), cur) != 0) { c->err = "infer: input failed"; return AJI_ERR; }
            ncnn::Mat o;
            if (ex.extract(nm->out_name.c_str(), o) != 0) { c->err = "infer: extract failed"; return AJI_ERR; }
            cur = o;
        }
    }
    if (cur.w != c->out_w || cur.h != c->out_h || cur.c < 3) { c->err = "infer: unexpected output dims"; return AJI_ERR_SHAPE; }
    double tm1 = kTiming ? aji_now() : 0;
    if (cur.elempack != 1 || cur.c != 3) {
        ncnn::Mat tmp; ncnn::convert_packing(cur, tmp, 1);
        out_rgb.create(cur.w, cur.h, 3);
        for (int k = 0; k < 3; k++) memcpy(out_rgb.channel(k), tmp.channel(k), (size_t)cur.w * cur.h * sizeof(float));
    } else {
        out_rgb = cur;
    }
    if (kTiming) {
        static double s_m = 0, s_f = 0; static int s_n = 0; static int s_pack = -1;
        s_m += (tm1 - tm0) * 1000; s_f += (aji_now() - tm1) * 1000; s_n++; s_pack = cur.elempack;
        if (s_n % 30 == 0) fprintf(stderr, "[run_chain n=%d] model+D2H=%.2f ms  flatten=%.2f ms  (out elempack=%d c=%d)\n",
                                   s_n, s_m/s_n, s_f/s_n, s_pack, cur.c);
    }
    return AJI_OK;
}

// Build the shared GPU-post pipelines + spline36 chroma-down weights for a given
// output (W2,H2,fmt,mat,rng). Read-only after build, shared across workers.
static bool build_gp_shared(aji_ctx* c, int W2, int H2, int fmt, int matx, int rng) {
    std::unique_lock<std::mutex> lk(c->gpmx);
    aji_ctx::GpShared& g = c->gp;
    const ncnn::VulkanDevice* vkdev = c->models.empty() ? nullptr : c->models[0]->net.vulkan_device();
    if (!vkdev) return false;
    // --- Precision of the chroma intermediates (Un/Vn/hu/hv) ---
    // fp32 is the DEFAULT: bit-exact vs the CPU resample.h reference for EVERY format (incl.
    // P010/10-bit), and on this class of GPU fp16 color gives no speedup anyway (the model,
    // not the color, is the bottleneck). fp16 storage (halves resample bandwidth) is available
    // opt-in via AJI_VK_FP16COLOR for bandwidth-bound GPUs, but it is HW-GATED: it requires the
    // 16-bit-storage + fp16-arithmetic features the cs_*_f16 kernels `require` (ncnn's
    // record_upload honors use_fp16_storage RAW with no capability check, and a device that
    // merely runs the model does NOT necessarily support storage16 — ncnn silently downgrades
    // the model to fp16-packed/fp32 — so forcing fp16 on an incapable GPU would fail pipeline
    // creation). fp16 is bit-exact only for 8-bit (NV12); for P010 it can diverge up to ~6
    // ten-bit LSB, so the opt-in is for advanced/bandwidth-bound use. AJI_VK_CPUCOLOR (handled
    // in aji_create) disables GPU color entirely and still wins.
    const bool hw_fp16 = vkdev->info.support_fp16_storage() && vkdev->info.support_fp16_arithmetic();
    bool want_fp16 = (getenv("AJI_VK_FP16COLOR") != nullptr) && hw_fp16;
    if (g.ready && g.W2 == W2 && g.H2 == H2 && g.fmt == fmt && g.mat == matx && g.rng == rng && g.fp16 == want_fp16) return true;
    if (!g.p_mat || g.fp16 != want_fp16) {  // (re)create pipelines whenever the precision changes
        delete g.p_mat; delete g.p_h; delete g.p_uvv; g.p_mat = g.p_h = g.p_uvv = nullptr;
        auto mk = [&](const char* n) -> ncnn::Pipeline* {
            bool ok = false; std::string p = std::string(kSpvDir) + "/" + n + ".spv";
            auto s = load_spv(p, &ok);
            if (!ok || s.empty()) { c->err = "gp: load " + p; return nullptr; }
            ncnn::Pipeline* pp = new ncnn::Pipeline(vkdev); pp->set_local_size_xyz(32, 8, 1);
            if (pp->create(s.data(), s.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) { c->err = "gp: create " + std::string(n); delete pp; return nullptr; }
            return pp;
        };
        g.p_mat = mk(want_fp16 ? "cs_post_matrix3_f16" : "cs_post_matrix3");
        g.p_h   = mk(want_fp16 ? "cs_h_f16"            : "cs_h_f32");
        g.p_uvv = mk(want_fp16 ? "cs_post_uv_v_f16"    : "cs_post_uv_v");
        if (!g.p_mat || !g.p_h || !g.p_uvv) { logmsg(c, 2, c->err.c_str()); return false; }
    }
    g.fp16 = want_fp16;
    int cw2 = W2 >> 1, ch2 = H2 >> 1;
    g.csp = aji_resample::make_csp(fmt, matx, rng);
    double sx, sy; aji_resample::chroma_shifts(AJI_SITING_LEFT, false, &sx, &sy);
    weights ph = aji_resample::compute(W2, cw2, sx, AJI_FILTER_SPLINE36);
    weights pv = aji_resample::compute(H2, ch2, sy, AJI_FILTER_SPLINE36);
    g.phtaps = ph.taps; g.pvtaps = pv.taps;
    std::vector<float> phs(ph.start.begin(), ph.start.end()), pvs(pv.start.begin(), pv.start.end());
    ncnn::Option uo; uo.use_vulkan_compute = true; uo.use_packing_layout = false;
    uo.use_fp16_packed = false; uo.use_fp16_storage = false; uo.use_fp16_arithmetic = false;  // weights as plain fp32
    uo.blob_vkallocator = vkdev->acquire_blob_allocator(); uo.workspace_vkallocator = uo.blob_vkallocator;
    uo.staging_vkallocator = vkdev->acquire_staging_allocator();
    g.vPHs = ncnn::VkMat(); g.vPHw = ncnn::VkMat(); g.vPVs = ncnn::VkMat(); g.vPVw = ncnn::VkMat();
    { ncnn::VkCompute up(vkdev);
      auto U = [&](std::vector<float>& h, ncnn::VkMat& v){ ncnn::Mat m((int)h.size(), (void*)h.data(), (size_t)4u); up.record_upload(m, v, uo); };
      U(phs, g.vPHs); U(ph.wt, g.vPHw); U(pvs, g.vPVs); U(pv.wt, g.vPVw); up.submit_and_wait(); }
    vkdev->reclaim_blob_allocator(uo.blob_vkallocator); vkdev->reclaim_staging_allocator(uo.staging_vkallocator);
    g.W2 = W2; g.H2 = H2; g.fmt = fmt; g.mat = matx; g.rng = rng; g.ready = true;
    return true;
}

// Build the GPU IN-color pipelines + chroma-UPsample weights for a SOURCE resolution (lazy,
// cached in GpShared; rebuilt when source dims/format/matrix/siting change).
static bool build_gp_incolor(aji_ctx* c, int W, int H, int fmt, int matx, int rng, int siting) {
    std::unique_lock<std::mutex> lk(c->gpmx);
    aji_ctx::GpShared& g = c->gp;
    const ncnn::VulkanDevice* vkdev = c->models.empty() ? nullptr : c->models[0]->net.vulkan_device();
    if (!vkdev) return false;
    if (g.iready && g.iW == W && g.iH == H && g.ifmt == fmt && g.imat == matx && g.irng == rng && g.isit == siting) return true;
    if (!g.ip_h) {
        auto mk = [&](const char* n) -> ncnn::Pipeline* {
            bool ok = false; std::string p = std::string(kSpvDir) + "/" + n + ".spv";
            auto s = load_spv(p, &ok);
            if (!ok || s.empty()) { c->err = "gp in: load " + p; return nullptr; }
            ncnn::Pipeline* pp = new ncnn::Pipeline(vkdev); pp->set_local_size_xyz(32, 8, 1);
            if (pp->create(s.data(), s.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) { c->err = "gp in: create " + std::string(n); delete pp; return nullptr; }
            return pp;
        };
        g.ip_h = mk("cs_pre_h"); g.ip_v = mk("cs_pre_v"); g.ip_mat = mk("cs_pre_matrix");
        if (!g.ip_h || !g.ip_v || !g.ip_mat) { logmsg(c, 2, c->err.c_str()); return false; }
    }
    int cw = W >> 1, ch = H >> 1;
    g.icsp = aji_resample::make_csp(fmt, matx, rng);
    double sx, sy; aji_resample::chroma_shifts(siting, true, &sx, &sy);   // up=true: chroma UPsample siting
    weights ph = aji_resample::compute(cw, W, sx, AJI_FILTER_SPLINE36);
    weights pv = aji_resample::compute(ch, H, sy, AJI_FILTER_SPLINE36);
    g.iphtaps = ph.taps; g.ipvtaps = pv.taps;
    std::vector<float> phs(ph.start.begin(), ph.start.end()), pvs(pv.start.begin(), pv.start.end());
    ncnn::Option uo; uo.use_vulkan_compute = true; uo.use_packing_layout = false;
    uo.use_fp16_packed = false; uo.use_fp16_storage = false; uo.use_fp16_arithmetic = false;
    uo.blob_vkallocator = vkdev->acquire_blob_allocator(); uo.workspace_vkallocator = uo.blob_vkallocator;
    uo.staging_vkallocator = vkdev->acquire_staging_allocator();
    g.ivUPHs = ncnn::VkMat(); g.ivUPHw = ncnn::VkMat(); g.ivUPVs = ncnn::VkMat(); g.ivUPVw = ncnn::VkMat();
    { ncnn::VkCompute up(vkdev);
      auto U = [&](std::vector<float>& h, ncnn::VkMat& v){ ncnn::Mat m((int)h.size(), (void*)h.data(), (size_t)4u); up.record_upload(m, v, uo); };
      U(phs, g.ivUPHs); U(ph.wt, g.ivUPHw); U(pvs, g.ivUPVs); U(pv.wt, g.ivUPVw); up.submit_and_wait(); }
    vkdev->reclaim_blob_allocator(uo.blob_vkallocator); vkdev->reclaim_staging_allocator(uo.staging_vkallocator);
    g.iW = W; g.iH = H; g.ifmt = fmt; g.imat = matx; g.irng = rng; g.isit = siting; g.iready = true;
    return true;
}

// GPU IN-color: raw source YUV (host) -> 2D uint VkMats (avoids the 2^18 1D cap) -> chroma
// UPsample + YUV->RGB kernels -> fp32 RGB VkMat -> convert_packing to the model's fp16 layout.
// Returns the model-input VkMat in out_model. Bit-exact vs the CPU gpu_pre (validated, all res).
static bool gpu_in(aji_ctx* c, aji_ctx::GpBuf& b, ncnn::VkAllocator* blob, ncnn::VkAllocator* staging,
                   const aji_ctx::Slot& s, ncnn::VkMat& out_model) {
    const ncnn::VulkanDevice* vkdev = c->models[0]->net.vulkan_device();
    if (!build_gp_incolor(c, s.in_w, s.in_h, s.in_format, s.in_matrix, s.in_range, s.in_siting)) return false;
    aji_ctx::GpShared& g = c->gp;
    const int W = s.in_w, H = s.in_h, cw = W >> 1, ch = H >> 1, bpp = (s.in_format == AJI_FMT_P010) ? 2 : 1;
    ncnn::Option popt; popt.use_vulkan_compute = true; popt.use_packing_layout = false;
    popt.use_fp16_packed = false; popt.use_fp16_storage = false; popt.use_fp16_arithmetic = false;
    popt.blob_vkallocator = blob; popt.workspace_vkallocator = blob; popt.staging_vkallocator = staging;
    if (!b.iready) {
        auto ib = [&](int w, int h, int cc_, ncnn::VkMat& v){ ncnn::Mat z(w, h, cc_); z.fill(0.f); v = ncnn::VkMat(); ncnn::VkCompute cc(vkdev); cc.record_upload(z, v, popt); cc.submit_and_wait(); };
        ib(W, ch, 1, b.it0u); ib(W, ch, 1, b.it0v); ib(W, H, 1, b.it1u); ib(W, H, 1, b.it1v); ib(W, H, 3, b.ivRGB);
        b.idH.create(W, ch, (size_t)4u, blob); b.idV.create(W, H, (size_t)4u, blob); b.idM.create(W, H, (size_t)4u, blob);
        b.iready = true;
    }
    // upload raw planes as 2D uint Mats (data linear; 2D shape lifts the 2^18 1D-VkMat read cap)
    int yuw = (W * bpp) / 4, uvuw = (cw * 2 * bpp) / 4;
    { ncnn::Mat ym(yuw, H, 1, (void*)s.rawY.data(), (size_t)4u); ncnn::VkCompute cc(vkdev); cc.record_upload(ym, b.ivRawY, popt); cc.submit_and_wait(); }
    { ncnn::Mat um(uvuw, ch, 1, (void*)s.rawUV.data(), (size_t)4u); ncnn::VkCompute cc(vkdev); cc.record_upload(um, b.ivRawUV, popt); cc.submit_and_wait(); }
    const int cstep = (int)b.ivRGB.cstep;
    const aji_csp& csp = g.icsp;
    ncnn::Option mopt; mopt.use_vulkan_compute = true; mopt.use_fp16_packed = true; mopt.use_fp16_storage = true;
    mopt.use_fp16_arithmetic = true; mopt.use_packing_layout = true;
    mopt.blob_vkallocator = blob; mopt.workspace_vkallocator = blob; mopt.staging_vkallocator = staging;
    { ncnn::VkCompute cmd(vkdev);
      { std::vector<ncnn::vk_constant_type> cc(5); cc[0].i=W;cc[1].i=ch;cc[2].i=g.iphtaps;cc[3].i=cw;cc[4].i=bpp;
        cmd.record_pipeline(g.ip_h, {b.ivRawUV, g.ivUPHs, g.ivUPHw, b.it0u, b.it0v}, cc, b.idH); }
      { std::vector<ncnn::vk_constant_type> cc(4); cc[0].i=W;cc[1].i=H;cc[2].i=g.ipvtaps;cc[3].i=ch;
        cmd.record_pipeline(g.ip_v, {b.it0u, g.ivUPVs, g.ivUPVw, b.it1u}, cc, b.idV);
        cmd.record_pipeline(g.ip_v, {b.it0v, g.ivUPVs, g.ivUPVw, b.it1v}, cc, b.idV); }
      { std::vector<ncnn::vk_constant_type> cc(10); cc[0].i=W;cc[1].i=H;cc[2].f=csp.kr;cc[3].f=csp.kb;cc[4].f=csp.yoff;cc[5].f=csp.yscale;cc[6].f=csp.coff;cc[7].f=csp.cscale;cc[8].i=bpp;cc[9].i=cstep;
        cmd.record_pipeline(g.ip_mat, {b.ivRawY, b.it1u, b.it1v, b.ivRGB}, cc, b.idM); }
      vkdev->convert_packing(b.ivRGB, b.ivModel, 1, 2, cmd, mopt);   // fp32 elempack1 -> fp16 (model input layout)
      if (cmd.submit_and_wait() != 0) { c->err = "gp in: submit"; return false; } }
    out_model = b.ivModel;
    return true;
}

// GPU-resident single-model path: rgb_in (host) -> model VkMat -> GPU post -> out planes.
// All on GPU; only the (already-quantized) ~12-50MB YUV crosses to the host.
static int run_gpu_post(aji_ctx* c, aji_ctx::GpBuf& b, ncnn::VkAllocator* blob, ncnn::VkAllocator* staging,
                        const ncnn::Mat& rgb_in, const aji_frame* out, int out_format, int out_matrix, int out_range,
                        aji_ctx::Slot& s) {
    const ncnn::VulkanDevice* vkdev = c->models[0]->net.vulkan_device();
    NcnnModel* nm = c->models[0].get();
    ncnn::Option mopt; mopt.use_vulkan_compute = true; mopt.use_fp16_packed = true; mopt.use_fp16_storage = true;
    mopt.use_fp16_arithmetic = true; mopt.use_packing_layout = true;
    mopt.blob_vkallocator = blob; mopt.workspace_vkallocator = blob; mopt.staging_vkallocator = staging;
    ncnn::Option popt; popt.use_vulkan_compute = true; popt.use_packing_layout = false;
    popt.use_fp16_packed = false; popt.use_fp16_storage = false; popt.use_fp16_arithmetic = false;  // post buffers must be plain fp32
    popt.blob_vkallocator = blob; popt.workspace_vkallocator = blob; popt.staging_vkallocator = staging;

    // Run the model to a HOST fp32 Mat (ncnn's extract-to-VkMat gives wrong output for
    // this gemm-CM model; extract-to-host is correct). Then upload that fp32 RGB to the
    // GPU and run the post there -> the expensive spline36 resample stays on the GPU,
    // only a small YUV comes back. (host RGB up = ~99MB; still far cheaper than the CPU
    // resample it replaces.)
    // Model output stays a VkMat on the GPU (fp16); the matrix kernel decodes it with
    // unpackHalf2x16. No host round-trip -> the GPU stays saturated and boosts.
    (void)mopt;

    aji_ctx::GpShared& g = c->gp;
    const int W2 = g.W2, H2 = g.H2, cw2 = W2 >> 1, ch2 = H2 >> 1;
    if (!b.ready || b.fp16 != g.fp16) {  // (re)build when precision changes (e.g. NV12<->P010)
        // chroma intermediates (Un/Vn/hu/hv) can be fp16 storage (halves resample bandwidth);
        // the downloaded outputs (Yout/uvout) stay fp32 — ncnn's fp16 VkMat->host download
        // forces an fp16 host buffer (convert_packing sets use_fp16_storage), which the fp32
        // pack loop would read out of bounds.
        ncnn::Option bopt = popt; if (g.fp16) bopt.use_fp16_storage = true;  // elempack-1 (c=1, no packing)
        auto ib  = [&](int w, int h, ncnn::VkMat& v, const ncnn::Option& o){ ncnn::Mat z(w, h, 1); z.fill(0.f); v = ncnn::VkMat(); ncnn::VkCompute cc(vkdev); cc.record_upload(z, v, o); cc.submit_and_wait(); };
        ib(W2, H2, b.Un, bopt); ib(W2, H2, b.Vn, bopt); ib(W2, H2, b.Yout, popt); ib(cw2, H2, b.hu, bopt); ib(cw2, H2, b.hv, bopt); ib(cw2 * 2, ch2, b.uvout, popt);
        b.dV.create(W2, H2, (size_t)4u, blob); b.dH.create(cw2, H2, (size_t)4u, blob); b.dUV.create(cw2, ch2, (size_t)4u, blob);
        b.ready = true;
    }
    const float qdiv = (out_format == AJI_FMT_P010) ? 64.0f : 1.0f;
    const float qmax = (out_format == AJI_FMT_P010) ? 1023.0f : 255.0f;
    // FUSED: model eval + GPU post-color + download in ONE command submission, so the GPU runs
    // them back-to-back with no host round-trip / idle gap between the model and the post kernels
    // (was two submit_and_wait per frame -> the GPU stalled between them).
    double tt0 = kTiming ? aji_now() : 0;
    // GPU in-color (AJI_VK_GPUINCOLOR): raw YUV -> RGB on the GPU, fed straight to the model as a
    // VkMat (no CPU spline36, no host RGB upload). Else the host rgb_in from the CPU gpu_pre.
    bool use_gpu_in = c->gpu_incolor && !s.rawY.empty();
    ncnn::VkMat vModelIn;
    if (use_gpu_in && !gpu_in(c, b, blob, staging, s, vModelIn)) { logmsg(c, 2, c->err.c_str()); return AJI_ERR; }
    ncnn::VkMat vRGB;
    { ncnn::VkCompute cmd(vkdev);
      { ncnn::Extractor ex = nm->net.create_extractor();
        ex.set_blob_vkallocator(blob); ex.set_workspace_vkallocator(blob); ex.set_staging_vkallocator(staging);
        if (use_gpu_in) ex.input(nm->in_name.c_str(), vModelIn); else ex.input(nm->in_name.c_str(), rgb_in);
        if (ex.extract(nm->out_name.c_str(), vRGB, cmd) != 0) { c->err = "gp: model extract"; return AJI_ERR; } }
      { std::vector<ncnn::vk_constant_type> cc(8); cc[0].i=W2;cc[1].i=H2;cc[2].f=g.csp.kr;cc[3].f=g.csp.kb;cc[4].f=g.csp.yoff;cc[5].f=g.csp.yscale;cc[6].f=qdiv;cc[7].f=qmax;
        cmd.record_pipeline(g.p_mat, {vRGB, b.Un, b.Vn, b.Yout}, cc, b.dV); }
      { std::vector<ncnn::vk_constant_type> cc(4); cc[0].i=cw2;cc[1].i=H2;cc[2].i=g.phtaps;cc[3].i=W2;
        cmd.record_pipeline(g.p_h, {b.Un, g.vPHs, g.vPHw, b.hu}, cc, b.dH); cmd.record_pipeline(g.p_h, {b.Vn, g.vPHs, g.vPHw, b.hv}, cc, b.dH); }
      { std::vector<ncnn::vk_constant_type> cc(9); cc[0].i=cw2;cc[1].i=ch2;cc[2].i=g.pvtaps;cc[3].i=H2;cc[4].f=g.csp.coff;cc[5].f=g.csp.cscale;cc[6].f=qdiv;cc[7].f=qmax;cc[8].i=0;
        cmd.record_pipeline(g.p_uvv, {b.hu, b.hv, g.vPVs, g.vPVw, b.uvout}, cc, b.dUV); }
      (void)blob;
      cmd.record_download(b.Yout, b.oy, popt); cmd.record_download(b.uvout, b.ouv, popt);
      if (cmd.submit_and_wait() != 0) { c->err = "gp: fused submit"; return AJI_ERR; } }
    double tt1 = kTiming ? aji_now() : 0;
    { static bool once=false; if(!once && getenv("AJI_VK_GPDBG")){ once=true;
        const float* yf=(const float*)b.oy.data; double s=0,mn=1e9,mx=-1e9; long t=(long)W2*H2,nz=0;
        for(long i=0;i<t;i++){float v=yf[i]; s+=v; if(v<mn)mn=v; if(v>mx)mx=v; if(v!=0)nz++;}
        fprintf(stderr,"[gp Yout] mean=%.2f min=%.1f max=%.1f nonzero=%.1f%% | oy.dims=%d w=%d h=%d ep=%d\n",s/t,mn,mx,100.0*nz/t,b.oy.dims,b.oy.w,b.oy.h,b.oy.elempack); } }

    // pack the already-quantized fp32 Y / interleaved-UV to the host out planes
    const float* yf = (const float*)b.oy.data; const float* uvf = (const float*)b.ouv.data;
    if (out_format == AJI_FMT_P010) {
        uint16_t* yp = (uint16_t*)out->plane[0]; uint16_t* uvp = (uint16_t*)out->plane[1];
        ptrdiff_t ys = out->stride[0] ? out->stride[0] : (ptrdiff_t)W2 * 2;
        ptrdiff_t uvs = out->stride[1] ? out->stride[1] : (ptrdiff_t)cw2 * 2 * 2;
        for (int y = 0; y < H2; y++) { uint16_t* row = (uint16_t*)((char*)yp + (ptrdiff_t)y * ys); for (int x = 0; x < W2; x++) row[x] = (uint16_t)yf[(size_t)y * W2 + x]; }
        for (int y = 0; y < ch2; y++) { uint16_t* row = (uint16_t*)((char*)uvp + (ptrdiff_t)y * uvs); for (int x = 0; x < cw2 * 2; x++) row[x] = (uint16_t)uvf[(size_t)y * cw2 * 2 + x]; }
    } else {
        uint8_t* yp = (uint8_t*)out->plane[0]; uint8_t* uvp = (uint8_t*)out->plane[1];
        ptrdiff_t ys = out->stride[0] ? out->stride[0] : (ptrdiff_t)W2;
        ptrdiff_t uvs = out->stride[1] ? out->stride[1] : (ptrdiff_t)cw2 * 2;
        for (int y = 0; y < H2; y++) { uint8_t* row = yp + (ptrdiff_t)y * ys; for (int x = 0; x < W2; x++) row[x] = (uint8_t)yf[(size_t)y * W2 + x]; }
        for (int y = 0; y < ch2; y++) { uint8_t* row = uvp + (ptrdiff_t)y * uvs; for (int x = 0; x < cw2 * 2; x++) row[x] = (uint8_t)uvf[(size_t)y * cw2 * 2 + x]; }
    }
    if (kTiming) { static double sg=0,sk=0; static long sn=0;
        sg+=(tt1-tt0)*1000; sk+=(aji_now()-tt1)*1000; sn++;
        if (sn%60==0) fprintf(stderr,"[gp] fused gpu(model+post+dl)=%.2fms pack=%.2fms (W=%d)\n", sg/sn,sk/sn,c->num_workers); }
    return AJI_OK;
}

// Worker: pops submitted slots and runs the model back-to-back, so the GPU stays
// saturated (the amdgpu governor only boosts clocks under sustained load) while the
// filter thread does CPU color for already-finished frames.
static void worker_loop(aji_ctx* c, int widx) {
    // Per-worker Vulkan allocators (blob allocators are not thread-safe). Acquired
    // lazily once the models exist (worker starts after aji_configure loads them).
    const ncnn::VulkanDevice* vkdev = c->models.empty() ? nullptr : c->models[0]->net.vulkan_device();
    ncnn::VkAllocator* blob = vkdev ? vkdev->acquire_blob_allocator() : nullptr;
    ncnn::VkAllocator* staging = vkdev ? vkdev->acquire_staging_allocator() : nullptr;
    aji_ctx::Scratch& scr = c->wscratch[widx];
    for (;;) {
        int idx;
        {
            std::unique_lock<std::mutex> lk(c->amx);
            c->work_cv.wait(lk, [&] { return c->worker_stop || !c->workq.empty(); });
            if (c->worker_stop && c->workq.empty()) break;
            idx = c->workq.front(); c->workq.pop();
            c->ring[idx].state = aji_ctx::SLOT_RUNNING;
        }
        aji_ctx::Slot& s = c->ring[idx];
        // Both model and color run on the worker (parallel across workers; never blocks
        // the filter submit loop). Default = GPU-resident color (model output stays a
        // VkMat -> GPU post -> small YUV download); CPU color is the fallback for chains
        // or AJI_VK_CPUCOLOR=1.
        bool use_gp = c->gpu_color && c->steps.size() == 1 && c->steps[0].kind == Step::MODEL
                      && build_gp_shared(c, c->out_w, c->out_h, s.out_format, s.out_matrix, s.out_range);
        if (use_gp) {
            run_gpu_post(c, c->gpbuf[widx], blob, staging, s.rgb_in, &s.out_frame, s.out_format, s.out_matrix, s.out_range, s);
        } else {
            run_chain(c, s.rgb_in, s.rgb_out, blob, staging);
            gpu_post(c, s.rgb_out, s.out_format, s.out_matrix, s.out_range, &s.out_frame, scr);
        }
        {
            std::unique_lock<std::mutex> lk(c->amx);
            s.state = aji_ctx::SLOT_DONE;
            c->done_cv.notify_all();
        }
    }
    if (vkdev) { if (blob) vkdev->reclaim_blob_allocator(blob); if (staging) vkdev->reclaim_staging_allocator(staging); }
}

static void ensure_worker(aji_ctx* c) {
    if (c->worker_started || c->sync_mode) return;
    // Multiple concurrent model workers keep the GPU saturated so the amdgpu governor
    // boosts clocks (single-worker model+download leaves it ~half-clocked). Resize
    // steps share c->scratch.rt0 (not thread-safe) -> force 1 worker if any RESIZE.
    int n = c->num_workers;
    for (const Step& st : c->steps) if (st.kind == Step::RESIZE) { n = 1; break; }
    if (n < 1) n = 1;
    c->worker_stop = false;
    c->workers.clear();
    c->wscratch.resize(n);
    c->gpbuf.clear(); c->gpbuf.resize(n);   // per-worker GPU-post buffers
    for (int i = 0; i < n; i++) c->workers.emplace_back(worker_loop, c, i);
    c->worker_started = true;
}

static void stop_worker(aji_ctx* c) {
    if (!c->worker_started) return;
    { std::unique_lock<std::mutex> lk(c->amx); c->worker_stop = true; c->work_cv.notify_all(); }
    for (auto& t : c->workers) if (t.joinable()) t.join();
    c->workers.clear();
    c->worker_started = false;
    std::queue<int> empty; std::swap(c->workq, empty);
    for (auto& s : c->ring) { s.state = aji_ctx::SLOT_FREE; s.ticket = 0; }
}

extern "C" {

AJI_EXPORT aji_ctx* aji_create(const aji_create_params* p) {
    if (!p || p->api_version < 7) return nullptr;
    aji_ctx* c = new aji_ctx();
    c->logfn = p->log;
    c->log_opaque = p->log_opaque;
#ifdef _OPENMP
    // CPU color threads: more help the big 4K out-color; AJI_VK_NTHREADS overrides.
    c->nthreads = std::max(1, std::min(16, omp_get_max_threads()));
    if (getenv("AJI_VK_NTHREADS")) c->nthreads = std::max(1, atoi(getenv("AJI_VK_NTHREADS")));
#endif
    c->sync_mode = getenv("AJI_VK_SYNC") != nullptr;   // async overlap pipeline is the default
    // Default to several concurrent model workers: a single worker runs model->GPU-post->blocking
    // download serially, idling the GPU each frame and leaving the RADV governor unboosted (~14fps
    // at 1080p->4K). Multiple workers overlap frames -> GPU saturates -> sclk boosts (~24-27fps,
    // real-time). ensure_worker() force-clamps to 1 for RESIZE chains (shared scratch), so this
    // only engages on the single-MODEL GPU-resident path. AJI_VK_WORKERS overrides (1-2 for
    // low-VRAM GPUs; each worker holds its own full-res post buffers + model activations).
    c->num_workers = getenv("AJI_VK_WORKERS") ? std::max(1, atoi(getenv("AJI_VK_WORKERS"))) : 4;
    // GPU-resident color is the default (bit-exact vs the CPU resample.h path, and keeps
    // the GPU saturated so it boosts). AJI_VK_CPUCOLOR=1 forces the CPU fallback.
    c->gpu_color = getenv("AJI_VK_CPUCOLOR") == nullptr;
    c->gpu_incolor = getenv("AJI_VK_GPUINCOLOR") != nullptr;   // GPU-resident in-color (opt-in; frees the filter thread)
    c->rife_model_dir = p->rife_model_dir ? p->rife_model_dir : "";   // dir with rife_v*.param/.bin (NULL disables RIFE)

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
    stop_worker(c);   // no model run may be in flight while build_plan rebuilds models
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

    const int bpp = (format == AJI_FMT_P010) ? 2 : 1;
    const bool gin = c->gpu_incolor && !c->sync_mode;   // GPU in-color: only on the async player path
    std::vector<uint8_t> rawY, rawUV;
    ncnn::Mat rgb_in;
    double t_pre0 = kTiming ? aji_now() : 0;
    if (gin) {
        // GPU in-color: copy the raw source planes contiguously (cheap) — the GPU does the
        // spline36 chroma upsample + YUV->RGB. No CPU resample on the filter thread.
        const int yrow = W * bpp, uvrow = cw * 2 * bpp;
        rawY.resize((size_t)yrow * H); rawUV.resize((size_t)uvrow * ch);
        const uint8_t* yb = (const uint8_t*)in->plane[0]; const uint8_t* uvb = (const uint8_t*)in->plane[1];
        ptrdiff_t ys = in->stride[0] ? in->stride[0] : yrow;
        ptrdiff_t uvs = in->stride[1] ? in->stride[1] : uvrow;
        for (int y = 0; y < H; y++)  memcpy(&rawY[(size_t)y * yrow], yb + (ptrdiff_t)y * ys, yrow);
        for (int y = 0; y < ch; y++) memcpy(&rawUV[(size_t)y * uvrow], uvb + (ptrdiff_t)y * uvs, uvrow);
    } else {
        // ---- CPU in-color: read host YUV planes into fp32 -> gpu_pre (resample.h, bit-exact) ----
        std::vector<float>& Yf = c->scratch.Yf; Yf.resize((size_t)W * H);
        std::vector<float>& Uf = c->scratch.Uf; Uf.resize((size_t)cw * ch);
        std::vector<float>& Vf = c->scratch.Vf; Vf.resize((size_t)cw * ch);
        ptrdiff_t ys = in->stride[0] ? in->stride[0] : (ptrdiff_t)W * bpp;
        ptrdiff_t uvs = in->stride[1] ? in->stride[1] : (ptrdiff_t)cw * 2 * bpp;
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
        if (gpu_pre(c, W, H, format, matrix, range, siting, Yf, Uf, Vf, rgb_in) != AJI_OK) {
            logmsg(c, 2, c->err.c_str()); return AJI_ERR;
        }
    }

    if (c->sync_mode) {
        // ---- synchronous path (A/B + non-pipelining callers): model + out-color inline ----
        double t_model0 = kTiming ? aji_now() : 0;
        ncnn::Mat rgb_out;
        if (run_chain(c, rgb_in, rgb_out) != AJI_OK) { logmsg(c, 2, c->err.c_str()); return AJI_ERR; }
        double t_post0 = kTiming ? aji_now() : 0;
        if (gpu_post(c, rgb_out, out_format, out_matrix, out_range, out, c->scratch) != AJI_OK) {
            logmsg(c, 2, c->err.c_str()); return AJI_ERR;
        }
        if (kTiming) {
            double t_end = aji_now();
            static double s_pre = 0, s_model = 0, s_post = 0; static int s_n = 0;
            s_pre += (t_model0 - t_pre0) * 1000; s_model += (t_post0 - t_model0) * 1000; s_post += (t_end - t_post0) * 1000; s_n++;
            if (s_n % 30 == 0) {
                char m[160]; snprintf(m, sizeof m, "[aji_vk timing n=%d mean ms] in-color=%.2f  model=%.2f  out-color=%.2f  total=%.2f",
                         s_n, s_pre/s_n, s_model/s_n, s_post/s_n, (s_pre+s_model+s_post)/s_n);
                fprintf(stderr, "%s\n", m);
            }
        }
        return AJI_OK;
    }

    // ---- async submit: hand rgb_in to the worker, remember the out frame, return ----
    ensure_worker(c);
    int idx = -1;
    {
        std::unique_lock<std::mutex> lk(c->amx);
        c->free_cv.wait(lk, [&] { for (int i = 0; i < aji_ctx::kRing; i++) if (c->ring[i].state == aji_ctx::SLOT_FREE) return true; return false; });
        for (int i = 0; i < aji_ctx::kRing; i++) if (c->ring[i].state == aji_ctx::SLOT_FREE) { idx = i; c->ring[i].state = aji_ctx::SLOT_SUBMITTED; break; }
    }
    aji_ctx::Slot& s = c->ring[idx];
    s.rgb_in = rgb_in;            // shared ref; worker reads only after enqueue (happens-before via mutex)
    s.rgb_out = ncnn::Mat();
    s.rawY = std::move(rawY); s.rawUV = std::move(rawUV);   // GPU in-color: raw source planes (empty if CPU in-color)
    s.in_format = format; s.in_matrix = matrix; s.in_range = range; s.in_siting = siting; s.in_w = W; s.in_h = H;
    s.out_frame = *out; s.out_format = out_format; s.out_matrix = out_matrix; s.out_range = out_range;
    {
        std::unique_lock<std::mutex> lk(c->amx);
        s.ticket = ++c->ticket_counter;
        c->workq.push(idx);
        c->work_cv.notify_one();
    }
    return AJI_OK;
}

// aji_flush: marker after all work submitted so far. Returns the latest ticket (0 if
// nothing pending). aji_wait(ticket) blocks for the model then does the out-color.
AJI_EXPORT uint64_t aji_flush(aji_ctx* c, void*) {
    if (!c || c->sync_mode) return 1;
    std::unique_lock<std::mutex> lk(c->amx);
    return c->ticket_counter;
}
AJI_EXPORT int aji_done(aji_ctx* c, uint64_t ticket) {
    if (!c || c->sync_mode || ticket == 0) return 1;
    std::unique_lock<std::mutex> lk(c->amx);
    for (int i = 0; i < aji_ctx::kRing; i++)
        if (c->ring[i].ticket == ticket && c->ring[i].state != aji_ctx::SLOT_FREE)
            return c->ring[i].state == aji_ctx::SLOT_DONE ? 1 : 0;
    return 1;   // not found -> already waited/freed -> complete
}
AJI_EXPORT int aji_wait(aji_ctx* c, uint64_t ticket) {
    if (!c || c->sync_mode || ticket == 0) return AJI_OK;
    int idx = -1;
    {
        std::unique_lock<std::mutex> lk(c->amx);
        for (int i = 0; i < aji_ctx::kRing; i++)
            if (c->ring[i].ticket == ticket && c->ring[i].state != aji_ctx::SLOT_FREE) { idx = i; break; }
        if (idx < 0) return AJI_OK;   // already completed + freed
        c->done_cv.wait(lk, [&] { return c->ring[idx].state == aji_ctx::SLOT_DONE; });
        // the worker already did model + color into the out frame; just release the slot
        c->ring[idx].state = aji_ctx::SLOT_FREE; c->ring[idx].ticket = 0;
        c->free_cv.notify_one();
    }
    return AJI_OK;
}

AJI_EXPORT const char* aji_current_log(aji_ctx* c) { return c ? c->log.c_str() : ""; }
AJI_EXPORT int aji_scale_factor(aji_ctx* c) { return c ? c->chain_scale : 0; }
// RIFE factor of the active chain: 1 + num/den iff a RIFE engine is actually loaded (so the
// filter only calls aji_infer_rife once interpolation is live), else 0.
AJI_EXPORT int aji_rife_factor(aji_ctx* c, int* num, int* den) {
    if (!c || !c->rife.enabled || !c->rife.loaded) return 0;
    if (num) *num = c->rife.num;
    if (den) *den = c->rife.den;
    return 1;
}

// RIFE ordering: 1 if interpolation runs at the SOURCE resolution (caller feeds source pairs),
// 0 otherwise (RIFE inactive, or it runs on the upscaled output).
AJI_EXPORT int aji_rife_before_upscale(aji_ctx* c) {
    return (c && c->rife.enabled && c->rife.before_upscale) ? 1 : 0;
}

// aji_vk loads RIFE synchronously (ncnn has no engine-compile step), so there is never a
// deferred build to report.
AJI_EXPORT int aji_poll(aji_ctx*) { return 0; }

// Per-frame RIFE interpolation (4:2:0, SYNCHRONOUS). Produces the frame at timestep t in (0,1)
// between A and B. Color is forced BT.709 inside rife_cpu (range follows the source); the model
// (incl. the GridSample warps) runs GPU-native on ncnn-Vulkan. Mirrors aji_rocm::aji_infer_rife
// but ncnn handles the fp32->fp16 cast, H2D, eval and D2H internally. Returns:
//   AJI_OK    — `out` written.  AJI_SCENE — scene change, `out` untouched (caller dups A).
//   AJI_ERR_FORMAT / AJI_ERR_SHAPE / AJI_ERR — invalid input or eval failure.
AJI_EXPORT int aji_infer_rife(aji_ctx* c, const aji_frame* a, const aji_frame* b,
                              double t, const aji_frame* out, void* /*stream*/) {
    if (!c || !c->rife.loaded) { if (c) c->err = "rife not loaded"; return AJI_ERR; }
    auto& R = c->rife;

    // 1. Validate: formats equal and 4:2:0; dims equal the RIFE geometry's source w x h.
    if (a->format != b->format || a->format != out->format) { c->err = "rife fmt mismatch"; return AJI_ERR_FORMAT; }
    if (a->format != AJI_FMT_NV12 && a->format != AJI_FMT_P010) { c->err = "rife needs nv12/p010"; return AJI_ERR_FORMAT; }
    if (a->width != R.g.w || a->height != R.g.h || b->width != R.g.w || b->height != R.g.h ||
        out->width != R.g.w || out->height != R.g.h) { c->err = "rife dim mismatch"; return AJI_ERR_SHAPE; }

    const int pw = R.g.pw, ph = R.g.ph;
    const size_t plane = (size_t)pw * ph;
    double tm0 = kTiming ? aji_now() : 0;

    // 2. Scene detect on UNPADDED luma (raw container values). A scene change skips
    // interpolation: return AJI_SCENE with `out` untouched (caller duplicates A).
    const double norm = (a->format == AJI_FMT_P010) ? 1.0 / 65472.0 : 1.0 / 255.0;
    bool scene = (a->format == AJI_FMT_P010)
        ? rife_cpu::scene_detect((const uint16_t*)a->plane[0], a->stride[0],
                                 (const uint16_t*)b->plane[0], b->stride[0], R.g.w, R.g.h, pw, ph, norm, R.scd_threshold)
        : rife_cpu::scene_detect((const uint8_t*)a->plane[0], a->stride[0],
                                 (const uint8_t*)b->plane[0], b->stride[0], R.g.w, R.g.h, pw, ph, norm, R.scd_threshold);
    if (scene) return AJI_SCENE;

    // 3. Assemble the 11-ch fp32 tensor. Consts (ch7-10) are already in R.assembly. A -> ch0-2,
    // B -> ch3-5 (BT.709 + bilinear chroma upsample, centered into pw x ph with black borders);
    // ch6 <- the timestep plane.
    rife_cpu::yuv420_to_rgb_planes(*a, R.g, R.assembly.data(), 0, (aji_range)a->range);
    rife_cpu::yuv420_to_rgb_planes(*b, R.g, R.assembly.data(), 1, (aji_range)a->range);
    float* ch6 = R.assembly.data() + 6 * plane;
    for (size_t i = 0; i < plane; ++i) ch6[i] = (float)t;
    double tm1 = kTiming ? aji_now() : 0;

    // 4. Run the RIFE net on ncnn-Vulkan: 11-ch planar fp32 in -> 3-ch RGB out. ncnn does the
    // fp32->fp16 cast, upload, GPU eval (warps via gridsample_vulkan) and download.
    ncnn::Mat in(pw, ph, 11);
    for (int k = 0; k < 11; ++k) memcpy(in.channel(k), R.assembly.data() + (size_t)k * plane, plane * sizeof(float));
    ncnn::Mat o;
    {
        ncnn::Extractor ex = R.model->net.create_extractor();
        if (ex.input(R.model->in_name.c_str(), in) != 0) { c->err = "rife input failed"; return AJI_ERR; }
        if (ex.extract(R.model->out_name.c_str(), o) != 0) { c->err = "rife extract failed"; return AJI_ERR; }
    }
    double tm2 = kTiming ? aji_now() : 0;

    // 5. Flatten to a contiguous 3-ch planar buffer, then crop the centered w x h window into
    // `out` (BT.709 RGB->YUV + bilinear chroma downsample, range follows A).
    if (o.w != pw || o.h != ph || o.c < 3) {
        char eb[160]; snprintf(eb, sizeof eb, "rife: unexpected model output dims w=%d h=%d c=%d dims=%d elempack=%d (want %dx%dx>=3)",
                               o.w, o.h, o.c, o.dims, o.elempack, pw, ph);
        c->err = eb; return AJI_ERR_SHAPE;
    }
    ncnn::Mat o1;
    if (o.elempack != 1) ncnn::convert_packing(o, o1, 1); else o1 = o;
    std::vector<float> rgb(3 * plane);
    for (int k = 0; k < 3; ++k) memcpy(rgb.data() + (size_t)k * plane, o1.channel(k), plane * sizeof(float));
    rife_cpu::rgb_planes_to_yuv420(rgb.data(), R.g, *(aji_frame*)out, (aji_range)a->range);

    if (kTiming) {
        static double s_pre = 0, s_mdl = 0, s_post = 0; static int s_n = 0;
        s_pre += (tm1 - tm0) * 1000; s_mdl += (tm2 - tm1) * 1000; s_post += (aji_now() - tm2) * 1000; s_n++;
        if (s_n % 50 == 0) fprintf(stderr, "[rife n=%d] color_pre=%.2f model=%.2f color_post=%.2f ms\n",
                                   s_n, s_pre / s_n, s_mdl / s_n, s_post / s_n);
    }
    return AJI_OK;
}

AJI_EXPORT const char* aji_last_error(aji_ctx* c) { return c ? c->err.c_str() : "null ctx"; }

AJI_EXPORT void aji_destroy(aji_ctx** c) {
    if (c && *c) { stop_worker(*c); color_free((*c)->color); delete *c; *c = nullptr; }
}

} // extern "C"

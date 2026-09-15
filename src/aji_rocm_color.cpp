// Host side of the GPU-resident color kernels. The device kernels (aji_rocm_color_
// device.hip, embedded as AJI_ROCM_COLOR_SRC) are compiled for the GPU actually
// present with hipRTC at first use and launched through the HIP module API. This is
// what makes libaji_rocm.so portable across AMD archs (no baked --offload-arch),
// mirroring MIGraphX's per-device .mxr JIT. Compiled code objects are cached to
// animejanai/cache/ (or $AJI_ROCM_CACHE_DIR) and self-heal on any load failure.
#include "aji_rocm_color.h"
#include "aji_rocm_color_device_src.h"   // generated: AJI_ROCM_COLOR_SRC[]
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>

namespace {

struct ColorModule {
    bool ok = false;
    hipModule_t mod = nullptr;
    hipFunction_t selftest = nullptr, y_uvdiff = nullptr, chroma_h = nullptr, chroma_v = nullptr;
    hipFunction_t pre444 = nullptr, post444 = nullptr;   // Phase B 4:4:4
    std::string err;
};

// FNV-1a over the embedded source + a version tag; invalidates stale .co on a kernel edit.
std::string src_hash() {
    unsigned long long h = 1469598103934665603ULL;
    const char* p = AJI_ROCM_COLOR_SRC;
    const char* tag = "v2";                 // bump on launch-convention changes (v2: +pre444/post444)
    for (const char* t = tag; *t; t++) { h ^= (unsigned char)*t; h *= 1099511628211ULL; }
    for (; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
    char buf[17]; snprintf(buf, sizeof buf, "%016llx", h);
    return buf;
}

// Cache dir: $AJI_ROCM_CACHE_DIR, else <dir-of-libaji_rocm.so>/../cache, else /tmp.
std::string cache_dir() {
    if (const char* e = getenv("AJI_ROCM_CACHE_DIR")) return e;
    Dl_info info{};
    if (dladdr((void*)&src_hash, &info) && info.dli_fname) {
        std::string so = info.dli_fname;
        size_t slash = so.find_last_of('/');
        std::string dir = (slash == std::string::npos) ? "." : so.substr(0, slash);
        return dir + "/../cache";
    }
    return "/tmp";
}

std::string cache_path(const std::string& arch) {
    return cache_dir() + "/aji_color." + arch + "." + src_hash() + ".co";
}

bool read_file(const std::string& p, std::vector<char>& out) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg(); if (n <= 0) return false;
    out.resize((size_t)n); f.seekg(0);
    return (bool)f.read(out.data(), n);
}

void write_file_atomic(const std::string& p, const std::vector<char>& data) {
    // best-effort: mkdir the cache dir, write tmp, rename. Failure just means recompile next time.
    std::string dir = p.substr(0, p.find_last_of('/'));
    std::error_code ec; std::filesystem::create_directories(dir, ec);  // best-effort; ofstream open below fails -> recompile next time
    std::string tmp = p + ".tmp";
    { std::ofstream f(tmp, std::ios::binary); if (!f) return; f.write(data.data(), (std::streamsize)data.size()); }
    if (std::rename(tmp.c_str(), p.c_str()) != 0) std::remove(tmp.c_str());
}

// Compile AJI_ROCM_COLOR_SRC for the current device and resolve the kernels.
ColorModule compile_module() {
    ColorModule m;
    hipDeviceProp_t prop{};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        m.err = "hipGetDeviceProperties failed"; return m;
    }
    std::string arch = prop.gcnArchName;   // e.g. "gfx1201" / "gfx1100:xnack-"
    const std::string co = cache_path(arch);

    // Resolve a loaded module's functions into m. Returns true on full success.
    auto resolve = [&]() -> bool {
        auto fn = [&](hipFunction_t* f, const char* name) {
            return hipModuleGetFunction(f, m.mod, name) == hipSuccess;
        };
        return fn(&m.selftest, "k_selftest") &&
               fn(&m.y_uvdiff, "k_out_y_uvdiff") &&
               fn(&m.chroma_h, "k_out_chroma_h") &&
               fn(&m.chroma_v, "k_out_chroma_v") &&
               fn(&m.pre444, "k_pre444") &&
               fn(&m.post444, "k_post444");
    };

    // 1) Try the cached code object. On ANY load failure, delete it and recompile.
    std::vector<char> code;
    if (read_file(co, code)) {
        if (hipModuleLoadData(&m.mod, code.data()) == hipSuccess) {
            if (resolve()) { m.ok = true; return m; }
            hipModuleUnload(m.mod); m.mod = nullptr;
        }
        std::remove(co.c_str());   // stale/incompatible: heal by recompiling
        code.clear();
    }

    // 2) Compile with hipRTC, then cache the bytes.
    hiprtcProgram prog{};
    if (hiprtcCreateProgram(&prog, AJI_ROCM_COLOR_SRC, "aji_rocm_color.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
        m.err = "hiprtcCreateProgram failed"; return m;
    }
    // Match the former static hipcc build: default fp contraction, -O3, target the
    // present arch. (No fast-math; the kernels rely on rintf/IEEE rounding.)
    std::string archopt = "--offload-arch=" + arch;
    const char* opts[] = { archopt.c_str(), "-O3" };
    hiprtcResult cr = hiprtcCompileProgram(prog, 2, opts);
    if (cr != HIPRTC_SUCCESS) {
        size_t lsz = 0; hiprtcGetProgramLogSize(prog, &lsz);
        std::string log(lsz, '\0'); if (lsz) hiprtcGetProgramLog(prog, &log[0]);
        m.err = std::string("hiprtcCompileProgram failed (") + arch + "): " + log;
        hiprtcDestroyProgram(&prog); return m;
    }
    size_t csz = 0; hiprtcGetCodeSize(prog, &csz);
    code.resize(csz);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    write_file_atomic(co, code);

    if (hipModuleLoadData(&m.mod, code.data()) != hipSuccess) { m.err = "hipModuleLoadData failed"; return m; }
    if (!resolve()) { m.err = "hipModuleGetFunction failed"; return m; }
    m.ok = true;
    return m;
}

ColorModule& module() {
    static ColorModule m = compile_module();   // function-local static: thread-safe init
    return m;
}

// Launch helper: kernelParams is an array of pointers to each argument.
inline hipError_t launch(hipFunction_t f, dim3 grid, dim3 block, hipStream_t s, void** args) {
    return hipModuleLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                                 0, s, args, nullptr);
}

} // namespace

extern "C" int aji_color_selftest(void) {
    ColorModule& m = module();
    if (!m.ok) { fprintf(stderr, "[aji_color] JIT failed: %s\n", m.err.c_str()); return -10; }
    const int n = 1024;
    float* d = nullptr;
    if (hipMalloc(&d, n * sizeof(float)) != hipSuccess) return -1;
    if (hipMemset(d, 0, n * sizeof(float)) != hipSuccess) { hipFree(d); return -1; }
    int nn = n;
    void* args[] = { &d, &nn };
    if (launch(m.selftest, dim3((n + 255) / 256), dim3(256), nullptr, args) != hipSuccess) { hipFree(d); return -4; }
    if (hipDeviceSynchronize() != hipSuccess) { hipFree(d); return -3; }
    float h0 = -1.0f;
    hipMemcpy(&h0, d, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d);
    return (h0 == 1.0f) ? 0 : -2;
}

extern "C" void aji_gpu_out_color(const void* rgb_fp16, int W, int H, aji_color_csp csp,
                                  const int* ph_start, const float* ph_wt, int ph_taps,
                                  const int* pv_start, const float* pv_wt, int pv_taps,
                                  float* Un, float* Vn, float* hu, float* hv,
                                  void* yplane, void* uvplane, void* stream) {
    ColorModule& m = module();
    if (!m.ok) { fprintf(stderr, "[aji_color] JIT failed: %s\n", m.err.c_str()); return; }
    hipStream_t s = (hipStream_t)stream;
    const int cw = W >> 1, ch = H >> 1;
    const long n = (long)W * H;
    const int bs = 256;

    // Pass 1
    void* a1[] = { (void*)&rgb_fp16, &W, &H, &csp, &yplane, &Un, &Vn };
    launch(m.y_uvdiff, dim3((unsigned)((n + bs - 1) / bs)), dim3(bs), s, a1);
    // Pass 2
    void* a2[] = { &Un, &Vn, &W, &H, (void*)&cw, (void*)&ph_start, (void*)&ph_wt, &ph_taps, &hu, &hv };
    launch(m.chroma_h, dim3((cw + 15) / 16, (H + 15) / 16), dim3(16, 16), s, a2);
    // Pass 3
    void* a3[] = { &hu, &hv, (void*)&cw, &H, (void*)&ch, (void*)&pv_start, (void*)&pv_wt, &pv_taps, &csp, &uvplane };
    launch(m.chroma_v, dim3((cw + 15) / 16, (ch + 15) / 16), dim3(16, 16), s, a3);
}

// ---- 4:4:4 (Phase B): pure-matrix, no chroma resample (one block grid each) ----
// post444: model output RGB fp16 (device, NCHW {3,H,W}) -> three full-res u16 planes.
// pre444:  three full-res u16 planes -> RGB fp16 (RIFE input). ys = Y byte stride,
// cs = Cb/Cr byte stride. All pointers device.
extern "C" void aji_gpu_post444(const void* rgb_fp16, int w, int h, aji_color_csp csp,
                                void* yplane, void* cbplane, void* crplane,
                                ptrdiff_t ys, ptrdiff_t cs, void* stream) {
    ColorModule& m = module();
    if (!m.ok) { fprintf(stderr, "[aji_color] JIT failed: %s\n", m.err.c_str()); return; }
    long ysl = (long)ys, csl = (long)cs;
    void* args[] = { (void*)&rgb_fp16, &w, &h, &csp, &yplane, &ysl, &cbplane, &crplane, &csl };
    launch(m.post444, dim3((w + 31) / 32, (h + 7) / 8), dim3(32, 8), (hipStream_t)stream, args);
}

extern "C" void aji_gpu_pre444(const void* yplane, const void* cbplane, const void* crplane,
                               ptrdiff_t ys, ptrdiff_t cs, int w, int h, aji_color_csp csp,
                               void* rgb_fp16, void* stream) {
    ColorModule& m = module();
    if (!m.ok) { fprintf(stderr, "[aji_color] JIT failed: %s\n", m.err.c_str()); return; }
    long ysl = (long)ys, csl = (long)cs;
    void* args[] = { (void*)&yplane, &ysl, (void*)&cbplane, (void*)&crplane, &csl, &w, &h, &csp, &rgb_fp16 };
    launch(m.pre444, dim3((w + 31) / 32, (h + 7) / 8), dim3(32, 8), (hipStream_t)stream, args);
}

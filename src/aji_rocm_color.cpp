// Host side of the GPU-resident color kernels. The device kernels (aji_rocm_color_
// device.hip, embedded as AJI_ROCM_COLOR_SRC) are compiled for the GPU actually
// present with hipRTC at first use and launched through the HIP module API. This is
// what makes libaji_rocm.so portable across AMD archs (no baked --offload-arch),
// mirroring MIGraphX's per-device .mxr JIT. Disk caching is added in a later step.
#include "aji_rocm_color.h"
#include "aji_rocm_color_device_src.h"   // generated: AJI_ROCM_COLOR_SRC[]
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <mutex>
#include <string>
#include <vector>
#include <cstdio>

namespace {

struct ColorModule {
    bool ok = false;
    hipModule_t mod = nullptr;
    hipFunction_t selftest = nullptr, y_uvdiff = nullptr, chroma_h = nullptr, chroma_v = nullptr;
    std::string err;
};

// Compile AJI_ROCM_COLOR_SRC for the current device and resolve the kernels.
ColorModule compile_module() {
    ColorModule m;
    hipDeviceProp_t prop{};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        m.err = "hipGetDeviceProperties failed"; return m;
    }
    std::string arch = prop.gcnArchName;   // e.g. "gfx1201" / "gfx1100:xnack-"

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
    std::vector<char> code(csz);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    if (hipModuleLoadData(&m.mod, code.data()) != hipSuccess) { m.err = "hipModuleLoadData failed"; return m; }
    auto fn = [&](hipFunction_t* f, const char* name) {
        return hipModuleGetFunction(f, m.mod, name) == hipSuccess;
    };
    if (!fn(&m.selftest, "k_selftest") || !fn(&m.y_uvdiff, "k_out_y_uvdiff") ||
        !fn(&m.chroma_h, "k_out_chroma_h") || !fn(&m.chroma_v, "k_out_chroma_v")) {
        m.err = "hipModuleGetFunction failed"; return m;
    }
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

# ROCm color-kernel JIT portability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `libaji_rocm.so` run on any AMD GPU (not just the build machine's `gfx1201`) by compiling the HIP color kernels on the user's machine at runtime with hipRTC, instead of baking a single GPU arch into the shared library at build time.

**Architecture:** The hand-written color kernels (`aji_rocm_color.hip`) are the only GPU-arch-locked code in the ROCm backend — MIGraphX already JIT-compiles the model per device. We turn the kernel device source into an embedded string, compile it for the present GPU with hipRTC on first use (mirroring MIGraphX's per-device `.mxr` flow), cache the resulting code object on disk, and launch the kernels through the HIP module API. The `extern "C"` color ABI (`aji_rocm_color.h`) is unchanged, so `aji_rocm.cpp` call sites are untouched. The result is a `libaji_rocm.so` with **zero embedded device code**.

**Tech Stack:** C++17, HIP runtime (`libamdhip64`), hipRTC (`libhiprtc`), MIGraphX, CMake. ROCm 7.2.4 on the dev box (`gfx1201`, RX 9070 XT).

## Global Constraints

- **No CPU color in the normal path.** The existing CPU color path stays only as a catastrophic-failure safety net; GPU JIT is the operating path.
- **Bit-exact color.** JIT'd kernel output must match the current static-`hipcc` build's NV12/P010 output byte-for-byte at ≤720p (the project's existing GPU-vs-CPU `maxdiff=0` bar). Carry matching fp flags into hipRTC.
- **Scope = any AMD GPU + a compatible ROCm.** Do not bundle ROCm, do not add NVIDIA/Intel paths, do not touch the `.mxr` cache keying.
- **Color ABI is frozen.** `aji_rocm_color.h` keeps `aji_color_csp`, `aji_color_selftest`, `aji_gpu_out_color` with identical signatures. Adding a new `extern "C"` symbol is allowed only if additive.
- **Cache location:** code objects live in a `cache/` dir resolved as the sibling of `libaji_rocm.so`'s directory (`animejanai/inference/` → `animejanai/cache/`), overridable via env `AJI_ROCM_CACHE_DIR`. File name: `aji_color.<gcnArchName>.<srcHash>.co`.
- ROCm libs: `libhiprtc.so`, `libmigraphx_c.so.3`, `libamdhip64.so.7` are runtime prerequisites on the target (documented, not bundled).

---

## File Structure

- **Create** `src/aji_rocm_color_device.hip` — self-contained device-only source (helpers + `extern "C" __global__` kernels + the `aji_color_csp` struct). Not compiled by CMake; read at build time and embedded as a string. Single source of truth for the GPU math.
- **Create** `cmake/embed_text.cmake` — CMake `-P` script that wraps a text file into a C header (`const char[]` raw-string literal).
- **Create** `src/aji_rocm_color.cpp` — host side: hipRTC JIT loader + on-disk code-object cache + the `extern "C"` launchers (`aji_color_selftest`, `aji_gpu_out_color`) reimplemented over `hipModuleLaunchKernel`. Plain C++ (g++), replaces the compiled `.hip`.
- **Delete** `src/aji_rocm_color.hip` (its device code moves to `_device.hip`, its host launchers move to `.cpp`).
- **Modify** `CMakeLists.txt:168-198` — drop `enable_language(HIP)` + the `CMAKE_HIP_ARCHITECTURES gfx1201` block, add the codegen custom command, swap the TU, link `hiprtc`. Add the `aji_rocm_color_test` target.
- **Create** `test/rocm_color_test.cpp` — characterization harness: runs `aji_gpu_out_color` on a fixed synthetic frame; `--emit` writes a golden, default mode byte-compares against it; also asserts `aji_color_selftest()==0`.
- **Create** `test/fixtures/rocm_color_nv12_128.bin`, `test/fixtures/rocm_color_p010_128.bin` — goldens captured from the current static build (Task 1).
- **Modify** `src/aji_rocm_color.h` — header comment only (no longer "Compiled with hipcc"); ABI unchanged.

Unchanged: `src/aji_rocm.cpp` (call sites, `gpu_color_ensure`, `.mxr` flow), `resample.h`.

---

### Task 1: Characterization harness + golden from the current static build

Establish the regression baseline **before** changing the compile path. The harness is plain C++ calling the frozen color ABI, so it links against the current `.hip`-built object now and the new `.cpp` later without changes.

**Files:**
- Create: `test/rocm_color_test.cpp`
- Create: `test/fixtures/rocm_color_nv12_128.bin`, `test/fixtures/rocm_color_p010_128.bin`
- Modify: `CMakeLists.txt` (add `aji_rocm_color_test` target after `aji_kernel_test`, line ~152)

**Interfaces:**
- Consumes (from the frozen ABI in `src/aji_rocm_color.h`): `typedef struct {...} aji_color_csp;`, `int aji_color_selftest(void);`, `void aji_gpu_out_color(const void* rgb_fp16,int W,int H,aji_color_csp csp,const int* ph_start,const float* ph_wt,int ph_taps,const int* pv_start,const float* pv_wt,int pv_taps,float* Un,float* Vn,float* hu,float* hv,void* yplane,void* uvplane,void* stream);`
- Consumes (from `resample.h`): `aji_resample::make_csp(int fmt,int mat,int rng)`, `aji_resample::chroma_shifts(int siting,bool,double*,double*)`, `aji_resample::compute(int src,int dst,double shift,int filter)` returning `weights{ std::vector<int> start; std::vector<float> wt; int taps; }`.
- Produces: executable `aji_rocm_color_test [--emit]`, exit 0 = pass.

- [ ] **Step 1: Write the harness**

Create `test/rocm_color_test.cpp`:

```cpp
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
    // host fp32->fp16 via the device to avoid a separate half impl
    float* df; unsigned short* dh;
    hipMalloc(&df, f.size() * sizeof(float));
    hipMalloc(&dh, f.size() * sizeof(unsigned short));
    hipMemcpy(df, f.data(), f.size() * sizeof(float), hipMemcpyHostToDevice);
    // reuse: cast on host instead (no kernel) — _Float16 is a host type with -mf16c
    for (size_t i = 0; i < f.size(); i++) { _Float16 v = (_Float16)f[i]; std::memcpy(&h[i], &v, 2); }
    hipFree(df); hipFree(dh);
    return h;
}

static int run_one(int fmt, const char* golden, bool emit) {
    const int mat = 1 /*BT709*/, rng = 0 /*limited*/;
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

int main(int argc, char** argv) {
    bool emit = argc > 1 && std::strcmp(argv[1], "--emit") == 0;
    if (aji_color_selftest() != 0) { fprintf(stderr, "selftest FAILED\n"); return 10; }
    fprintf(stderr, "selftest OK\n");
    int r = 0;
    r |= run_one(AJI_FMT_NV12, "test/fixtures/rocm_color_nv12_128.bin", emit);
    r |= run_one(AJI_FMT_P010, "test/fixtures/rocm_color_p010_128.bin", emit);
    fprintf(stderr, r ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return r;
}
```

> Note: confirm the exact spelling of `AJI_FMT_NV12`, `AJI_FMT_P010`, `AJI_SITING_LEFT`, `AJI_FILTER_SPLINE36`, and the `weights` field names against `resample.h` / `aji.h` while implementing; fix includes if a name differs. These come straight from `gpu_color_ensure` in `src/aji_rocm.cpp:744-778`.

- [ ] **Step 2: Add the CMake test target**

In `CMakeLists.txt`, immediately after the `aji_rocm` block closes (after line 198, inside reach of the same `if(AJI_MIGRAPHX_LIB ...)` so it only builds when ROCm is present), add:

```cmake
    # Characterization test for the GPU out-color kernels (golden regression).
    add_executable(aji_rocm_color_test test/rocm_color_test.cpp src/aji_rocm_color.hip)
    set_source_files_properties(src/aji_rocm_color.hip PROPERTIES LANGUAGE HIP)
    target_include_directories(aji_rocm_color_test PRIVATE include src "${AJI_ROCM_ROOT}/include")
    target_compile_options(aji_rocm_color_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mf16c -mavx2 -mfma>)
    target_compile_definitions(aji_rocm_color_test PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(aji_rocm_color_test PRIVATE ${AJI_HIP_LIB})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aji_rocm_color_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
```

(This target temporarily compiles the existing `.hip` to produce the baseline. Task 3 swaps `src/aji_rocm_color.hip` → `src/aji_rocm_color.cpp` here too.)

- [ ] **Step 3: Configure + build the test against the CURRENT static kernels**

Run:
```bash
cd /home/nathan/Projects/animejanai-inference
cmake --build build --target aji_rocm_color_test 2>&1 | tail -20
```
Expected: builds (HIP compiles `aji_rocm_color.hip` for `gfx1201`), produces `build/aji_rocm_color_test`.

- [ ] **Step 4: Emit the golden fixtures**

Run from the repo root (the harness writes relative paths):
```bash
cd /home/nathan/Projects/animejanai-inference
./build/aji_rocm_color_test --emit
```
Expected: `selftest OK`, two `[emit] test/fixtures/...bin` lines. Files exist:
```bash
ls -l test/fixtures/rocm_color_nv12_128.bin test/fixtures/rocm_color_p010_128.bin
```
Expected sizes: NV12 = 128*128 + 64*64*2 = 24576 bytes; P010 = 49152 bytes.

- [ ] **Step 5: Verify the test passes in compare mode (baseline green)**

Run:
```bash
cd /home/nathan/Projects/animejanai-inference && ./build/aji_rocm_color_test; echo "exit=$?"
```
Expected: `[check] ... maxdiff=0 baddiff=0` for both, `RESULT: PASS`, `exit=0`.

- [ ] **Step 6: Commit**

```bash
git add test/rocm_color_test.cpp test/fixtures/rocm_color_nv12_128.bin test/fixtures/rocm_color_p010_128.bin CMakeLists.txt
git commit -m "test: golden characterization harness for ROCm GPU out-color"
```

---

### Task 2: Embed the kernel source (device-only file + codegen)

Split the device code into a self-contained, includes-free source file and add a CMake step that embeds it as a C string. Mark kernels `extern "C"` so the module API can resolve them by unmangled name.

**Files:**
- Create: `src/aji_rocm_color_device.hip`
- Create: `cmake/embed_text.cmake`
- (Generated at build: `build/gen/aji_rocm_color_device_src.h`)

**Interfaces:**
- Produces: `static const char AJI_ROCM_COLOR_SRC[]` (in the generated header) — the device source string consumed by Task 3's JIT loader.
- Produces kernel symbols (resolved later by name): `k_selftest`, `k_out_y_uvdiff`, `k_out_chroma_h`, `k_out_chroma_v` (all `extern "C"`).

- [ ] **Step 1: Write the device-only source**

Create `src/aji_rocm_color_device.hip`. This is the verbatim device code from `src/aji_rocm_color.hip:13-95` with three changes: (a) no `#include`s (hipRTC supplies device builtins); (b) the `aji_color_csp` struct defined inline so the string is self-contained; (c) each `__global__` wrapped in `extern "C"`. **Host launchers are NOT here.**

```cpp
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
```

- [ ] **Step 2: Write the embed script**

Create `cmake/embed_text.cmake`:

```cmake
# Wrap a text file into a C header as a raw-string char[]. Usage:
#   cmake -DIN=<src> -DOUT=<hdr> -DVAR=<symbol> -P cmake/embed_text.cmake
file(READ "${IN}" _content)
# Guard against the raw-string delimiter appearing in the source.
if(_content MATCHES "\\)AJIHIP\"")
    message(FATAL_ERROR "embed_text: source contains the )AJIHIP\" delimiter")
endif()
file(WRITE "${OUT}" "// Generated from ${IN}. Do not edit.\n")
file(APPEND "${OUT}" "static const char ${VAR}[] = R\"AJIHIP(\n")
file(APPEND "${OUT}" "${_content}")
file(APPEND "${OUT}" ")AJIHIP\";\n")
```

- [ ] **Step 3: Verify the embed script runs standalone**

Run:
```bash
cd /home/nathan/Projects/animejanai-inference
cmake -DIN=src/aji_rocm_color_device.hip -DOUT=/tmp/aji_src.h -DVAR=AJI_ROCM_COLOR_SRC -P cmake/embed_text.cmake
head -3 /tmp/aji_src.h; echo "---"; grep -c "k_out_chroma_v" /tmp/aji_src.h
```
Expected: header starts with the generated comment + `static const char AJI_ROCM_COLOR_SRC[] = R"AJIHIP(`, and the grep prints `1`.

- [ ] **Step 4: Commit**

```bash
git add src/aji_rocm_color_device.hip cmake/embed_text.cmake
git commit -m "feat: extract embeddable device color source + text-embed cmake step"
```

---

### Task 3: hipRTC JIT loader + launchers (compile-every-time), CMake swap

Reimplement the color ABI over hipRTC + the module API, compiling once per process (no disk cache yet — that's Task 4). Swap the build from the `.hip` TU to the new `.cpp`. After this task the characterization test from Task 1 must be green again.

**Files:**
- Create: `src/aji_rocm_color.cpp`
- Modify: `src/aji_rocm_color.h` (comment only)
- Modify: `CMakeLists.txt:168-198` (the `aji_rocm` block) and the `aji_rocm_color_test` target from Task 1
- Delete: `src/aji_rocm_color.hip`

**Interfaces:**
- Consumes: `AJI_ROCM_COLOR_SRC[]` (generated header from Task 2); the frozen ABI in `aji_rocm_color.h`.
- Produces: `aji_color_selftest`, `aji_gpu_out_color` (unchanged signatures), now hipRTC-backed.

- [ ] **Step 1: Write the JIT module + launchers**

Create `src/aji_rocm_color.cpp`:

```cpp
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
```

> Arg-packing care: every entry in `aXX[]` must be the address of an lvalue whose type matches the kernel parameter exactly. `cw`/`ch` are `const int` locals (kernel takes `int`), `rgb_fp16` is `const void*` (kernel takes `const _Float16*` — same pointer width), pointer params pass the address of the pointer variable. This is the classic `hipModuleLaunchKernel` convention; a mismatch shows up as garbage output and the Task 1 test catches it.

- [ ] **Step 2: Update the header comment**

In `src/aji_rocm_color.h`, change the top comment (lines 1-2) from "Compiled from aji_rocm_color.hip with hipcc" to reflect JIT:

```cpp
// Host-callable launchers for the GPU-resident color kernels (Phase B). The device
// kernels are compiled for the present GPU with hipRTC at runtime (see
// src/aji_rocm_color.cpp); these launchers are called from aji_rocm.cpp. All pointers
// are DEVICE pointers unless noted. The chroma resample weights match resample.h's
```

- [ ] **Step 3: Swap the build (CMakeLists.txt)**

Replace the `aji_rocm` block (`CMakeLists.txt:169-195`, from the `# GPU-resident color` comment through `target_link_options`) with:

```cmake
    # GPU-resident color (Phase B): aji_rocm_color_device.hip holds the YUV<->RGB +
    # chroma resample kernels; they are JIT-compiled per-GPU with hipRTC at runtime
    # (src/aji_rocm_color.cpp), so libaji_rocm.so embeds NO device code and runs on any
    # AMD arch. Embed the device source as a string for the JIT loader.
    find_library(AJI_HIPRTC_LIB NAMES hiprtc PATHS "${AJI_ROCM_ROOT}/lib")
    set(AJI_COLOR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/aji_rocm_color_device.hip)
    set(AJI_COLOR_GEN ${CMAKE_CURRENT_BINARY_DIR}/gen/aji_rocm_color_device_src.h)
    add_custom_command(
        OUTPUT ${AJI_COLOR_GEN}
        COMMAND ${CMAKE_COMMAND} -DIN=${AJI_COLOR_SRC} -DOUT=${AJI_COLOR_GEN}
                -DVAR=AJI_ROCM_COLOR_SRC -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
        DEPENDS ${AJI_COLOR_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
        COMMENT "Embedding aji_rocm_color device source")
    add_library(aji_rocm SHARED src/aji_rocm.cpp src/aji_rocm_color.cpp src/aji_conf.cpp ${AJI_COLOR_GEN})
    target_include_directories(aji_rocm
        PUBLIC include
        PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/gen "${AJI_ROCM_ROOT}/include")
    target_compile_options(aji_rocm PRIVATE -mf16c -mavx2 -mfma)
    target_compile_definitions(aji_rocm PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(aji_rocm PRIVATE ${AJI_MIGRAPHX_LIB} ${AJI_HIP_LIB} ${AJI_HIPRTC_LIB})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aji_rocm PRIVATE OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(aji_rocm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        BUILD_RPATH "${AJI_ROCM_ROOT}/lib"
        INSTALL_RPATH "$ORIGIN:${AJI_ROCM_ROOT}/lib")
    if(NOT WIN32)
        target_link_options(aji_rocm PRIVATE -Wl,-Bsymbolic)
    endif()
```

Remove the now-dead `if(NOT DEFINED CMAKE_HIP_ARCHITECTURES) ... set(... gfx1201) ... endif()` and the `enable_language(HIP)` line entirely.

Also update the `aji_rocm_color_test` target (from Task 1, Step 2): replace `src/aji_rocm_color.hip` with `src/aji_rocm_color.cpp ${AJI_COLOR_GEN}`, drop the `set_source_files_properties(... HIP)` line, add `${CMAKE_CURRENT_BINARY_DIR}/gen` to its includes, and add `${AJI_HIPRTC_LIB}` to its link libs:

```cmake
    add_executable(aji_rocm_color_test test/rocm_color_test.cpp src/aji_rocm_color.cpp ${AJI_COLOR_GEN})
    target_include_directories(aji_rocm_color_test PRIVATE include src ${CMAKE_CURRENT_BINARY_DIR}/gen "${AJI_ROCM_ROOT}/include")
    target_compile_options(aji_rocm_color_test PRIVATE -mf16c -mavx2 -mfma)
    target_compile_definitions(aji_rocm_color_test PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(aji_rocm_color_test PRIVATE ${AJI_HIP_LIB} ${AJI_HIPRTC_LIB})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aji_rocm_color_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
```

- [ ] **Step 4: Delete the old TU**

```bash
cd /home/nathan/Projects/animejanai-inference
git rm src/aji_rocm_color.hip
```

- [ ] **Step 5: Reconfigure + build**

```bash
cd /home/nathan/Projects/animejanai-inference
cmake -S . -B build > /tmp/cfg.log 2>&1; tail -5 /tmp/cfg.log
cmake --build build --target aji_rocm aji_rocm_color_test 2>&1 | tail -25
```
Expected: configure succeeds (no `enable_language(HIP)`), both targets build, no `.hip` compiled.

- [ ] **Step 6: Run the characterization test (must reproduce the golden)**

```bash
cd /home/nathan/Projects/animejanai-inference && ./build/aji_rocm_color_test; echo "exit=$?"
```
Expected: `selftest OK`, both `[check] ... maxdiff=0 baddiff=0`, `RESULT: PASS`, `exit=0`.

> If `maxdiff` is small but nonzero (1–2) on a handful of bytes, it is FMA/contraction codegen drift between hipRTC and the former offline hipcc. First try adding `"-ffp-contract=fast"` (then `"-ffp-contract=off"`) to the `opts[]` array in `compile_module()` and rebuild. Only if exact match is unreachable after trying both, accept it by re-emitting the golden against the JIT build AND recording the decision in the task notes — do not silently raise `AJI_COLOR_TOL`.

- [ ] **Step 7: Confirm the .so has no embedded device code**

```bash
roc-obj-ls build/lib/libaji_rocm.so 2>/dev/null || roc-obj-ls build/libaji_rocm.so 2>/dev/null
```
Expected: only a `host-x86_64` line (no `gfx....` code object). (Adjust the path to wherever the build writes `libaji_rocm.so`.)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: JIT the ROCm color kernels with hipRTC (arch-agnostic libaji_rocm.so)"
```

---

### Task 4: On-disk code-object cache (sibling cache dir, self-healing)

Avoid recompiling on every process start: cache the hipRTC code object to `animejanai/cache/aji_color.<arch>.<hash>.co`, mirroring the `.mxr` atomic temp+rename and self-healing-on-load-failure pattern.

**Files:**
- Modify: `src/aji_rocm_color.cpp` (extend `compile_module()` with cache load/save)

**Interfaces:**
- Consumes: `AJI_ROCM_COLOR_SRC[]`, hip/hiprtc, `<dlfcn.h>` (`dladdr`).
- Produces: same `ColorModule`; side effect = a `.co` file in the cache dir.

- [ ] **Step 1: Add cache helpers + load/save to `compile_module()`**

In `src/aji_rocm_color.cpp`, add includes `#include <dlfcn.h>`, `#include <fstream>`, `#include <cstdlib>`, `#include <cstring>`, and within the anonymous namespace add:

```cpp
// FNV-1a over the embedded source + a version tag; invalidates stale .co on a kernel edit.
std::string src_hash() {
    unsigned long long h = 1469598103934665603ULL;
    const char* p = AJI_ROCM_COLOR_SRC;
    const char* tag = "v1";                 // bump on launch-convention changes
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
    std::string mk = "mkdir -p '" + dir + "'"; (void)system(mk.c_str());
    std::string tmp = p + ".tmp";
    { std::ofstream f(tmp, std::ios::binary); if (!f) return; f.write(data.data(), (std::streamsize)data.size()); }
    if (std::rename(tmp.c_str(), p.c_str()) != 0) std::remove(tmp.c_str());
}
```

Then change `compile_module()` so that, after computing `arch`, it tries the cache first and falls through to hipRTC, writing the result back:

```cpp
    std::string arch = prop.gcnArchName;
    const std::string co = cache_path(arch);

    // 1) Try the cached code object. On ANY load failure, delete it and recompile.
    std::vector<char> code;
    if (read_file(co, code)) {
        if (hipModuleLoadData(&m.mod, code.data()) == hipSuccess) {
            // resolve functions (same as below); if that fails, fall through to recompile
            if (hipModuleGetFunction(&m.selftest, m.mod, "k_selftest") == hipSuccess &&
                hipModuleGetFunction(&m.y_uvdiff, m.mod, "k_out_y_uvdiff") == hipSuccess &&
                hipModuleGetFunction(&m.chroma_h, m.mod, "k_out_chroma_h") == hipSuccess &&
                hipModuleGetFunction(&m.chroma_v, m.mod, "k_out_chroma_v") == hipSuccess) {
                m.ok = true; return m;
            }
            hipModuleUnload(m.mod); m.mod = nullptr;
        }
        std::remove(co.c_str());   // stale/incompatible: heal by recompiling
        code.clear();
    }

    // 2) Compile with hipRTC (existing code), then cache the bytes:
    //    ... hiprtcCreateProgram / CompileProgram / GetCode into `code` ...
    //    (after hiprtcGetCode(prog, code.data()); hiprtcDestroyProgram(&prog);)
    write_file_atomic(co, code);
    //    ... hipModuleLoadData + hipModuleGetFunction as before ...
```

> Implementation note: refactor `compile_module()` so the `hipModuleLoadData` + 4×`hipModuleGetFunction` resolution is a small local lambda `resolve(code)` reused by both the cache path and the compile path, to keep it DRY. The hipRTC compile block from Task 3 stays; only its result is now also written to `co` before loading.

- [ ] **Step 2: Rebuild**

```bash
cd /home/nathan/Projects/animejanai-inference
cmake --build build --target aji_rocm aji_rocm_color_test 2>&1 | tail -15
```
Expected: builds clean.

- [ ] **Step 3: First run compiles + writes the cache**

```bash
cd /home/nathan/Projects/animejanai-inference
rm -f /tmp/ajicache/* 2>/dev/null
AJI_ROCM_CACHE_DIR=/tmp/ajicache ./build/aji_rocm_color_test; echo "exit=$?"
ls -l /tmp/ajicache/
```
Expected: `RESULT: PASS`, `exit=0`, and a file `aji_color.gfx1201.<hash>.co` exists in `/tmp/ajicache/`.

- [ ] **Step 4: Second run loads from cache (still correct)**

```bash
cd /home/nathan/Projects/animejanai-inference
AJI_ROCM_CACHE_DIR=/tmp/ajicache ./build/aji_rocm_color_test; echo "exit=$?"
```
Expected: `RESULT: PASS`, `exit=0` (byte-identical, now via the cached object).

- [ ] **Step 5: Self-heal on a corrupt cache**

```bash
cd /home/nathan/Projects/animejanai-inference
echo "garbage" > /tmp/ajicache/aji_color.gfx1201.*.co 2>/dev/null || true
AJI_ROCM_CACHE_DIR=/tmp/ajicache ./build/aji_rocm_color_test; echo "exit=$?"
```
Expected: still `RESULT: PASS`, `exit=0` (corrupt `.co` was deleted and recompiled).

- [ ] **Step 6: Commit**

```bash
git add src/aji_rocm_color.cpp
git commit -m "feat: cache JIT color code objects to animejanai/cache (self-healing)"
```

---

### Task 5: Docs, end-to-end verification on gfx1201, and packaging note

Lock in the user-facing facts and verify the real engine path (not just the unit test) still produces identical output and writes the cache where the spec says.

**Files:**
- Modify: `BUILD.md` (ROCm build no longer needs a HIP arch; note hipRTC runtime dep)
- Modify: `docs/superpowers/specs/2026-06-19-rocm-color-kernel-jit-portability-design.md` (mark Status: implemented)
- Verify only: the shipped/installed engine path

- [ ] **Step 1: Real-engine smoke test on this box**

Run an actual upscale through the ROCm backend on the dev box (gfx1201) using the existing benchmark/CLI harness the project already uses for ROCm (e.g. the `aji_harness`/benchmark with `backend=ROCm`, slots 1010/1011, `--load-scripts=no` per the project's benchmark convention). Confirm: it runs, output looks correct, and a `cache/aji_color.gfx1201.<hash>.co` appears next to the install's `animejanai/` (sibling of `inference/`).

```bash
# adjust to the project's standard ROCm benchmark invocation
ls -l /home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/animejanai/cache/ 2>/dev/null
```
Expected: the `.co` file is present after a play/benchmark; playback unchanged.

- [ ] **Step 2: Update BUILD.md**

In `BUILD.md`, in the ROCm section, replace any "builds for gfx1201 / set CMAKE_HIP_ARCHITECTURES" guidance with: the ROCm backend no longer bakes a GPU arch — color kernels JIT-compile per device with hipRTC at runtime; the build host only needs the ROCm SDK, and target machines need a compatible ROCm runtime (`libhiprtc.so`, `libmigraphx_c.so.3`, `libamdhip64.so.7`). Code objects are cached in `animejanai/cache/` (override with `AJI_ROCM_CACHE_DIR`).

- [ ] **Step 3: Packaging note (no code change)**

Confirm the packaging for the ROCm tarball does **not** ship a prebuilt `cache/` dir (it is per-machine, created at runtime) and does not ship `.co` files. The current packaging already ships no `.mxr`, so this is a verification, not an edit. If a `.gitignore`/package-exclude for `*.co` and `cache/` is warranted, add it.

- [ ] **Step 4: Mark the spec implemented + commit**

```bash
cd /home/nathan/Projects/animejanai-inference
git add BUILD.md docs/superpowers/specs/2026-06-19-rocm-color-kernel-jit-portability-design.md
git commit -m "docs: ROCm color kernels JIT per-GPU with hipRTC; mark design implemented"
```

---

## Acceptance (from the spec)

1. `roc-obj-ls build/.../libaji_rocm.so` shows host code only, **no** `gfx` code object. — Task 3 Step 7.
2. On gfx1201: first run JIT-compiles into `animejanai/cache/aji_color.gfx1201.<hash>.co`; later runs load from cache; color output bit-identical to the pre-change build. — Task 1 golden + Task 3 Step 6 + Task 4 Steps 3-4 + Task 5 Step 1.
3. The same tarball runs with GPU color on a second AMD arch, compiling its own `.co` on first run — no rebuild, no CPU fallback. — Structural: no baked arch (Task 3 Step 7), per-device `--offload-arch` from `gcnArchName` (Task 3 Step 1), self-healing cache (Task 4 Step 5). **Final cross-GPU confirmation requires a second AMD machine and is the user's acceptance test.**

## Self-Review notes

- **Spec coverage:** embedded source (T2) ✓, hipRTC per-device compile (T3) ✓, module-API launch (T3) ✓, cache dir `animejanai/cache` + env override (T4) ✓, self-healing load (T4) ✓, atomic temp+rename (T4) ✓, parity guard / bit-exact (T1 golden, asserted T3) ✓, build simplification / no HIP lang / no arch list (T3) ✓, prerequisite docs (T5) ✓, CPU path remains only as safety net (untouched in aji_rocm.cpp) ✓.
- **Type consistency:** `aji_color_csp` defined identically in `aji_rocm_color.h` and the embedded `aji_rocm_color_device.hip`; kernel names (`k_selftest`, `k_out_y_uvdiff`, `k_out_chroma_h`, `k_out_chroma_v`) consistent across device source, `hipModuleGetFunction`, and launchers; `AJI_ROCM_COLOR_SRC` symbol name consistent between `embed_text.cmake` (`-DVAR=`) and `aji_rocm_color.cpp`.
- **Known risk:** hipRTC-vs-hipcc FMA codegen drift could make the Task 3 golden differ by ≤1 quant step; mitigation (fp-contract flag, then a documented re-baseline) is in Task 3 Step 6.
- **Out of scope (unchanged):** `.mxr` cache keying, ROCm bundling, Vulkan/NVIDIA.
```

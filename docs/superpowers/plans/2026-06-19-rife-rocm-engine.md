# RIFE-on-ROCm Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Linux/AMD ROCm/MIGraphX engine backend (`aji_rocm`) run RIFE frame interpolation with parity to the Windows TensorRT/DirectML backends — 4:2:0 first (Phase A), then full 4:4:4 + GPU-resident color (Phase B).

**Architecture:** Port the proven RIFE pipeline (11-channel fp16 input tensor: 2 frames RGB + timestep + 4 const mesh/multiplier planes → 3-channel RGB out) from `aji_dml.cpp`/`aji_trt.cpp` onto the existing `compile_mxr`/`load_model`/`.mxr`-cache MIGraphX path. Pure-CPU RIFE math (name resolution, padding, scene-detect, BT.709+bilinear color) lives in a new dependency-free `rife_cpu.h` so it is unit-testable without a GPU; the GPU-bound pieces (MIGraphX eval, Phase-B HIP kernels) are verified by build + a migraphx compile smoke test + in-player play. `aji_infer_rife` is synchronous-by-contract.

**Tech Stack:** C++17, ROCm 7.2.4 / MIGraphX 2.15 (`migraphx::parse_onnx`/`quantize_fp16`/`compile`), HIP (`hipcc`, `aji_rocm_color.hip`, `gfx1201`), `resample.h` (CPU color), CMake. Reference backends: `aji_dml.cpp`, `aji_trt.cpp`, `kernels.cu`.

**Design spec:** `docs/superpowers/specs/2026-06-19-rife-rocm-engine-design.md` (read it first — every parity detail and verified file:line anchor is there).

## Global Constraints

- **Color is hardcoded BT.709** for the RIFE YUV↔RGB round-trip regardless of source matrix: `make_csp(fmt, AJI_MATRIX_BT709, range)` (range follows source). Never use the source matrix.
- **11-channel NCHW input** `{1,11,ph,pw}`, `plane = pw*ph`: ch0-2 = frame A planar R,G,B; ch3-5 = frame B R,G,B; ch6 = timestep; ch7 = meshX; ch8 = meshY; ch9 = mulW; ch10 = mulH. Output `{1,3,ph,pw}`. RGB order (`dst[idx]=R, dst[plane+idx]=G, dst[2*plane+idx]=B`).
- **Padding** mod-64: `pw=(w+63)/64*64`, `ph=(h+63)/64*64`; centered even offsets `pad_l=((pw-w)/2)&~1`, `pad_t=((ph-h)/2)&~1`. Chroma plane offset uses `pad_t/2` for 4:2:0, `pad_t` for 444.
- **Const channels 7-10 filled ONCE** at setup; only ch0-6 change per frame.
- **Scene-detect:** `d=|Ya−Yb|·norm` over UNPADDED luma (raw container values, no shift), `sum/(pw·ph) > scd_threshold → AJI_SCENE`. `norm = 1/255` (NV12), `1/65472` (P010), `1/65535` (YUV444P16). Default threshold 0.150.
- **Timestep ch6 cast = `(_Float16)t`** (round-to-nearest-even), matching TRT.
- **Fixed-shape compile only** (MIGraphX cannot do dynamic shapes here); compile per resolution, cache to `.mxr`.
- **`aji_infer_rife` is synchronous on return.** The `stream` arg is NULL on the sw path and is ignored.
- **`aji_rife_factor` returns 0 until the RIFE `.mxr` is actually loaded** (`enabled && loaded`), independent of `c->active`.
- **ABI export invariant:** `aji_rocm.so` must keep exporting `aji_scale_factor`, `aji_rife_factor`, `aji_rife_before_upscale`, `aji_infer_rife` (the dispatcher hard-fails the whole backend if any is missing).
- **No `AJI_API_VERSION` bump** (currently 7) — widening 444 to ROCm is additive.
- **RIFE buffers/scratch are dedicated** to `RifeState` — never reuse the worker's `mdl_in`/`mdl_out` or `c->scratch` (data race with the upscale worker thread).
- **`_heavy` RIFE variant is out of scope** (not in the reference, not expressible by the int code).
- Build with the repo's normal command (assumed `cmake --build build -j`; adjust to the actual build dir). `kernels.cu` is CUDA-only and NOT linked into `aji_rocm`; new GPU kernels go in `aji_rocm_color.hip`.

---

# Phase A — RIFE working at 4:2:0 (must-have)

Engine-only, no filter change, no new HIP kernels.

## Task A1: `rife_cpu.h` skeleton + `rife_selftest` harness

**Files:**
- Create: `src/rife_cpu.h`
- Create: `src/rife_selftest.cpp`
- Modify: `CMakeLists.txt` (add the `rife_selftest` target)

**Interfaces:**
- Produces: `namespace rife_cpu` with (filled in later tasks) `std::string model_name(int code, bool ensemble)`, `struct Geom { int w,h,pw,ph,pad_l,pad_t; }; Geom geometry(int w,int h)`, scene-detect and color functions. This task only establishes the header + a runnable test main.

- [ ] **Step 1: Create the header skeleton**

```cpp
// src/rife_cpu.h — pure-CPU RIFE math, no HIP/MIGraphX deps (unit-testable).
#pragma once
#include <string>
#include <cstdint>
#include <cstddef>

namespace rife_cpu {
// (functions added in A2, A3, A6, A7)
}  // namespace rife_cpu
```

- [ ] **Step 2: Create the self-test harness with a trivial passing assertion**

```cpp
// src/rife_selftest.cpp — standalone CPU unit tests (no GPU). Build: g++ -std=c++17.
#include "rife_cpu.h"
#include <cstdio>
#include <cstdlib>
static int g_fail = 0;
#define CHECK(cond) do { if(!(cond)){ printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while(0)
int main() {
    CHECK(1 + 1 == 2);
    printf(g_fail ? "rife_selftest: %d FAILURES\n" : "rife_selftest: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
```

- [ ] **Step 3: Add the CMake target**

In `CMakeLists.txt`, near the other test/host targets (e.g. the `kernel_test` target), add:

```cmake
add_executable(rife_selftest src/rife_selftest.cpp)
target_include_directories(rife_selftest PRIVATE src)
```

- [ ] **Step 4: Build and run — verify the harness works**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: `rife_selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add src/rife_cpu.h src/rife_selftest.cpp CMakeLists.txt
git commit -m "test: add rife_cpu.h skeleton + rife_selftest harness"
```

## Task A2: `rife_cpu::model_name`

**Files:**
- Modify: `src/rife_cpu.h`
- Modify: `src/rife_selftest.cpp`

**Interfaces:**
- Produces: `std::string rife_cpu::model_name(int code, bool ensemble)` — returns the onnx basename WITHOUT extension/dir (e.g. `"rife_v4.14"`), or `""` for an invalid code (`<2` digits). Mirrors `aji_trt.cpp:707-718` / `aji_dml.cpp:1253-1265`.

- [ ] **Step 1: Write the failing tests**

In `rife_selftest.cpp` `main()`, before the summary:

```cpp
CHECK(rife_cpu::model_name(414, false)  == "rife_v4.14");
CHECK(rife_cpu::model_name(47,  false)  == "rife_v4.7");
CHECK(rife_cpu::model_name(4251, false) == "rife_v4.25_lite");      // 4-digit ending in 1
CHECK(rife_cpu::model_name(414, true)   == "rife_v4.14_ensemble");
CHECK(rife_cpu::model_name(4251, true)  == "rife_v4.25_lite_ensemble");
CHECK(rife_cpu::model_name(4,   false)  == "");                     // <2 digits → invalid
```

- [ ] **Step 2: Run — verify it fails**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: FAIL (compile error: `model_name` not declared).

- [ ] **Step 3: Implement in `rife_cpu.h`**

```cpp
inline std::string model_name(int code, bool ensemble) {
    std::string s = std::to_string(code);
    if (s.size() < 2) return "";                                   // invalid
    std::string dec = (s.size() == 2) ? s.substr(1, 1) : s.substr(1, 2);
    std::string name = "rife_v" + s.substr(0, 1) + "." + dec;      // substr, NOT s[0] (UB)
    if (s.size() == 4 && s.back() == '1') name += "_lite";
    if (ensemble)                         name += "_ensemble";
    return name;
}
```

- [ ] **Step 4: Run — verify it passes**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: `rife_selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add src/rife_cpu.h src/rife_selftest.cpp
git commit -m "feat(rife): model_name code->onnx-basename resolution"
```

## Task A3: `rife_cpu::geometry` (mod-64 padding)

**Files:**
- Modify: `src/rife_cpu.h`
- Modify: `src/rife_selftest.cpp`

**Interfaces:**
- Produces: `struct Geom { int w,h,pw,ph,pad_l,pad_t; }; Geom rife_cpu::geometry(int w,int h)`. Mirrors `aji_dml.cpp:1288-1293`.

- [ ] **Step 1: Write the failing tests**

```cpp
{ auto g = rife_cpu::geometry(1920, 1080);
  CHECK(g.pw == 1920);                       // 1920 already mod-64
  CHECK(g.ph == 1088);                       // 1080 -> 1088
  CHECK(g.pad_l == 0);
  CHECK(g.pad_t == ((1088-1080)/2) & ~1);    // = 4
}
{ auto g = rife_cpu::geometry(854, 480);
  CHECK(g.pw == 896);                        // (854+63)/64*64
  CHECK(g.ph == 512);                        // (480+63)/64*64
  CHECK(g.pad_l == (((896-854)/2) & ~1));    // 21 -> 20
  CHECK(g.pad_t == (((512-480)/2) & ~1));    // 16
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: FAIL (`geometry` not declared).

- [ ] **Step 3: Implement in `rife_cpu.h`**

```cpp
struct Geom { int w, h, pw, ph, pad_l, pad_t; };
inline Geom geometry(int w, int h) {
    Geom g; g.w = w; g.h = h;
    g.pw = (w + 63) / 64 * 64;
    g.ph = (h + 63) / 64 * 64;
    g.pad_l = ((g.pw - w) / 2) & ~1;           // centered, rounded down to even
    g.pad_t = ((g.ph - h) / 2) & ~1;
    return g;
}
```

- [ ] **Step 4: Run — verify it passes**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: `rife_selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add src/rife_cpu.h src/rife_selftest.cpp
git commit -m "feat(rife): mod-64 padding geometry"
```

## Task A4: `rife_cpu::scene_detect`

**Files:**
- Modify: `src/rife_cpu.h`
- Modify: `src/rife_selftest.cpp`

**Interfaces:**
- Produces: `bool rife_cpu::scene_detect(const uint8_t* ya, ptrdiff_t sa, const uint8_t* yb, ptrdiff_t sb, int w, int h, int pw, int ph, double norm, double threshold)` — returns true (= scene change, skip interp) when `Σ|Ya−Yb|·norm / (pw·ph) > threshold`. Luma read as raw container values; `bpp` is encoded by `norm` + the caller choosing the right read width via two overloads (u8 / u16). Mirrors `aji_dml.cpp:2222,2255`.

- [ ] **Step 1: Write the failing tests**

```cpp
{   // identical frames -> no scene change
    const int w=4,h=2; uint8_t a[8]={10,20,30,40,50,60,70,80}, b[8]; for(int i=0;i<8;i++) b[i]=a[i];
    CHECK(rife_cpu::scene_detect(a, w, b, w, w, h, 64, 64, 1.0/255.0, 0.150) == false);
}
{   // hard cut: max difference everywhere -> scene change
    const int w=4,h=2; uint8_t a[8]; uint8_t b[8]; for(int i=0;i<8;i++){a[i]=0;b[i]=255;}
    // sum = 8*255*(1/255) = 8 ; /(64*64=4096) = 0.00195 < 0.15 -> NOT a scene at 64x64 padded
    CHECK(rife_cpu::scene_detect(a, w, b, w, w, h, 64, 64, 1.0/255.0, 0.150) == false);
    // but with padded area == unpadded (pw=4,ph=2 -> /8) -> 8/8=1.0 > 0.15 -> scene
    CHECK(rife_cpu::scene_detect(a, w, b, w, w, h, 4, 2, 1.0/255.0, 0.150) == true);
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: FAIL (`scene_detect` not declared).

- [ ] **Step 3: Implement in `rife_cpu.h`** (u8 + u16 overloads)

```cpp
inline bool scene_detect(const uint8_t* ya, ptrdiff_t sa, const uint8_t* yb, ptrdiff_t sb,
                         int w, int h, int pw, int ph, double norm, double threshold) {
    double sum = 0.0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* ra = ya + (ptrdiff_t)y * sa;
        const uint8_t* rb = yb + (ptrdiff_t)y * sb;
        for (int x = 0; x < w; ++x) sum += std::abs((int)ra[x] - (int)rb[x]) * norm;
    }
    return sum / ((double)pw * ph) > threshold;
}
inline bool scene_detect(const uint16_t* ya, ptrdiff_t sa, const uint16_t* yb, ptrdiff_t sb,
                         int w, int h, int pw, int ph, double norm, double threshold) {
    double sum = 0.0;
    for (int y = 0; y < h; ++y) {
        const uint16_t* ra = (const uint16_t*)((const uint8_t*)ya + (ptrdiff_t)y * sa);
        const uint16_t* rb = (const uint16_t*)((const uint8_t*)yb + (ptrdiff_t)y * sb);
        for (int x = 0; x < w; ++x) sum += std::abs((int)ra[x] - (int)rb[x]) * norm;
    }
    return sum / ((double)pw * ph) > threshold;
}
```
(Add `#include <cmath>` to `rife_cpu.h`. The u8 stride args are byte strides for u8; the u16 overload treats strides as byte strides too — pass `w*2` for tight u16.)

- [ ] **Step 4: Run — verify it passes**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: `rife_selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add src/rife_cpu.h src/rife_selftest.cpp
git commit -m "feat(rife): CPU scene-detect (padded-area divisor, raw luma)"
```

## Task A5: `rife_cpu` const-channel fill + BT.709 bilinear color (pre/post)

**Files:**
- Modify: `src/rife_cpu.h` (or split into `src/rife_cpu.cpp` if it needs `resample.h`)
- Modify: `src/rife_selftest.cpp`

**Interfaces:**
- Produces:
  - `void rife_cpu::fill_consts(float* tensor, int pw, int ph)` — writes ch7-10 (mesh/mul) into the fp32 staging tensor at element offset `7*pw*ph`. Formulas per Global Constraints. (Cast to `_Float16` happens at H2D staging in A8.)
  - `void rife_cpu::yuv420_to_rgb_planes(const aji_frame& f, const Geom& g, float* tensor, int frame_index, aji_range range)` — BT.709 + bilinear chroma upsample of an NV12/P010 frame into tensor channels `3*frame_index .. 3*frame_index+2`, centered into the padded `pw×ph`, borders = BT.709(studio black). Built on `resample.h` `compute(..., AJI_FILTER_BILINEAR)` + `make_csp(fmt, AJI_MATRIX_BT709, range)` (mirror `aji_dml.cpp:2132-2135` plan creation, NOT `gpu_pre` which bakes Spline36).
  - `void rife_cpu::rgb_planes_to_yuv420(const float* tensor3, const Geom& g, aji_frame& out, aji_range range)` — inverse: crop the centered `w×h` window, BT.709 RGB→YUV + bilinear chroma downsample into the NV12/P010 `out`.

> Note: this task pulls `resample.h` and `aji.h` (for `aji_frame`/`aji_format`/`aji_range`) into `rife_cpu`. Keep it free of HIP/MIGraphX. If `resample.h` is heavy, implement the color in `src/rife_cpu.cpp` and keep declarations in the header; add `rife_cpu.cpp` to both the `rife_selftest` and `aji_rocm` targets in CMake.

- [ ] **Step 1: Write the failing tests**

```cpp
{   // fill_consts: corners of a small grid
    const int pw=4, ph=2; std::vector<float> t((size_t)11*pw*ph, -123.f);
    rife_cpu::fill_consts(t.data(), pw, ph);
    const int plane = pw*ph; auto at=[&](int ch,int x,int y){ return t[(size_t)ch*plane + y*pw + x]; };
    CHECK(at(7,0,0) == -1.0f);                          // meshX at x=0
    CHECK(at(7,pw-1,0) == 1.0f);                        // meshX at x=pw-1
    CHECK(at(8,0,ph-1) == 1.0f);                        // meshY at y=ph-1
    CHECK(std::abs(at(9,2,1) - (2.0f/(pw-1))) < 1e-6);  // mulW constant
    CHECK(std::abs(at(10,1,0) - (2.0f/(ph-1))) < 1e-6); // mulH constant
}
{   // color round-trip parity: NV12 gray frame -> RGB -> NV12 within tolerance
    // (construct a small NV12 aji_frame of mid-gray, run yuv420_to_rgb_planes then
    //  rgb_planes_to_yuv420, assert |out - in| <= 2 on Y and chroma)
    // ... see spec §7.3; build a 64x64 mid-gray NV12, geometry(64,64), tolerance 2.
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: FAIL (`fill_consts`/color not declared).

- [ ] **Step 3: Implement `fill_consts` (header) and the color (rife_cpu.cpp)**

```cpp
inline void fill_consts(float* t, int pw, int ph) {
    const size_t plane = (size_t)pw * ph;
    float* mx = t + 7*plane; float* my = t + 8*plane; float* mw = t + 9*plane; float* mh = t + 10*plane;
    const float kw = 2.0f/(pw-1), kh = 2.0f/(ph-1);
    for (int y = 0; y < ph; ++y) for (int x = 0; x < pw; ++x) {
        size_t i = (size_t)y*pw + x;
        mx[i] = 2.0f*x/(pw-1) - 1.0f;
        my[i] = 2.0f*y/(ph-1) - 1.0f;
        mw[i] = kw;
        mh[i] = kh;
    }
}
```
For the color: create bilinear pre/post plans once (cache on the `RifeState`, passed in) and run them; reuse the exact matrix/quant math `aji_dml.cpp` uses for RIFE (`record_pre`/`record_post`, `aji_dml.cpp:690-779`) but on CPU via `resample.h`. Pre-fill the destination RGB planes with BT.709(studio black) before writing the centered window.

- [ ] **Step 4: Run — verify it passes**

Run: `cmake --build build -j --target rife_selftest && ./build/rife_selftest`
Expected: `rife_selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add src/rife_cpu.h src/rife_cpu.cpp src/rife_selftest.cpp CMakeLists.txt
git commit -m "feat(rife): const-channel fill + BT.709/bilinear CPU color"
```

## Task A6: Plumb `rife_model_dir` + `RifeState` + `gpu_eval_mtx`

**Files:**
- Modify: `src/aji_rocm.cpp` (`aji_ctx` struct `:150`, `aji_create` `:903-933`, `run_chain` eval site `:712-713`, `run_chain_gpu` eval sites `:805-806,:837-838`)

**Interfaces:**
- Produces: `aji_ctx::rife_model_dir` (std::string), `aji_ctx::rife` (a `RifeState`), `aji_ctx::gpu_eval_mtx` (std::mutex). `RifeState` holds: the RIFE `MgxModel` (or its handle), `Geom`, `int num,den; double scd_threshold; bool before_upscale, enabled, loaded;`, device buffers `void* dev_in,*dev_out`, pinned host `void* pin_in,*pin_out`, dedicated color scratch, and a `std::vector<float>` const-template.

- [ ] **Step 1: Add the struct members and mutex**

In `aji_ctx` (`:150`):

```cpp
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
};
RifeState rife;
std::string rife_model_dir;
std::mutex gpu_eval_mtx;
```
Add `#include "rife_cpu.h"` to `aji_rocm.cpp`.

- [ ] **Step 2: Read `rife_model_dir` in `aji_create`**

In `aji_create` (`:903-933`), alongside the other `params->` copies (mirror `aji_dml.cpp:1425`):

```cpp
if (params->rife_model_dir) c->rife_model_dir = params->rife_model_dir;
```

- [ ] **Step 3: Wrap every eval+sync site in the mutex**

In `run_chain` around `:712-713`, `run_chain_gpu` around `:805-806` and `:837-838`:

```cpp
{
    std::lock_guard<std::mutex> lk(c->gpu_eval_mtx);
    outs = prog.eval(pp);
    hipDeviceSynchronize();
}
```
(Apply to each existing eval+`hipDeviceSynchronize` pair; the device-wide sync is why the lock must span both.)

- [ ] **Step 4: Build — verify it compiles and upscale still works**

Run: `cmake --build build -j` then play a normal (non-RIFE) upscale chain in mpv for ~10 s.
Expected: builds clean; upscale output unchanged, no deadlock/stutter.

- [ ] **Step 5: Commit**

```bash
git add src/aji_rocm.cpp
git commit -m "feat(rife): plumb rife_model_dir, RifeState, GPU eval mutex"
```

## Task A7: `compile_mxr` channel param + RIFE-aware `load_model` output selection

**Files:**
- Modify: `src/aji_rocm.cpp` (`compile_mxr` `:266-294`, `mxr_cache_path` `:253-255`, `load_model` `:317-370`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `compile_mxr(..., int in_channels = 3)` (RIFE passes 11). A RIFE-aware loader that selects the output param by name (`main:#output_0`) or `channel==3` and asserts `{1,3,ph,pw}`; the input param `input` must be `{1,11,ph,pw}` or the load fails loudly.

- [ ] **Step 1: Parameterize `compile_mxr` channel count + cache key**

`:277` becomes `oo.set_input_parameter_shape("input", {1, (size_t)in_channels, (size_t)in_h, (size_t)in_w});` with `in_channels` a new arg (default 3). In `mxr_cache_path` (`:253-255`) append the channel count to the cache filename defensively (e.g. `…<W>x<H>.c<in_channels>.dev.mlir.fp16.mxr`).

- [ ] **Step 2: Add a RIFE-aware output selection in `load_model`**

After the param loop, when loading the RIFE model, pick the output explicitly rather than "last 4D non-input":

```cpp
// RIFE: output is the 3-channel param; assert exact shape.
for (auto& p : m->params) {
    if (p.name == m->in_name) continue;
    auto& d = p.dims;
    if (d.size() == 4 && d[1] == 3) { m->out_w = d[3]; m->out_h = d[2]; /* out param */ }
}
// assert input {1,11,ph,pw}: locate p.name==in_name, dims=={1,11,ph,pw}
```
Disable RIFE loudly (set `c->err`, leave `rife.loaded=false`) on any mismatch.

- [ ] **Step 3: Smoke test — compile the RIFE onnx at a fixed shape (runnable on the box)**

Run:
```bash
migraphx-driver compile \
  /home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/animejanai/rife/rife_v4.14.onnx \
  --onnx --input-dim "@input" 1 11 256 256 --fp16 --gpu
```
Expected: `Complete: …` exit 0, input `input {1,11,256,256}`, output `{1,3,256,256}`, no unsupported ops. (Confirms the channel param + shape are right before wiring into the engine.)

- [ ] **Step 4: Build — verify it compiles**

Run: `cmake --build build -j`
Expected: clean build; existing upscale `compile_mxr(...)` calls still pass `in_channels` defaulted to 3.

- [ ] **Step 5: Commit**

```bash
git add src/aji_rocm.cpp
git commit -m "feat(rife): compile_mxr 11-ch param + RIFE-aware output selection"
```

## Task A8: `setup_rife()` + `build_plan` hook + async-build cascade + OSD marker

**Files:**
- Modify: `src/aji_rocm.cpp` (`build_plan` `:584-674`, esp. the `:659` drop; `ensure_model`/`start_async_build` cascade `:300,:560-574`; `aji_poll` `:1118-1126`)

**Interfaces:**
- Consumes: `rife_cpu::model_name/geometry/fill_consts`, `compile_mxr(...,11)`, the RIFE loader (A7), `RifeState`.
- Produces: `bool setup_rife(aji_ctx* c, const ChainConf* chain, int rw, int rh, double fps)` — populates `c->rife`, enrolls the RIFE `.mxr` build in the single-build cascade, writes an OSD build marker, sets `enabled=true` and (once the `.mxr` is cached+loaded) `loaded=true`.

- [ ] **Step 1: Implement `setup_rife`**

Replace the `:659` log-only drop with a call to `setup_rife`. `setup_rife`:
1. `rw,rh = chain->rife_before_upscale ? (source w,h) : (cw,ch)`; `c->rife.g = rife_cpu::geometry(rw,rh)`.
2. `name = rife_cpu::model_name(chain->rife_model, chain->rife_ensemble)`; if `name.empty() || c->rife_model_dir.empty()` → log + `return false` (RIFE disabled, chain continues).
3. `onnx = c->rife_model_dir + "/" + name + ".onnx"`. Enroll its compile in the SAME one-build-at-a-time path as upscale models (`ensure_model`-style); while uncached, write `c->log` marker `"Building MIGraphX engine for " + name + "…"` (mirror `:572`) and leave `loaded=false` (passthrough, no interp).
4. Once the `.mxr` is present: load via the RIFE loader (A7); verify shapes; `fill_consts` into `c->rife.assembly`; allocate `dev_in/dev_out`, `pin_in/pin_out`; H2D the const-bearing 11-ch buffer once; set `before_upscale`, `num/den` from the chain, `scd_threshold`, `enabled=true`, `loaded=true`.
5. Append the stats string to the steps log.

> Place the `setup_rife` call BEFORE the `c->active` computation (`:666`) so a RIFE-only chain (zero upscale models, `active=false`) still configures RIFE.

- [ ] **Step 2: Make `aji_poll` surface the RIFE build**

Ensure the RIFE compile is reported by `aji_poll` (`:1118-1126`) like upscale builds, so the player OSD/`engine_monitor` shows progress and auto-resumes when cached.

- [ ] **Step 3: Build**

Run: `cmake --build build -j`
Expected: clean build.

- [ ] **Step 4: Smoke test — first play of a RIFE chain shows the build then loads**

Configure a conf chain with `chain_1_rife=yes`, `rife_model=414`, an upscale model, on a clip. Delete any cached `rife_v4.14.*.mxr`. Play.
Expected: OSD shows "Building MIGraphX engine for rife_v4.14…", passthrough (no interp) for ~72 s, then the engine loads and `aji_rife_factor` starts returning 1 (verified in the next task by interpolation actually happening).

- [ ] **Step 5: Commit**

```bash
git add src/aji_rocm.cpp
git commit -m "feat(rife): setup_rife + build_plan hook + async-build cascade + OSD marker"
```

## Task A9: ABI functions + per-frame `aji_infer_rife` (4:2:0)

**Files:**
- Modify: `src/aji_rocm.cpp` (`aji_rife_factor` `:1112`, `aji_rife_before_upscale` `:1113`, `aji_infer_rife` `:1128-1132`)

**Interfaces:**
- Consumes: `RifeState`, `rife_cpu::scene_detect/yuv420_to_rgb_planes/rgb_planes_to_yuv420`, `gpu_eval_mtx`, the RIFE `MgxModel`.
- Produces: the three real ABI bodies (signatures unchanged — see `include/aji.h:184-216`).

- [ ] **Step 1: Implement the accessors**

```cpp
extern "C" int aji_rife_factor(aji_ctx* c, int* num, int* den) {
    if (!c || !c->rife.enabled || !c->rife.loaded) return 0;     // 0 until the .mxr is loaded
    if (num) *num = c->rife.num; if (den) *den = c->rife.den;
    return 1;
}
extern "C" int aji_rife_before_upscale(aji_ctx* c) {
    return (c && c->rife.enabled && c->rife.before_upscale) ? 1 : 0;
}
```

- [ ] **Step 2: Implement `aji_infer_rife` (4:2:0, synchronous)**

```cpp
extern "C" int aji_infer_rife(aji_ctx* c, const aji_frame* a, const aji_frame* b,
                              double t, const aji_frame* out, void* /*stream*/) {
    if (!c || !c->rife.loaded) { if(c) c->err = "rife not loaded"; return AJI_ERR; }
    auto& R = c->rife;
    if (a->format != b->format || a->format != out->format) { c->err="rife fmt mismatch"; return AJI_ERR_FORMAT; }
    if (a->format != AJI_FMT_NV12 && a->format != AJI_FMT_P010) { c->err="rife needs nv12/p010"; return AJI_ERR_FORMAT; }
    // 1. scene detect on unpadded luma
    const double norm = (a->format==AJI_FMT_P010) ? 1.0/65472.0 : 1.0/255.0;
    bool scene = /* call the u8 or u16 scene_detect overload on a->plane[0]/b->plane[0] */;
    if (scene) return AJI_SCENE;                                  // caller duplicates A
    // 2. assemble 11-ch fp32: consts already in R.assembly; write A->ch0-2, B->ch3-5, t->ch6
    rife_cpu::yuv420_to_rgb_planes(*a, R.g, R.assembly.data(), 0, a->range);
    rife_cpu::yuv420_to_rgb_planes(*b, R.g, R.assembly.data(), 1, a->range);
    float* ch6 = R.assembly.data() + 6*(size_t)R.g.pw*R.g.ph;
    for (size_t i=0;i<(size_t)R.g.pw*R.g.ph;++i) ch6[i] = (float)t;
    // 3. fp32 -> fp16 pinned, H2D full 11-ch into dev_in
    //    (cast assembly -> R.pin_in as _Float16; hipMemcpy to R.dev_in)
    // 4. eval under the mutex, D2H dev_out -> pin_out
    {
        std::lock_guard<std::mutex> lk(c->gpu_eval_mtx);
        /* migraphx::program_parameters binding dev_in/dev_out; outs = prog.eval(pp); */
        hipDeviceSynchronize();
    }
    // 5. fp16 -> fp32, RGB->YUV (BT.709 bilinear), crop centered w x h into out
    rife_cpu::rgb_planes_to_yuv420(/*pin_out fp32*/, R.g, *(aji_frame*)out, a->range);
    return AJI_OK;
}
```
(Fill the elided staging with the exact `run_chain` H2D/eval/D2H pattern `:695-728`, binding `R.dev_in`/`R.dev_out`. `(_Float16)t` cast per Global Constraints.)

- [ ] **Step 3: Build**

Run: `cmake --build build -j`
Expected: clean build; all four RIFE symbols still exported (verify: `nm -D build/...aji_rocm.so | grep -E 'aji_(scale_factor|rife_factor|rife_before_upscale|infer_rife)'` shows all four).

- [ ] **Step 4: In-player verification (manual — no automated ROCm harness exists)**

Play a RIFE chain (`rife_model=414`, factor 2/1) on a clip with motion and a hard cut.
Expected: output FPS doubles; intermediate frames are smoothly interpolated (not duplicates); across the hard cut, interpolation is skipped (no ghosting blend) — confirming SCDetect. Then test a fractional factor (e.g. `rife_factor_num=5, rife_factor_den=2`) to exercise non-0.5 `t`.

- [ ] **Step 5: Commit**

```bash
git add src/aji_rocm.cpp
git commit -m "feat(rife): aji_rife_factor/before_upscale + 4:2:0 aji_infer_rife"
```

---

# Phase B — 444 + GPU-resident (nice-to-have)

New HIP kernels + a 444 output path + one filter boolean + a device-resident infer path.

## Task B1: HIP color kernels `pre444` / `post444`

**Files:**
- Modify: `src/aji_rocm_color.hip`, `src/aji_rocm_color.h`
- Modify: the color self-test (`aji_color_selftest`, referenced from `aji_rocm_color.hip`)

**Interfaces:**
- Produces: `extern "C" void aji_gpu_pre444(void* y,void* cb,void* cr, ptrdiff_t ys,ptrdiff_t cs, int w,int h, aji_color_csp csp, void* rgb_fp16, void* stream)` and `aji_gpu_post444(void* rgb_fp16, int w,int h, aji_color_csp csp, void* y,void* cb,void* cr, ptrdiff_t ys,ptrdiff_t cs, void* stream)`. Ported from `kernels.cu:278-316` (pre) and `kernels.cu:236-276` (post). `post444` writes u16 to ALL 3 planes, hardcodes `qdiv=1, qmax=65535`, does NOT branch the store on `is_p010`.

- [ ] **Step 1: Write the failing round-trip test**

In the color self-test, add a 444 case: build a small YUV444P16 mid-gray + gradient input, run `aji_gpu_pre444` then `aji_gpu_post444`, assert `|out − in| ≤ 1` per channel.

- [ ] **Step 2: Run — verify it fails** (`aji_gpu_pre444`/`post444` undefined).

Run: `cmake --build build -j --target aji_color_selftest && ./build/aji_color_selftest`
Expected: FAIL (link error).

- [ ] **Step 3: Port the kernels** (mechanical CUDA→HIP from `kernels.cu`)

Copy `k_pre444` (`kernels.cu:278-302`) and `k_post444` (`kernels.cu:236-261`) into `aji_rocm_color.hip` with `__half→_Float16`, `cudaStream_t→hipStream_t`. Reuse `aji_quant` (`:18-22`) for the post store. Add `extern "C"` launchers following `aji_gpu_out_color` (`:97-111`): `dim3((w+31)/32,(h+7)/8), dim3(32,8)`, device ptrs as `void*`, `aji_color_csp` by value (`is_p010=0` for 444), `void` return. Declare both in `aji_rocm_color.h`.

- [ ] **Step 4: Run — verify it passes**

Run: `cmake --build build -j --target aji_color_selftest && ./build/aji_color_selftest`
Expected: round-trip PASS within tolerance 1.

- [ ] **Step 5: Commit**

```bash
git add src/aji_rocm_color.hip src/aji_rocm_color.h
git commit -m "feat(rife): HIP pre444/post444 444 color kernels"
```

## Task B2: HIP kernels `rife_consts` / `fill_f16` / `fill_plane` / `scd_diff`

**Files:**
- Modify: `src/aji_rocm_color.hip`, `src/aji_rocm_color.h`, color self-test

**Interfaces:**
- Produces: `aji_gpu_rife_consts(void* in_tensor, int pw,int ph, void* stream)` (writes ch7-10, then the launcher does a device-wide sync); `aji_gpu_fill_f16(void* dst, size_t count, float value, void* stream)` (1-D grid, block 256); `aji_gpu_fill_plane(void* plane, ptrdiff_t stride, int w,int h, unsigned value, int bytes, void* stream)`; `aji_gpu_scd_diff(const void* ya, ptrdiff_t sa, const void* yb, ptrdiff_t sb, int w,int h, float norm, void* accum_f32, int bytes, void* stream)` (block MUST be 32×8=256).

- [ ] **Step 1: Write failing tests** (consts formulas at corners; fill value; scd_diff sum vs a CPU reference over a small frame).

- [ ] **Step 2: Run — verify it fails.**

Run: `cmake --build build -j --target aji_color_selftest && ./build/aji_color_selftest`
Expected: FAIL.

- [ ] **Step 3: Port the four kernels** from `kernels.cu:513-535` (consts), `:537-552` (fill_f16, 1-D block 256), `:554-577` (fill_plane, u8/u16), `:579-626` (scd_diff — keep block 32×8 so `__shared__ float partial[256]` and the power-of-2 tree reduction match; `norm` peak `format==YUV444P16 ? 1/65535 : 1/65472`). Add launchers + prototypes.

- [ ] **Step 4: Run — verify it passes.**

Run: `cmake --build build -j --target aji_color_selftest && ./build/aji_color_selftest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aji_rocm_color.hip src/aji_rocm_color.h
git commit -m "feat(rife): HIP rife_consts/fill_f16/fill_plane/scd_diff kernels"
```

## Task B3: Engine 444 output writer + lift the `:966` rejection

**Files:**
- Modify: `src/aji_rocm.cpp` (`aji_infer` output check `:966`, `gpu_post` `:471-548`, `GpuColor`/`gpu_color_ensure` `:671-672`, `aji_gpu_out_color` call site `:835-837`)

**Interfaces:**
- Produces: a `gpu_post444` path (3 tight full-res u16 planes → `out->plane[0..2]`) usable from BOTH `run_chain_gpu` (device) and the CPU `gpu_post` fallback. `aji_infer` accepts `AJI_FMT_YUV444P16` output.

- [ ] **Step 1: Lift the output rejection**

`:966` — allow `AJI_FMT_YUV444P16` as an output format (leave the input check `:965`).

- [ ] **Step 2: Add the 444 writer**

GPU path: a launcher `aji_gpu_out_color444` (post444 over the upscaler's device RGB → 3 device u16 planes) invoked from `run_chain_gpu` when `out->format==YUV444P16`. CPU fallback: a 444 branch in `gpu_post` using `rife_cpu` matrix (no chroma resample). Add a 444 variant to `GpuColor`/`gpu_color_ensure` (no `hu/hv/Un/Vn` downsample scratch).

- [ ] **Step 3: Build + smoke test (444 upscale output, no RIFE yet)**

Run: `cmake --build build -j`; play an upscale chain forcing `output_444` after the filter gate (B4) is in — or temporarily request 444 output via a test. Expected: builds; 444 output plays with correct color.

- [ ] **Step 4: Commit**

```bash
git add src/aji_rocm.cpp
git commit -m "feat(rife): engine YUV444P16 output writer (GPU + CPU paths)"
```

## Task B4: Filter 444 gate (mpv-fork repo)

**Files:**
- Modify: `mpv-fork/video/filter/vf_animejanai.c:456` (in the `animejanai-linux` repo)

- [ ] **Step 1: Drop the `!is_sw` clause**

`:456` `if (p->opts->output_444 && !p->is_d3d11 && !p->is_sw)` → `if (p->opts->output_444 && !p->is_d3d11)`. (sw pool alloc `:531-535`, frame-builders `:641-642,:798-806`, and the `out_params` fixup `:462-469` already carry 444.)

- [ ] **Step 2: Build the filter / mpv fork**

Run the mpv-fork build. Expected: clean build.

- [ ] **Step 3: Commit (in the animejanai-linux repo)**

```bash
git -C /home/nathan/Projects/animejanai-linux add mpv-fork/video/filter/vf_animejanai.c
git -C /home/nathan/Projects/animejanai-linux commit -m "feat(rife): allow 444 output on the sw/ROCm path"
```

## Task B5: Device-resident 444 `aji_infer_rife`

**Files:**
- Modify: `src/aji_rocm.cpp` (`RifeState`, `setup_rife` const fill via `aji_gpu_rife_consts`, `aji_infer_rife` 444 branch)

**Interfaces:**
- Consumes: B1/B2 launchers, `run_chain_gpu` device-binding pattern.
- Produces: a 444 branch in `aji_infer_rife` that stays GPU-resident.

- [ ] **Step 1: Add Phase-B device buffers to `RifeState`**

`in_tensor` (`11*plane*2`, consts in ch7-10 written ONCE by `aji_gpu_rife_consts` at setup — persistent), `out_tensor` (`3*plane*2`), `pad_a/pad_b/pad_o` (`3*py` each), `scd` (device float). Borders black-filled once via `aji_gpu_fill_plane`.

- [ ] **Step 2: Implement the 444 branch in `aji_infer_rife`**

When `a->format==AJI_FMT_YUV444P16`: D2D stage the 3 planes into `pad_a/pad_b` at `pad_t/pad_l` offsets → `aji_gpu_pre444` into `in_tensor` ch0-5 → `aji_gpu_fill_f16` ch6 = `t` → `aji_gpu_scd_diff` + a single D2H readback (`hipStreamSynchronize`); if `sum/(pw*ph) > scd_threshold` return `AJI_SCENE` → else `eval` binding `in_tensor`/`out_tensor` directly (under `gpu_eval_mtx`) → `aji_gpu_post444` into `pad_o` → D2D crop into `out->plane[0..2]`. No host RGB round-trip.

- [ ] **Step 3: Build**

Run: `cmake --build build -j`
Expected: clean build.

- [ ] **Step 4: In-player verification (manual)**

Play an upscale+RIFE chain with `output_444` enabled. Expected: interpolation works AND chroma is full-resolution (compare a saturated-color, high-motion clip vs the Phase-A 4:2:0 path — no chroma bleed/round-trip loss); steady-state has no stutter from the per-frame sync.

- [ ] **Step 5: Commit**

```bash
git add src/aji_rocm.cpp
git commit -m "feat(rife): device-resident 444 aji_infer_rife path"
```

---

## Self-Review (against the spec)

- **§7.1 rife_model_dir plumbing** → A6. **§7.2 RifeState** → A6 (Phase-A buffers), B5 (Phase-B buffers). **§7.3 dedicated BT.709/bilinear color** → A5. **§7.4 setup_rife + cascade + OSD + loaded-gating** → A8. **§7.5 load_model output-by-name** → A7. **§7.6 ABI (factor gated on loaded)** → A9. **§7.7 per-frame 4:2:0** → A9.
- **§4 model-name/geometry/scene-detect/consts** → A2/A3/A4/A5. **§6 gpu_eval_mtx + retrofit** → A6.
- **§8.1 HIP kernels** → B1/B2. **§8.2 engine 444 writer (both paths)** → B3. **§8.3 filter gate + no API bump** → B4. **§8.4 device-resident** → B5. **§8.5 444-only-on-upscale-chain** → documented limitation (no task; nothing to build).
- **§9 parity traps** → enforced across A4/A5/A9 (scene-detect divisor, BT.709, RNE timestep, border black) and B1/B5 (qmax 65535, no 444 siting).
- **§10 testing** → unit tests A2-A5,B1-B2; migraphx smoke A7; in-player A8/A9/B5. **Known gap:** no automated end-to-end ROCm parity harness (spec §10) — not built here; manual play + optional offline frame-diff.
- **§5 ABI export invariant** → verified in A9 step 3 (`nm -D`).

**Placeholder scan:** elided staging in A9 step 2 / A5 step 3 point at exact reference line ranges (`run_chain :695-728`, `aji_dml record_pre/post :690-779`) — concrete source, not "TBD". Mechanical kernel ports (B1/B2) cite exact `kernels.cu` ranges + the substitution recipe.

**Type consistency:** `rife_cpu::Geom`, `RifeState`, the four `aji_gpu_*444`/helper launcher signatures, and `aji_rife_factor/before_upscale/infer_rife` signatures are used identically across tasks and match `include/aji.h:184-216`.

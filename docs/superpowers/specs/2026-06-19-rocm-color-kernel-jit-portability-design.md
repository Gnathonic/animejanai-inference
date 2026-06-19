# ROCm color-kernel JIT portability — design

**Date:** 2026-06-19
**Repo:** `animejanai-inference` (engine). Packaging touches in `animejanai-linux`.
**Status:** Approved design, pre-implementation.

## Problem

The distributed Linux/AMD build (`mpv-upscale-2x_animejanai-v0.4.3-linux.tar.gz`, backend
`ROCm`) fails on any machine whose GPU is not a `gfx1201` (RX 9070 XT). The user hit
"the inference engine being compiled for the current GPU only" when moving the tarball to
another system.

### Root cause (verified)

`libaji_rocm.so` embeds exactly one GPU code object. `roc-obj-ls` on the shipped library:

```
hipv4-amdgcn-amd-amdhsa--gfx1201   ... size=15160
```

The hand-written HIP color kernels (`src/aji_rocm_color.hip`) are compiled at build time
for a single arch, hardcoded in `CMakeLists.txt:171-173`:

```cmake
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1201)   # gfx1201 = RDNA4 (RX 9070 XT)
endif()
```

On any other AMD GPU these kernels fail to launch (`hipErrorNoBinaryForGpu`,
"no kernel image is available for execution on the device").

### What is NOT the cause

- **MIGraphX model inference** is already portable: it JIT-compiles the `.onnx` to the
  device present at runtime. The `.mxr` engine files are a runtime cache and are **not**
  shipped in the tarball (`tar tzf | grep -c .mxr` → 0).
- The `.mxr` cache key omitting the GPU arch (`mxr_cache_path` keys on
  `(onnx, W, H)` only) is a separate latent bug but did not cause this failure and is
  **out of scope** here.

## Scope

Target: **any AMD GPU with a compatible ROCm installed.** Bundling the ROCm runtime, and
NVIDIA/Intel portability (the retired Vulkan backend), are out of scope.

Hard constraint from the user: **the CPU color path is too slow to use as the normal
path.** The solution must keep color on the GPU on every supported machine.

## Chosen approach: compile the color kernels on the user's machine with hipRTC

Mirror what MIGraphX already does for the model: compile for the device that is actually
present, at runtime, and cache the result. HIP's runtime compiler **hipRTC** (`libhiprtc`,
part of every ROCm install — the NVRTC analog) takes kernel source as a string and emits a
code object for the current GPU arch. No baked arch list, no fat binary, no CPU fallback in
the normal flow.

Rejected alternative: shelling out to `hipcc` at runtime would require the full clang
toolchain on the user's machine, not just the ROCm runtime. hipRTC is strictly lighter and
is the purpose-built tool.

### Architecture

The color subsystem splits into three units with clear boundaries:

1. **Embedded kernel source (build-time data).**
   `aji_rocm_color.hip` remains the single source of truth for the device code
   (`__device__` helpers + `__global__` kernels). At build time its device-side section is
   embedded into `libaji_rocm.so` as a C string (generated header, e.g.
   `aji_rocm_color_src.h`, produced by a small CMake step / `xxd`-style codegen from the
   same `.hip` file). One source → the bit-exact color parity already validated is
   preserved; no second copy to drift.

   *What it depends on:* nothing at runtime. *Interface:* a `const char*` of HIP source.

2. **JIT compile + on-disk cache (`aji_rocm_color_jit`, new unit in `aji_rocm.cpp` or a
   sibling `.cpp`).**
   On first use:
   - query the device arch via `hipGetDeviceProperties().gcnArchName`;
   - compute a cache path `<cacheDir>/aji_color.<gcnArchName>.<srcHash>.co` where
     `srcHash` is a hash of the embedded source + compile flags (so a kernel change
     invalidates stale objects automatically);
   - if the cached code object exists, `hipModuleLoadData` it; on **any** load failure,
     delete it and recompile (self-healing — same robustness pattern as the `.mxr` flow);
   - otherwise compile the embedded source with `hiprtcCreateProgram` /
     `hiprtcCompileProgram` (targeting the queried arch), write the code object to a temp
     file and atomically `rename` into place (an interrupted compile never leaves a
     half-written object), then load it.
   - Resolve each kernel via `hipModuleGetFunction` once and keep the `hipFunction_t`
     handles on the context.

   *Interface:* `bool aji_color_jit_ensure(ctx)` returns true once the module + function
   handles are ready; `false` only on catastrophic failure.

3. **Launchers (modified `extern "C"` functions in `aji_rocm_color`).**
   Replace the `<<<grid, block, shmem, stream>>>` triple-chevron launches with
   `hipModuleLaunchKernel(fn, grid…, block…, shmem, stream, argPtrs, nullptr)`, packing
   each kernel's args into a `void*` array. Grid/block/shared-mem/stream are unchanged, so
   runtime behavior is identical.

### Cache location

A `cache/` directory, sibling of the models/`onnx/` directory (i.e. `animejanai/cache/`),
created on demand. Overridable via env var `AJI_ROCM_CACHE_DIR`. Code objects are named
`aji_color.<gcnArchName>.<srcHash>.co`.

(The `.mxr` model engines stay where they are for now — next to the models. Relocating them
into the same `cache/` dir is a possible consistency follow-up but is out of scope.)

### Build changes (`CMakeLists.txt`)

- Drop `enable_language(HIP)`, the `CMAKE_HIP_ARCHITECTURES gfx1201` block, and compiling
  `src/aji_rocm_color.hip` as a HIP translation unit.
- Add the codegen step that turns `aji_rocm_color.hip` into the embedded-source header.
- Build `aji_rocm` as plain C++ (g++), linking `libhiprtc` in addition to the existing
  `libmigraphx_c` + `libamdhip64`. Result: **`libaji_rocm.so` contains zero embedded device
  code and is arch-agnostic by construction.**

### Parity guard

hipRTC must compile with the same floating-point flags the static `hipcc` build used
(fp-contract setting, no fast-math, fp16 handling) so the JIT'd kernels stay bit-exact
against the reference. Carry those flags through `hiprtcCompileProgram`. After
implementation, re-run the existing color self-test (`aji_color_selftest`) and the color
parity check on this `gfx1201` box, and confirm identical output to the pre-change build.

### Failure behavior

If hipRTC compilation itself fails (e.g. a genuinely unsupported/unknown arch, or a broken
ROCm), surface a clear, actionable error naming the arch and the cache path. The existing
CPU color path remains in the code as a last-resort safety net only; it is never the normal
operating path. `backend_auto_fallback` semantics are unchanged.

## Dependencies / prerequisites

- Target needs a ROCm install whose `libhiprtc`, `libmigraphx_c`, and `libamdhip64`
  sonames match the build (currently `libmigraphx_c.so.3`, `libamdhip64.so.7`). Documented
  as a prerequisite; bundling ROCm is out of scope.

## Out of scope

- Bundling the ROCm runtime into the tarball.
- NVIDIA/Intel / Vulkan portability.
- `.mxr` cache-key arch hardening and relocation (separate latent issue).

## Acceptance

1. A `libaji_rocm.so` built on the `gfx1201` box contains **no** embedded GPU code object
   (`roc-obj-ls` shows host only / errors with no device code).
2. On the `gfx1201` box: first run JIT-compiles the color kernels into
   `animejanai/cache/aji_color.gfx1201.<hash>.co`; subsequent runs load from cache; color
   output is bit-exact vs the pre-change build (self-test + parity pass).
3. The same tarball runs with GPU color on a second AMD arch (different `gfx`), compiling
   its own cached code object on first run — no rebuild, no CPU fallback.

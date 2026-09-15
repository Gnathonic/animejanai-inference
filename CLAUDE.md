# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`libaji` — a standalone C-ABI inference engine for real-time anime upscaling and RIFE frame
interpolation. It takes GPU-resident YUV frames, runs the chain described by `animejanai.conf`
(resize + ONNX models + RIFE) entirely on the GPU, and returns YUV.

The strict C ABI in `include/aji.h` is the point: `vf_animejanai` in the
[`the-database/mpv`](https://github.com/the-database/mpv) fork loads the library at **runtime**,
so mpv never links TensorRT. The backend and the TRT version can be swapped without rebuilding
the player, and on Windows an MSVC-built shim coexists with a mingw-built mpv.

`libaji` has **zero mpv dependency** — `aji_encode` (offline CLI) and `VideoJaNai` use the same
ABI.

**This branch (`linux-vulkan-backend`) adds two Linux backends** on top of upstream's
TensorRT-only engine: `aji_rocm` (AMD ROCm/MIGraphX) and `aji_vk` (ncnn-Vulkan, vendor-neutral),
both with RIFE. The branch name is a misnomer — it is the whole Linux/AMD line. See
[Linux/AMD backends](#linuxamd-backends-aji_rocm-aji_vk) below.

## Documentation map

- **[`docs/BUILD-WINDOWS.md`](docs/BUILD-WINDOWS.md)** — the Windows dependency setup and
  release build. **There is no Windows CI**: `aji-windows-x64.zip` is built and uploaded by hand
  from this workstation, so this doc is the only record of how.
- **[`docs/BUILD-LINUX.md`](docs/BUILD-LINUX.md)** — the Linux/WSL build, the CI workflow, and
  the parity harness, plus the optional `aji_rocm` / `aji_vk` builds.
- **[`BUILD.md`](BUILD.md)** — the ROCm backend in depth: build-host vs. target-machine
  requirements, the hipRTC JIT cache, and the color parity test.

`README.md` is the public-facing overview. Its "Build (Linux)" section is still accurate; the
prose describing the engine as "Phase 0 spike scope" predates chain selection, RIFE, and the
DirectML backend.

## Architecture

```
vf_animejanai (mpv)  ──dlopen──►  aji.dll / libaji.so        (aji_dispatch.cpp — thin dispatcher)
                                        │ forwards over aji.h
              ┌───────────────┬─────────┴─────────┬─────────────────┐
        aji_trt.dll       aji_dml.dll        libaji_rocm.so     libaji_vk.so
      TensorRT + CUDA   DirectML + D3D12   MIGraphX + HIP     ncnn + Vulkan
                        (Windows only)     (Linux/AMD only)   (Linux, any GPU)
```

`aji` is deliberately dependency-free — no CUDA, no TensorRT, no ORT — so the player can load it
on any machine and the backend resolves later. The backends load `onnxruntime.dll` /
`DirectML.dll` from their own directory at runtime, which is why nothing is import-linked for
the DML path.

The dispatcher maps the conf's `[global] backend=` key (case-insensitive) to a library stem:
`directml` → `aji_dml`, `rocm` → `aji_rocm`, `vulkan` or `ncnn` → `aji_vk`, anything else →
`aji_trt`. There is no `vulkan`→`rocm` alias in the engine.

**Built-in slots are code, not config.** `add_builtin_slots` in `src/aji_conf.cpp` hardcodes
slots 1001–1003 (Quality/Balanced/Performance) and 1010–1013 (benchmark / RIFE-order templates),
**including the ONNX model filenames** via the `HD_BAL` / `HD_PERF` / `SD` constants. A model
rename has to be made here, in the package's committed `animejanai/onnx/`, and in the AnimeJaNai
Manager's default profiles.

User slots 1–9 come from `animejanai.conf`, which the Manager writes.

## Build system

CMake only (`cmake_minimum_required(VERSION 3.24)`, C11 / C++17). There is no Cargo, meson,
vcpkg, or conan manifest anywhere.

On this branch `project()` declares only `C CXX`; CUDA is probed with `check_language(CUDA)`
and enabled when a toolchain is found (`AJI_HAVE_CUDA`). Every backend gates on its own
dependencies and prints a one-line `-- ... skipping` status when they are absent, so the tree
configures on a box with **no** CUDA, ROCm or ncnn (producing just `aji` and `rife_selftest`).
Upstream's `find_package(CUDAToolkit REQUIRED)` behaviour is preserved whenever CUDA is present.

### Cache options

| Option | Type | Default (Windows) | Default (Linux) |
|---|---|---|---|
| `AJI_TRT_ROOT` | PATH | `""` | `$ENV{HOME}/sdk/tensorrt/usr` |
| `AJI_NVINFER` | STRING | `nvinfer_10` | `nvinfer` (plain `set`, not cached) |
| `AJI_FFMPEG_ROOT` | PATH | `""` | *(unused; pkg-config instead)* |
| `AJI_ORT_ROOT` | PATH | `""` | *(Windows only)* |
| `AJI_DML_ROOT` | PATH | `""` | *(Windows only)* |
| `AJI_ROCM_ROOT` | PATH | *(Linux only)* | `/opt/rocm` |
| `AJI_NCNN_ROOT` | PATH | *(Linux only)* | `""` (aji_vk skipped unless set) |

The include/lib layout under `AJI_TRT_ROOT` differs by platform: Windows expects headers in
`<root>/include` and libs at `<root>` itself; Linux expects `<root>/include/x86_64-linux-gnu`
and `<root>/lib/x86_64-linux-gnu`.

> **`AJI_NVINFER` defaults to `nvinfer_10` but the project now builds against TensorRT 11.**
> Every current build passes `-DAJI_NVINFER=nvinfer_11` explicitly. Omitting it on Windows
> produces a link error for a library that is not there.

### Targets

| Target | Kind | Links | Notes |
|---|---|---|---|
| `aji` | SHARED | `${CMAKE_DL_LIBS}` | the dispatcher; `CXX_VISIBILITY_PRESET hidden`, PIC on |
| `aji_trt` | SHARED | `${AJI_NVINFER}`, `CUDA::cuda_driver`, `CUDA::cudart` | non-Windows adds `-Wl,-Bsymbolic`; needs CUDA |
| `aji_dml` | SHARED | `d3d12 d3d11 dxgi d3dcompiler` | **WIN32 only**; ORT/DirectML loaded at runtime, not import-linked |
| `aji_rocm` | SHARED | `migraphx_c`, `amdhip64`, `hiprtc`, OpenMP | **Linux, needs ROCm**; `-Wl,-Bsymbolic`; zero baked device code |
| `aji_rocm_compile` | EXE | `migraphx_c`, `amdhip64` | out-of-process MIGraphX engine compiler, ships next to `libaji_rocm.so` |
| `aji_vk` | SHARED | `ncnn`, `vulkan`, OpenMP | **Linux, needs `AJI_NCNN_ROOT`**; copies `spv/` next to the `.so` post-build |
| `aji_harness` | EXE | `aji`, `CUDA::cudart` | depends on `aji_trt`; needs CUDA |
| `aji_harness_dml` | EXE | `aji`, `d3d11` | **WIN32 only** |
| `aji_encode` | EXE | `aji`, CUDA, ffmpeg libs | **conditional** — see below; needs CUDA |
| `aji_kernel_test` | EXE | `CUDA::cudart` | kernel unit tests; needs CUDA |
| `rife_selftest` | EXE | — | CPU-only RIFE unit tests; always built |
| `aji_rocm_color_test`, `rife_harness`, `rife_pipeline_bench`, `rocm_rife_transcode` | EXE | `aji_rocm` | ROCm tests/tools; built with `aji_rocm` |

**`aji_encode` is skipped silently** unless its ffmpeg dependency resolves. On Windows that
means `AJI_FFMPEG_ROOT` must be set; on Linux, pkg-config must find `libavformat libavcodec
libavutil libavfilter libswscale`. The configure output says which:

```
-- AJI_FFMPEG_ROOT not set; skipping aji_encode target
-- ffmpeg/libav not found via pkg-config; skipping aji_encode target
```

If a release build comes out without `aji_encode.exe`, this is why.

### The CUDA-architecture trap

`CMakeLists.txt` sets a full arch list **only** `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 100-real 120-real 120-virtual)
```

> **Upstream's fallback never fires.** CMake pre-defines `CMAKE_CUDA_ARCHITECTURES` during
> `project(... CUDA)`, so omitting `-DCMAKE_CUDA_ARCHITECTURES=...` yields a **single default
> arch** (e.g. `sm_75`) that runs on one GPU generation and dies on every other. This branch
> sets the list before `enable_language(CUDA)` so the fallback can fire, but that is unverified
> on a CUDA box — the release script and the Linux CI workflow still pass the full list
> explicitly. Never drop it from a build that ships.

For local dev on one machine, `-DCMAKE_CUDA_ARCHITECTURES=120` (or `native`) is much faster to
compile — the release list builds eight architectures.

## Linux/AMD backends (aji_rocm, aji_vk)

Both are optional: a missing ROCm or ncnn only prints a `-- ... skipping` status, so upstream's
TensorRT-only CI build is unaffected. Full recipe in `docs/BUILD-LINUX.md`; ROCm runtime
details in `BUILD.md`.

- **`aji_rocm`** (`src/aji_rocm.cpp`, `src/aji_rocm_color.cpp`): direct `libmigraphx_c` ("Path
  B"), **not** ONNX Runtime — the Arch `onnxruntime-rocm` package is broken (no ROCm EP; the
  MIGraphX EP cache segfaults). Runs the same fp16 `.onnx` as `aji_trt`/`aji_dml`. Needs only a
  C++ compiler + ROCm headers/libs at `AJI_ROCM_ROOT`; no HIP compiler and no gfx arch list.
- **`aji_vk`** (`src/aji_vk.cpp`, `shaders/` → `spv/`): ncnn-Vulkan, any Vulkan GPU. ncnn MUST be
  built from source with Vulkan (the pip wheel and stock pnnx/onnx2ncnn are broken for our
  models); on this workstation that tree is `~/Projects/ncnn-vk` (`build-fast/`, carries the
  custom `gridsample_vulkan` layer RIFE's warps need). Configure with
  `-DAJI_NCNN_ROOT=/home/nathan/Projects/ncnn-vk`; CMake looks for `libncnn.so` under
  `<root>/build/src`, `<root>/build-fast/src` or `<root>/lib`. ncnn `.param`/`.bin` models are
  converted from the fp16 `.onnx` (`tools/rife_ncnn/` for RIFE; the upscale converters live in
  `animejanai-linux/spikes/`).
- The workstation for this branch: AMD RX 9070 XT (RDNA4, gfx1201), ROCm 7.2.4, **no NVIDIA** —
  `aji_trt`/`aji_encode` never build here; TRT changes get runtime-verified on a separate
  RTX 4090 box.

### Tests

```sh
./build/rife_selftest                       # CPU, always
./build/aji_rocm_color_test                 # run from the repo root (reads test/fixtures/);
                                            # expect maxdiff=0 for NV12 and P010
./build/rife_harness                        # RIFE end-to-end on ROCm; first run compiles ~1-3 min
```

`rife_harness` / `rife_pipeline_bench` default to a hardcoded v0.4.3 install dir; override with
`AJI_RIFE_DIR` / `AJI_MODEL_DIR`. `tools/run_rife_transcode.sh` (ffmpeg piped through
`rocm_rife_transcode`) is the offline RIFE proof on AMD — `aji_encode` is CUDA/NVDEC-only.

### ROCm gotchas (each cost real time to rediscover)

- **ALWAYS wrap headless ROCm/mpv runs in `timeout -s KILL 300`** — known hang modes at exit
  and when the HIP ring is full. Plain `timeout` is not enough (SIGTERM is swallowed).
- Color kernels (`src/aji_rocm_color_device.hip`) are embedded as a string and **hipRTC-JIT'd
  per GPU at runtime** into `animejanai/cache/aji_color.<gfx>.<hash>.co` (`AJI_ROCM_CACHE_DIR`
  overrides). Portability invariant: `roc-obj-ls build/libaji_rocm.so` must print
  "No kernel section found" — baking one gfx arch into the `.so` broke every non-gfx1201 GPU.
- NEVER set `MIGRAPHX_DISABLE_MLIR=1`: the MIOpen fallback reads uninitialized workspace at 4K
  (column static / non-deterministic output). MLIR stays on; the cost is compile time only.
- MIGraphX engines are STATIC per resolution (dynamic shapes crash). First compile ~2-3 min,
  cached as `.mxr` by `aji_rocm_compile` out of process (a player quit mid-compile just
  SIGKILLs the child); async build + "Building ... engine" progress text already exist.
- Benchmarks need a SINGLE GPU job — a second migraphx/mpv process corrupts eval timings.
- CPU color (`src/resample.h`, OpenMP): cap threads ~8 (32 is slower than 1 on small frames);
  reuse ctx scratch buffers — per-frame allocation churn shows up as playback stutter.
- mpv parses `animejanai.conf` only at startup (`aji_create`) — conf edits need a full restart.
- Feed RIFE transcodes with `-fps_mode passthrough` on decode, or dropped frames cause judder.
- ncnn-Vulkan is ~2.4x faster than ROCm on the warp-heavy RIFE pipeline; ROCm wins on conv-only
  upscaling. Real-time mpv stays upscale-only (the RIFE-first filter path is off by default).

## Where the artifacts go

| Platform | Built by | Asset |
|---|---|---|
| Linux | `.github/workflows/build-linux.yml` | `aji-linux-x64.tar.zst` |
| Windows | **locally, by hand** (`docs/BUILD-WINDOWS.md`) | `aji-windows-x64.zip` |

Both are attached to the **same release tag**, because the consumer derives both URLs from one
`AjiVersion` constant. `the-database/mpv-AnimeJaNai`'s assembler downloads them in `InstallAji`
and extracts them flat into `animejanai/inference/`.

RIFE model weights ship from this repo too, under a separate tag
(`RifeModelsVersion`, currently `models-rife-fp16-1`), converted with
`tools/convert_rife_fp16.py`.

The Linux/AMD libraries (`libaji_rocm.so` + `aji_rocm_compile`, `libaji_vk.so` + `spv/` +
ncnn models) are packaged by the umbrella `animejanai-linux` project
(`packaging/`, `rocm`/`vulkan` component packs), not by the CI workflow here. NEVER bundle
`libvulkan.so.1` in a portable package — the loader must come from the target system.

## The ABI contract

`include/aji.h`'s `AJI_API_VERSION` is shared with `video/filter/aji.h` in the mpv fork —
**the two files must agree**. When the ABI changes:

1. update both headers,
2. rebuild and release both the engine and the mpv builds,
3. bump `AjiVersion` **and** `MpvForkVersion`/`MpvForkLinuxVersion` in the assembler.

The filter lives on the mpv fork's `master` (aji ABI v8). The old standalone `vf-animejanai`
branch is stale (ABI v4) and must not be used.

## Conventions

- Commits: author `the-database`, short imperative subject, no co-author trailers.
- `.gitignore` covers `build/`, `models/`, `*.engine`, and fixture binaries. The out-of-repo
  Windows build dirs are excluded locally via `.git/info/exclude`
  (`build-win/`, `build-win-trt11/`, `aji-build*/`) — note `build-win-release/` is in **neither**,
  so it shows up as untracked.
- ONNX models for the DirectML backend must be **opset ≤ 21**: the bundled ORT DirectML EP only
  registers `Conv`/`PReLU` kernels through opset 21, and a model exported at opset ≥ 22 silently
  falls back to the CPU EP (roughly 2000× slower per frame, which looks like a hang). Verify
  placement with `AJI_ORT_VERBOSE=1`.

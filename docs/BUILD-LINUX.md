# Building libaji on Linux

Windows build: [`BUILD-WINDOWS.md`](BUILD-WINDOWS.md). Engine overview:
[`../CLAUDE.md`](../CLAUDE.md).

Unlike Windows, Linux **is** covered by CI: `.github/workflows/build-linux.yml` produces the
`aji-linux-x64.tar.zst` release asset. Local Linux work on this workstation happens in WSL.

## Local build

```sh
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S .
cmake --build build -j
```

Requirements for the TensorRT backend: **CUDA toolkit 13.x** and **TensorRT 11.x**. CUDA is
probed, not required: without `nvcc` the configure prints
`-- No CUDA toolchain found; skipping the TensorRT backend (aji_trt) and CUDA tools.` and
builds the dispatcher plus whichever of the [other Linux backends](#other-linux-backends-aji_rocm-and-aji_vk)
have their dependencies.

`AJI_TRT_ROOT` defaults to `$HOME/sdk/tensorrt/usr` on Linux. For a system TensorRT from
NVIDIA's apt repo, pass `-DAJI_TRT_ROOT=/usr` instead. The Linux layout under that root is
`<root>/include/x86_64-linux-gnu` and `<root>/lib/x86_64-linux-gnu` — different from Windows.
`AJI_NVINFER` is plain `nvinfer` here and needs no override.

`aji_trt` gets `BUILD_RPATH`/`INSTALL_RPATH` pointing at `${AJI_TRT_LIB}` and links with
`-Wl,-Bsymbolic`, so the built `.so` finds TensorRT without `LD_LIBRARY_PATH` gymnastics.

> **Pass `-DCMAKE_CUDA_ARCHITECTURES` explicitly for anything you intend to ship.** Do not
> rely on the `if(NOT DEFINED ...)` default in `CMakeLists.txt` — see the CUDA-architecture
> trap in [`../CLAUDE.md`](../CLAUDE.md). For local iteration on one machine,
> `-DCMAKE_CUDA_ARCHITECTURES=native` (or your SM, e.g. `120`) compiles far faster than the
> eight-architecture release list.

`aji_encode` builds only if pkg-config finds `libavformat libavcodec libavutil libavfilter
libswscale`; otherwise CMake prints
`-- ffmpeg/libav not found via pkg-config; skipping aji_encode target` and moves on. Install
your distro's ffmpeg dev packages if you need the offline encoder.

There are no DirectML targets on Linux — `aji_dml` and `aji_harness_dml` are `WIN32` only.
TensorRT is the NVIDIA backend; AMD and vendor-neutral GPUs use `aji_rocm` / `aji_vk` below.

### Quick manual test, no player needed

Build an engine:

```sh
trtexec --onnx=models/<model>.onnx --fp16 \
        --minShapes=input:1x3x64x64 --optShapes=input:1x3x1080x1920 \
        --maxShapes=input:1x3x1088x1920 --saveEngine=models/<model>.engine
```

then run frames through the harness:

```sh
ffmpeg -i clip.mkv -frames:v 12 -pix_fmt p010le -f rawvideo in.raw
./build/aji_harness --engine models/<model>.engine --input in.raw \
    --width 1400 --height 1080 --format p010 --output out.raw
```

It reports device-side ms/frame for the whole pre + infer + post chain.

Or through mpv directly:

```sh
mpv --hwdec=nvdec --vf=animejanai=engine=models/<model>.engine:lib=build/libaji.so video.mkv
```

## Other Linux backends: `aji_rocm` and `aji_vk`

Both are **optional** and gated on their own dependencies in `CMakeLists.txt`: each
`find_library`/`find_path` group either enables the target or prints a `-- ... skipping`
status, so a box without ROCm or ncnn (the CI image included) still gets the TensorRT-only
build. The dispatcher picks them from `animejanai.conf`: `[global] backend=rocm` loads
`libaji_rocm.so`, `backend=vulkan` (or `ncnn`) loads `libaji_vk.so`; anything else is TensorRT.

| Backend | Runs on | Build needs | Configure status line |
|---|---|---|---|
| `aji_rocm` | AMD (ROCm) | `/opt/rocm` with MIGraphX (`libmigraphx_c`), HIP (`libamdhip64`) and hipRTC | `MIGraphX (... libmigraphx_c) not found; skipping aji_rocm.` |
| `aji_vk` | any Vulkan GPU | ncnn built from source with Vulkan, the Vulkan loader | `aji_vk (ncnn-Vulkan): ncnn/Vulkan not found (set -DAJI_NCNN_ROOT to your ncnn tree); skipping` |

### `aji_rocm` (AMD ROCm / MIGraphX)

```sh
cmake -B build -S . -DAJI_ROCM_ROOT=/opt/rocm      # /opt/rocm is the default
cmake --build build -j
roc-obj-ls build/libaji_rocm.so                    # expected: "No kernel section found"
```

Only a C++ compiler and the ROCm headers/libraries are needed — no HIP compiler and no gfx
arch list. The color kernels (`src/aji_rocm_color_device.hip`) are embedded as source and
JIT-compiled per GPU with hipRTC at first run into `animejanai/cache/`, which is why the `.so`
must contain zero baked device code (the `roc-obj-ls` check above). The build also produces
`aji_rocm_compile` (the out-of-process MIGraphX engine compiler; ship it next to
`libaji_rocm.so`) and the ROCm tests `aji_rocm_color_test`, `rife_harness`,
`rife_pipeline_bench` and `rocm_rife_transcode`. Runtime requirements, the JIT cache and the
color parity test are described in [`../BUILD.md`](../BUILD.md).

Quick check (run from the repo root; the test reads `test/fixtures/`):

```sh
timeout -s KILL 300 ./build/aji_rocm_color_test    # expect maxdiff=0 for NV12 and P010
```

Wrap every headless ROCm run in `timeout -s KILL` — there are known hang modes at process
exit.

### `aji_vk` (ncnn-Vulkan)

ncnn **must be built from source with Vulkan enabled** (`-DNCNN_VULKAN=ON`) **with the
`tools/ncnn_patch/` patch applied** (GridSample Vulkan layer for RIFE, the fused winograd43
convolution and the SpabTail layer the shipped `.param` files use — see `tools/ncnn_patch/README.md`); the pip wheel
and a stock package do not work here, and GPU RIFE additionally needs the `gridsample_vulkan`
layer from the ncnn tree this branch was developed against. Point CMake at the ncnn source
tree — it looks for `libncnn.so` in `<root>/build/src`, `<root>/build-fast/src` or
`<root>/lib`, headers in `<root>/src`, and the generated `ncnn_export.h` next to the library:

```sh
cmake -B build -S . -DAJI_NCNN_ROOT=$HOME/Projects/ncnn-vk    # ncnn source tree, built in build/ or build-fast/
cmake --build build -j
```

On success the configure prints `aji_vk (ncnn-Vulkan) backend: ENABLED (ncnn=...)`. The
post-build step copies the `spv/` color kernels next to `libaji_vk.so`; at runtime the library
finds them via `dladdr` (override with `AJI_VK_SPV_DIR`). `aji_vk` consumes ncnn
`.param`/`.bin` models converted from the fp16 `.onnx` files, not the `.onnx` directly
(`tools/rife_ncnn/` converts the RIFE models). There is no `build_aji_vk.sh` in this repo; the
CMake target is the only build path.

Do not bundle `libvulkan.so.1` (the loader) with a redistributable `aji_vk` — it must come from
the target system, like the GPU ICD.

## CI: `Build Linux (aji)`

```bash
gh workflow run "Build Linux (aji)" -R the-database/animejanai-inference \
  --ref main -f release_tag=v0.8.0
```

`release_tag` is **required** on dispatch and must match `AjiVersion` in the
`mpv-upscale-2x_animejanai` assembler. There is also a `push` trigger on the **`linux-support`**
branch (paths `.github/workflows/build-linux.yml`, `CMakeLists.txt`, `src/**`, `include/**`),
which falls back to `env.DEFAULT_TAG` — currently `v0.6.0`, i.e. **stale**. Dispatch from `main`
with an explicit tag rather than relying on the push path.

The job runs on `ubuntu-latest` inside `ghcr.io/the-database/animejanai-linux-build:ubuntu2204`
(GHCR login via `github.actor` + `GITHUB_TOKEN`). The toolchain comes from that image, but
**TensorRT does not**: a step fetches NVIDIA's `.deb`s (`TRT_VERSION` / `TRT_CUDA` in the
workflow env, which must match `TrtVersion` / `TrtCudaVersion` in the assembler) and unpacks
them over the throwaway container's `/usr`, replacing whatever TensorRT the image carries.

That keeps this build pinned to the exact TensorRT the package ships without waiting on a
rebuild of an image three repos share, and because the destination is still `/usr`,
`-DAJI_TRT_ROOT=/usr` is unchanged — so the RPATH baked into the shipped `libaji_trt.so`
stays a system path instead of a CI workspace path.

Only `libnvinfer.so.11` is unpacked from the ~1.9 GB `libnvinfer11` package (the rest is
per-SM builder resources, which the *package* ships but this build does not need), and each
`.deb` is deleted immediately to stay inside the runner's disk budget.

```bash
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DAJI_TRT_ROOT=/usr \
  -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
cmake --build build -j"$(nproc)"
```

Staging ships **four files only**, flat, so the assembler extracts them straight into
`animejanai/inference/`:

```bash
cp build/libaji.so build/libaji_trt.so build/aji_harness build/aji_kernel_test stage/
tar --zstd -C stage -cf aji-linux-x64.tar.zst .
```

Note what is **not** in the Linux tarball: `aji_encode` (no ffmpeg in the image) and both DML
targets (Windows-only). The Windows zip carries seven files.

The image is built by `build-image.yml` in `the-database/mpv-AnimeJaNai` from
`build/Dockerfile.ubuntu2204`, and is shared with that repo's Linux release leg and the mpv
fork's Linux bundle build — rebuilding it affects all three.

> **CUDA comes from the image, TensorRT does not.** The image's CUDA toolkit supplies `nvcc`;
> the CMake build here does not hardcode a CUDA path, so it follows whatever the image has.
> CUDA minor versions are compatible within CUDA 13, so building against the image's toolkit
> while the package ships a `cudart` from a different CUDA 13.x minor is fine — check this
> first only if a sibling build fails looking for `nvcc`.

## Parity harness (WSL)

`parity/` checks the CUDA kernels against VapourSynth/zimg goldens produced by a shipped
package's `VSPipe.exe` — the reference the native pipeline replaced.

```sh
parity/gen_fixtures.sh     # generate the canonical planar fixtures once, via ffmpeg
parity/run_parity.sh       # the sweep
```

`run_parity.sh` is explicitly a **WSL** script: Linux paths for the aji tools, Windows paths for
VSPipe. The locations come from a block of variables at the top; the three machine-specific ones
are environment-overridable, so set these rather than editing the script:

```sh
export AJI_PARITY_WIN='C:\path\to\parity'      # what VSPipe.exe (a Windows binary) sees
export AJI_PARITY_WSL=/mnt/c/path/to/parity    # the same directory from WSL
export AJI_VSPIPE=/mnt/c/<package-install>/VSPipe.exe
```

What it assumes:

| Variable | Expects |
|---|---|
| `WROOT` / `LROOT` | the same shared `parity` working directory, spelled Windows-side and WSL-side |
| `KT` | `aji_kernel_test` from a Linux `build/` of this repo |
| `PY` | a Python venv with the VapourSynth bindings |
| `CMP` | `parity/compare.py` from this repo |
| `VSPIPE` | `VSPipe.exe` from an installed 3.3.0 package — the golden reference; override with the `AJI_VSPIPE` env var |

The reference stack is VS R73 / vsmlrt 3.22.38 / TRT 10.16.0.

`compare.py` does the PSNR comparison:

```
compare.py rgbf16 OURS GOLDEN W H [--crop N] [--label S]
compare.py yuv8   OURS GOLDEN W H [--crop N] [--label S]   # ours NV12, golden planar
compare.py yuv10  OURS GOLDEN W H [--crop N] [--label S]   # ours P010, golden planar 10-bit
```

`aji_kernel_test` is the unit-level counterpart and needs no VapourSynth — prefer it when the
question is "did I break a kernel" rather than "does this match the old pipeline".

## RIFE model conversion

```
tools/convert_rife_fp16.py SRC_DIR DST_DIR [model.onnx ...]
```

No model list converts every `*.onnx` in `SRC_DIR`. Needs `numpy` and `onnx`. The output is
published under its own release tag, which the assembler pins separately as
`RifeModelsVersion` (currently `models-rife-fp16-1`).

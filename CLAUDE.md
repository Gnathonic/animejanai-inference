# animejanai-inference ("aji")

Fork of the-database/animejanai-inference: the C-ABI inference engine (`include/aji.h`) that
`vf_animejanai` in the-database's mpv fork dlopens. Goal: first-class, upstream-mergeable
Linux/AMD support. Siblings: `~/Projects/animejanai-linux` = umbrella Linux-port project
(mpv-fork patch, packaging, player testing; most past sessions on this engine ran from there);
`~/Projects/AnimeJaNaiManager` = the Avalonia config GUI.

## Branch / integration reality
- Trunk here is `linux-vulkan-backend` — the name is a MISNOMER: it is the ROCm/MIGraphX line
  (aji_rocm), which now also carries the re-integrated ncnn-Vulkan backend (aji_vk + RIFE).
- Repo is off GitHub for now; re-integration happens by porting files onto a fresh fork. Local
  commits are checkpoints — don't design around merging local branches; don't commit unasked.
- Old aji_vk worktrees archived at `~/Projects/_archive/{aji-vk-engine,aji-vk-integration}`.
  aji_vk builds against the ncnn fork at `~/Projects/ncnn-vk` (custom Vulkan GridSample layer).

## Build / test (this box: RX 9070 XT, RDNA4 gfx1201, ROCm 7.2.4, no NVIDIA)
- `cmake -B build -S . -DAJI_ROCM_ROOT=/opt/rocm && cmake --build build -j` — backends gate on
  toolchains found (TRT/CUDA skipped here). Details in BUILD.md; Vulkan build tree: `build-vk/`.
- Tests: `build/aji_rocm_color_test` (expect maxdiff=0 for NV12 and P010), `rife_harness`,
  `rocm_rife_transcode` + `tools/run_rife_transcode.sh` (ffmpeg-piped offline RIFE+upscale).
- `aji_encode` (product transcode CLI) is CUDA/NVDEC-only — will NOT run on this box; use
  `rocm_rife_transcode` for offline proof. TRT changes get runtime-verified on beast (RTX 4090).

## Architecture
- Dispatcher `src/aji_dispatch.cpp`: conf `[global] backend=` -> dlopen `libaji_<stem>.so`.
  Backends: `aji_trt` (TensorRT/CUDA) | `aji_dml` (DirectML, Windows) | `aji_rocm` (AMD, direct
  libmigraphx "Path B" — NOT ONNX Runtime; the Arch onnxruntime-rocm pkg is broken) | `aji_vk`
  (ncnn-Vulkan, vendor-neutral fallback).
- `src/aji_conf.cpp` = backend-agnostic conf/chain selection; `src/resample.h` = CPU color;
  `src/aji_rocm_color_device.hip` = HIP color kernels, hipRTC-JIT'd per GPU at runtime.
- Upscale models are fp16 .onnx (same files for all backends). RIFE conf token `rife_model=414`
  -> `rife_v4.14.onnx`. Model fixtures:
  `/home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/animejanai/{onnx,rife}/`.
- RIFE is an offline/transcode feature; real-time mpv stays upscale-only (the RIFE-first filter
  path exists but is off by default). ncnn-Vulkan is ~2.4x faster than ROCm on RIFE (warp-heavy);
  ROCm wins on conv-only upscaling.

## Gotchas (each cost real time to rediscover)
- NEVER set `MIGRAPHX_DISABLE_MLIR=1`: the MIOpen fallback reads uninitialized workspace at 4K ->
  column static / non-deterministic output. MLIR stays on; the cost is compile time only.
- MIGraphX engines are STATIC per resolution (dynamic shapes crash 2.15). First compile ~2-3 min,
  cached as `.mxr`; async build + "Building ... engine" progress text already implemented.
- Portability invariant: `libaji_rocm.so` must contain zero baked device code — verify with
  `roc-obj-ls build/libaji_rocm.so` -> "No kernel section found" (single-arch baking broke
  every non-gfx1201 GPU once).
- Benchmarks need a SINGLE GPU job — a second migraphx/mpv process corrupts eval timings.
- Hardcoded paths: `test/rife_harness.cpp` and `test/rife_pipeline_bench.cpp` hardcode the
  v0.4.3 install dir above; `tools/run_rife_transcode.sh` defaults output to `~/rife-test-deploy/`.
- mpv parses animejanai.conf only at startup (`aji_create`) — conf edits need a full mpv restart.
- Feed RIFE transcodes with `-fps_mode passthrough` on decode, or dropped frames cause judder.

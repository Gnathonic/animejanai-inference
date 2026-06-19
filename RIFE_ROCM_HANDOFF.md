# Handoff: RIFE inference on the ROCm (aji_rocm) engine — task #33

**Goal:** make the Linux/AMD ROCm backend run RIFE frame interpolation, like the Windows
TensorRT/DirectML backends already do. This is **engine code**, not a missing model — the models
are already downloaded (see fixtures below). Split from the Manager-side install tab (task #32),
which gates its RIFE UI on `aji_rife_factor() > 0`, so this work is independent and unblocks that.

## Current state — RIFE is stubbed out in aji_rocm

`src/aji_rocm.cpp`:
- `:659` `build_plan()` — `if (chain->rife) steps_log += "(RIFE requested — not yet supported in aji_rocm); "` then silently drops to upscale-only.
- `:1112` `aji_rife_factor(aji_ctx*, int*, int*) { return 0; }` — reports "no interpolation."
- `:1113` `aji_rife_before_upscale(aji_ctx*) { return 1; }` — moot stub.
- `:1128-1130` `aji_infer_rife(...) { c->err = "RIFE not supported in aji_rocm yet (P7)"; return AJI_ERR; }`

The config is already parsed — `src/aji_conf.h:21-27`: `rife`, `rife_factor_num/den`, `rife_model`
(e.g. `414` → `rife_v4.14`), `rife_ensemble`, `rife_scd_threshold` (SCDetect), `rife_before_upscale`.
So a RIFE chain reaches `build_plan()`; it's just dropped. No conf work needed.

## Reference implementation — port from aji_dml.cpp (and aji_trt.cpp)

`src/aji_dml.cpp` has the full, working RIFE path (64 rife refs). Key landmarks:
- `setup_rife()` `:1282` — builds the RIFE pipeline (mirrors aji_trt's setup_rife + stats text).
- `aji_rife_factor()` `:2034`, `aji_infer_rife()` `:2094` — the real entry points to replicate.
- Model name resolution `:1259` — `"rife_v" + s.substr(0,1) + "." + dec` from `rife_model`.
- `SCDetect` `:2253` — scene-change detection; above threshold → emit the source frame, skip interp.
- Constant channels 7..10 `:1351` — the **mesh grid + multipliers** fed alongside the two frames
  (RIFE input is the frame pair + a coordinate mesh + timestep). These become HIP kernels.
- `:1795` — the 4:4:4 / full-res RGB round-trip note (see the chroma gap below).

`src/aji_trt.cpp` has a second full reference (63 rife refs) if the DML one is unclear.

## The hard parts (in rough order)

1. **MIGraphX must compile the RIFE .onnx.** Unlike the SPAN upscalers (plain convs), RIFE graphs
   use warping / `grid_sample`-style ops and optical-flow blocks. VERIFY MIGraphX/rocMLIR compiles
   these (parse_onnx → quantize_fp16 → compile). If an op is unsupported, that's the first blocker —
   may need an op workaround or a model variant. Test compile a single `rife_v4.14.onnx` early.
2. **The 4:4:4 chroma chain.** RIFE consumes a full-res 4:4:4 / RGB input; aji_rocm's color path
   (CPU `resample.h` + the Phase-B GPU color in `aji_rocm_color.hip`) currently produces NV12/P010
   for the upscaler output. You need to feed RIFE full-res RGB and convert its output back. This is
   the chroma-path gap kernels.h flags. Reuse the Phase-B GPU color infra where possible.
3. **HIP kernels** for the mesh/const channels, warp, and SCDetect — port the DML/TRT compute
   shaders to HIP (the repo already does `enable_language(HIP)` for `aji_rocm_color.hip`).
4. **Filter side (mpv-fork).** The sw path (ROCm runs on `is_sw`) historically had no RIFE — the
   discovery noted a `!is_sw` RIFE gate in `vf_animejanai.c`. The filter already has the RIFE
   machinery (`rife_prev`, `render_interp`, `outq`, the output-grid logic in `process()`); the seek
   fix already lifted the *depth* gate for ROCm. Lift/verify the RIFE `is_sw` gate the same way.
   GREP `vf_animejanai.c` for `rife` + `is_sw` to find the exact gate.

## Test fixtures (already in place)

47 fp16 RIFE models at `/home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/animejanai/rife/`
— `rife_v4.10` … `rife_v4.26`, with `_ensemble` / `_lite` / `_heavy` variants. Naming matches what
aji_dml/aji_trt expect (`rife_v<maj>.<dec>.onnx`). Default `rife_model=414` → `rife_v4.14.onnx`.
The filter's `rife-model-dir=~~/../animejanai/rife` already points here.

Test loop: a conf chain with `chain_N_rife=yes` + a `rife_model`, play a clip, confirm fps doubling
+ visual interp quality, and that SCDetect skips interpolation across hard cuts.

## Watch out
- Start by proving MIGraphX compiles ONE rife model before building the whole path.
- Keep parity with aji_dml's SCDetect threshold + the before/after-upscale ordering (`rife_before_upscale`).
- The Phase-A async ring + the seek-drain `aji_wait` slot-free fix interact with any new infer entry
  point — RIFE submit/wait must respect the same slot bookkeeping (see [[aji-rocm-migraphx]]).

See [[linux-roadmap-rife-updater]] for the product context (RIFE planned before ship).

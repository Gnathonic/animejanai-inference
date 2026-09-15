# Design: RIFE frame interpolation on the ROCm (aji_rocm) engine — task #33

**Status:** approved design, hardened against code (spec-review pass v2), pre-implementation
**Date:** 2026-06-19
**Scope:** engine-side (`animejanai-inference/src/aji_rocm.cpp` + new kernels in `aji_rocm_color.hip`),
plus one one-line filter gate in Phase B.
**Reference brief:** `RIFE_ROCM_HANDOFF.md` (this repo).

> Line numbers below were verified against the current source by an adversarial review pass; where
> the review found drift it is corrected here. They will still drift as code changes — treat them
> as anchors to re-confirm, not constants.

## 1. Goal

Make the Linux/AMD ROCm/MIGraphX backend run RIFE frame interpolation, matching the Windows
TensorRT (`aji_trt.cpp`) and DirectML (`aji_dml.cpp`) backends. The RIFE models are already
shipped; this is engine code. The filter (`vf_animejanai.c`) is already RIFE-ready and drives
interpolation purely off the engine's reported `aji_rife_factor()`.

## 2. Feasibility — the gating blocker is cleared (verified empirically)

MIGraphX compiles RIFE. Proven end-to-end on this box with the exact `aji_rocm` recipe
(`parse_onnx → quantize_fp16 → compile(target("gpu"), offload_copy=false)`):

- `migraphx-driver compile rife_v4.14.onnx --onnx --input-dim "@input" 1 11 256 256 --fp16 --gpu`
  → **EXIT 0, ~72 s, zero CPU (`ref::`) fallbacks, zero unsupported ops**; input param `input`
  shape `{1,11,256,256}`, output `{1,3,256,256}`.
- **GridSample (28 instances)** — the feared optical-flow warp op — is fully supported:
  MIGraphX 2.15 ships `parse_gridsample`, decomposing it to `gathernd/gather + floor + clip +
  where + arith`, all GPU `code_object` kernels.
- Stack: MIGraphX 2.15.0 / ROCm 7.2.4 — the libraries `aji_rocm.cpp` already links.
- **Gotcha (satisfied by our design):** compile at a FIXED input shape. A dynamic
  `migraphx-driver read` with no `--input-dim` aborts on a Concat axis mismatch; `aji_rocm`
  always compiles at a concrete resolution.

Consequence: RIFE rides the existing `compile_mxr` + `.mxr` cache path. Compile is ~72 s per
resolution → must go through the async-build path (see §7.4) so the player does not freeze.

## 3. Corrections to the handoff brief

1. **RIFE input is 11 channels, not 7.** NCHW `{1,11,ph,pw}`, `plane = pw*ph`. `ch0-2` = frame A
   planar R,G,B; `ch3-5` = frame B planar R,G,B; `ch6` = timestep plane (per-frame); `ch7-10` =
   the four CONSTANT mesh/multiplier planes. Output `{1,3,ph,pw}` RGB. RGB (not BGR) order
   confirmed by `k_pre444`/`k_pre_combine` (`dst[idx]=R, dst[plane+idx]=G, dst[2*plane+idx]=B`).
2. **Filter: no change for Phase A; exactly one boolean for Phase B.** There is no `!is_sw` RIFE
   gate. RIFE activation is driven solely by `aji_rife_factor()`. `render_interp` treats `is_sw`
   as synchronous (`ok = is_d3d11 || is_sw`, `:820`), `alloc_out`'s `sw_pool` already maps
   `YUV444P16 → 3-plane` (`:531-535`), and the frame-builders fill `plane[0..2]`/`stride[0..2]`
   (`:641-642`, `:798-806`). The only filter touch in the task is dropping `!p->is_sw` in the
   444-output gate at `vf_animejanai.c:456` — Phase B only. **Caveat (§8.5):** that gate is inside
   the `aji_active` branch, so 444 only engages when an upscale chain is active; a RIFE-only chain
   stays at the input format.

## 4. The RIFE algorithm to replicate (from aji_trt / aji_dml)

**Input tensor** (NCHW `{1,11,ph,pw}`, fp16; `plane = pw*ph`):

| ch | content | when filled |
|----|---------|-------------|
| 0-2 | frame A planar R,G,B | per-frame |
| 3-5 | frame B planar R,G,B | per-frame |
| 6 | timestep `t ∈ (0,1)` (constant plane) | per-frame |
| 7 | `meshX = 2·x/(pw-1) − 1` | once at setup |
| 8 | `meshY = 2·y/(ph-1) − 1` | once at setup |
| 9 | `mulW = 2/(pw-1)` (constant) | once at setup |
| 10 | `mulH = 2/(ph-1)` (constant) | once at setup |

(vsmlrt `get_rife_input` v1 layout; `kernels.cu:513-525` `k_rife_consts`; the mesh dims are the
PADDED `pw/ph`, the kernel is launched at `(pw,ph)` — no off-by-one.)

**Padding:** `pw = (w+63)/64*64`, `ph = (h+63)/64*64` (mod-64). Centered, even offsets:
`pad_l = ((pw-w)/2) & ~1`, `pad_t = ((ph-h)/2) & ~1` (`&~1` rounds DOWN to even so 4:2:0 chroma
stays aligned). When staging YUV into a padded buffer (Phase B / device-resident), the chroma
plane offset uses `pad_t/2` for 4:2:0 (half-height chroma): `doff_uv = (pad_t/2)*prow + pad_l*bpp`
vs luma `doff_y = pad_t*prow + pad_l*bpp`; 444 uses `pad_t` for all planes (`aji_trt.cpp:1711-1714`).

**Border init (load-bearing):** the padded interior is overwritten per frame, but the borders must
hold studio black or GridSample warps near the edge read garbage. Studio black per format:
NV12 (8-bit) `Y=16, C=128`; P010/YUV444P16 (16-bit) `Y=16*256, C=128*256` (`aji_trt.cpp:1672-1673`).
In the Phase-A CPU path the RGB tensor's ch0-5 border region must be the BT.709 conversion of
studio black (i.e. pre-fill the whole RGB plane with black-RGB, then write the centered window),
not left uninitialized.

**Model name** — use `std::string substr` (NOT char indexing — `"rife_v" + s[0]` is pointer
arithmetic on the literal, UB):
```
s = to_string(code);
if (s.size() < 2) return "";                       // invalid → disable
dec  = (s.size()==2 ? s.substr(1,1) : s.substr(1,2));
name = "rife_v" + s.substr(0,1) + "." + dec;
if (s.size()==4 && s.back()=='1') name += "_lite";
if (ensemble)                     name += "_ensemble";
name += ".onnx";                                   // join rife_model_dir with '/' (DML used '\\')
```
Examples: `414→rife_v4.14`, `4251→rife_v4.25_lite`. `_heavy` does **not** exist in the reference
and is not expressible by the int code — out of scope. (`aji_trt.cpp:707-718`,
`aji_dml.cpp:1253-1265,1300-1303`.)

**Color (hardcoded BT.709 regardless of source matrix):** `make_csp(fmt, AJI_MATRIX_BT709, range)`
— only the matrix is forced; range follows the source.
- 4:2:0 (NV12/P010): bilinear chroma UPSAMPLE to full-res before YUV→RGB; bilinear DOWNSAMPLE
  after RGB→YUV. (Phase A. `resample.h` `compute(..., AJI_FILTER_BILINEAR)` exists at `:57-58`.)
- 4:4:4 (YUV444P16): planless — pure matrix, no resample. (Phase B.)

**Scene detect (SCDetect parity):** per-pixel `d = |Ya − Yb| · norm` over the UNPADDED luma,
read as RAW container values (no shift; P010 is `value<<6`, the norm handles the scale), summed;
decision `sum / (pw·ph) > scd_threshold → AJI_SCENE` (divisor is the PADDED area; black borders
contribute 0). `norm = 1/255` (NV12), `1/65472` (P010), `1/65535` (YUV444P16). Default
`scd_threshold = 0.150`. (`kernels.cu:594,615,620-623`, `aji_dml.cpp:2222,2255`.)

**Timestep ch6 cast:** use `(_Float16)t` (round-to-nearest-even) — matches TRT's `__float2half`.
DML hand-rolls a *truncating* float→half; the difference is ≤1 ULP and negligible for `t ∈ (0,1)`.
Pick RNE and document.

**Ordering (`rife_before_upscale`, default true):** `setup_rife` is configured with SOURCE dims
`(w,h)` when `before_upscale`, else chain-output dims `(cw,ch)`; per-frame inference code is
identical either way. `aji_rife_before_upscale()` must return `enabled && before_upscale` (the
current always-`1` stub is wrong).

**ABI contract (`include/aji.h:76-85,184-216`):**
- `aji_rife_factor(ctx, int* num, int* den)` → `0` if disabled OR the RIFE engine is not yet
  loaded (see §7.4); else `1` and writes `num`/`den`.
- `aji_rife_before_upscale(ctx)` → `1` iff `enabled && before_upscale`.
- `aji_infer_rife(ctx, a, b, t, out, stream)` → `AJI_OK` (out written) | `AJI_SCENE` (out
  untouched, caller duplicates A) | `AJI_ERR*` (codes: `AJI_OK=0, AJI_SCENE=1, AJI_ERR=-1,
  AJI_ERR_SHAPE=-2, AJI_ERR_FORMAT=-3, AJI_ERR_ENGINE=-5`). **Synchronous on return.** The
  `stream` arg is NULL on the sw path and is ignored by `aji_rocm` (it syncs on its own eval).

## 5. Current aji_rocm state (verified anchors)

- `compile_mxr()` `:266-294` — `set_input_parameter_shape("input",{1,3,(size_t)in_h,(size_t)in_w})`
  at `:277` (3-ch hardcoded) → `parse_onnx` `:278` → `quantize_fp16` `:279` → compile
  `co.set_offload_copy(false)` `:280-281` → save+atomic-rename `:284-288`.
- `load_model()` `:317-370` — per-param `hipMalloc` `:347` + `hipMemset(0)` `:355`; input by
  `name==m->in_name` (`in_name="input"`, `:321,:346`); output = a non-input `dims.size()==4`
  param `:358-360` (**last-wins; can latch a scratch param — see §7.5**); `scale=out_h/in_h` `:364`.
- `run_chain()` `:684-739` — host-staged synchronous eval: fp32→fp16 pinned `mdl_in` `:695-699`,
  H2D `:704`, `prog.eval(pp)` `:712`, `hipDeviceSynchronize` `:713`, D2H `:722`, fp16→fp32 `:728`.
- `run_chain_gpu()` `:785-852` — GPU-resident upscale: binds device params (`offload_copy=false`),
  feeds `aji_gpu_out_color(outs[0].data(), …)` `:835-837`, small YUV D2H `:843-844`,
  `hipDeviceSynchronize` `:838`. **Proves MIGraphX accepts device-pointer params bound to eval.**
- Async ring: `aji_infer` submit `:950` / `infer_worker_start` `:977`; `infer_worker_loop`
  `:855-884`; `aji_wait` `:1061` with the seek-drain lower-ticket free at `:1100-1104`.
- **Single async-build slot:** one `std::shared_ptr<BuildState> build` + one `std::thread
  build_thread` `:197-198`; `ensure_model` serializes "one build at a time" `:560-574`;
  `start_async_build` replaces the single thread `:300`; `aji_poll` reports on the single
  `c->build` `:1118-1126`. `build_plan` early-returns to passthrough (clearing models) when any
  model is uncached `:649-653` (also `:594-597`, `:623`).
- `build_plan()` `:584-674` — RIFE dropped at `:659` (`steps_log += "(RIFE requested — not yet
  supported in aji_rocm); "`).
- Stubs: `aji_rife_factor` `:1112` (→0), `aji_rife_before_upscale` `:1113` (→1, "moot"),
  `aji_infer_rife` `:1128-1132` (→`AJI_ERR`). `aji_scale_factor` `:1111`.
- `gpu_pre()` `:383-436` — CPU YUV→RGB, frame matrix + **`AJI_FILTER_SPLINE36` hardcoded**
  `:394-395`; takes `matrix` as a param. `gpu_post()` `:471-548` — CPU RGB→YUV, forced LEFT siting
  `:478`, Spline36 `:479-480`, 4:2:0 only. `aji_infer` rejects non-4:2:0 **input at `:965`, output
  at `:966`**.
- `gpu_pre`/`gpu_post` color SCRATCH (`c->scratch.t0u/…/mdl_in/mdl_out`, `:170-182`) is
  documented "one ctx processes frames sequentially" — **false once a synchronous RIFE eval runs
  on the filter thread concurrently with the upscale worker. RIFE needs its own scratch.**
- `aji_ctx` `:150` (`GpuColor gc` `:231`), `MgxModel` `:79-91`. **`aji_create` `:903-933` does NOT
  read `params->rife_model_dir`** (DML does at `aji_dml.cpp:1425`); the field exists in
  `aji_create_params` (`aji.h:99-100`).
- Conf fields parsed (`aji_conf.h:21-27`): `rife`(false), `rife_factor_num/den`(1/1),
  `rife_model`(414), `rife_ensemble`(false), `rife_scd_threshold`(0.150),
  `rife_before_upscale`(true).
- **ABI export invariant:** `aji_dispatch.cpp:162-166` hard-requires `aji_scale_factor`,
  `aji_rife_factor`, `aji_rife_before_upscale`, `aji_infer_rife` — dropping/renaming any takes down
  the ENTIRE ROCm backend, not just RIFE. The filter additionally requires `aji_rife_factor` +
  `aji_infer_rife` to load (`vf_animejanai.c:1306-1316`). All four are already exported; we replace
  bodies, keep symbols.

## 6. Concurrency

The upscale worker runs `eval` + a **device-wide** `hipDeviceSynchronize` on a SEPARATE thread on
MIGraphX's default stream (`:712-713`, `:805-806`, `:837-838`). `aji_infer_rife` runs its own eval
on the FILTER thread. Two threads issuing to the default stream + a device-wide sync race.

**Mitigation:** add a new `std::mutex gpu_eval_mtx` to `aji_ctx` and wrap **eval()+sync** at every
site — the existing worker sites in `run_chain`/`run_chain_gpu` MUST be retrofitted, not just the
RIFE one. Program/parameter/buffer state is per-`MgxModel` (not shared), so the only cross-model
hazard is the device-wide sync + shared host scratch (the latter solved by giving RIFE its own
scratch, §7.3). A per-model `hipStream` + `hipStreamSynchronize` (to avoid the RIFE sync stalling
the upscale ring) is an optional later optimization.

## 7. Phase A — RIFE working at 4:2:0 (must-have)

Engine-only. No filter change. No new HIP kernels (CPU color + CPU scene-detect, mirroring DML).

### 7.1 Plumbing
- Add `std::string rife_model_dir;` to `aji_ctx`; read `params->rife_model_dir` in `aji_create`
  (mirror `aji_dml.cpp:1425`). NULL/empty disables RIFE.

### 7.2 `RifeState` sub-struct on `aji_ctx`
- The RIFE `MgxModel` (compiled `{1,11,ph,pw}`).
- Geometry: `w,h,pw,ph,pad_l,pad_t`.
- `num,den,scd_threshold,before_upscale,enabled`, plus `loaded` (true once the `.mxr` is loaded).
- **Dedicated device buffers:** `dev_in` (`11*pw*ph*2` B), `dev_out` (`3*pw*ph*2` B). Do NOT reuse
  the worker `mdl_in`/`mdl_out`.
- **Dedicated pinned host staging:** `pin_in` (11-ch fp16), `pin_out` (3-ch fp16).
- **Dedicated color scratch** for the bilinear chroma temp planes (separate from `c->scratch`).
- A precomputed host const-template for ch7-10 (filled once).

### 7.3 Per-frame color helpers (NOT a reuse of gpu_pre)
`gpu_pre`/`gpu_post` bake in Spline36. Write dedicated `rife_pre`/`rife_post` (or parameterize
`gpu_pre/post` with a `filter` arg) that call `compute(..., AJI_FILTER_BILINEAR)` +
`make_csp(fmt, AJI_MATRIX_BT709, range)`, using RIFE's own scratch. (DML proves this — dedicated
bilinear plans at `aji_dml.cpp:2132-2135`.)

### 7.4 `setup_rife()` (from `build_plan`, replacing the `:659` drop)
1. Resolve dims: source `(w,h)` if `before_upscale` else chain-output `(cw,ch)`; compute
   `pw/ph/pad_l/pad_t`.
2. Resolve model name (§4). If `rife_model_dir` empty or name invalid (`size<2`), log + disable
   (don't hard-error the chain).
3. **Compile/load through the async-build cascade.** `compile_mxr` gains a `channels` arg
   (default 3) so RIFE requests `{1,11,ph,pw}`. The single-build slot means the RIFE `.mxr` is
   enrolled in the SAME one-build-at-a-time queue as upscale models (poll→reconfigure cascade,
   one engine per cycle). Write a RIFE-specific build marker into `c->log` (mirror
   `ensure_model:572`) so the OSD/engine_monitor shows "Building MIGraphX engine for
   rife_v4.14…" and auto-resumes. While compiling, the chain runs passthrough with **no
   interpolation**, and `loaded` stays false.
4. **Hard verification gate** once loaded: assert `get_parameter_shapes()` gives input named
   `input` `{1,11,ph,pw}` and an output `{1,3,ph,pw}`; on mismatch disable RIFE loudly (a
   name/shape mismatch silently compiles a dynamic-channel graph that aborts on Concat per §2).
5. Compute ch7-10 into the host const-template; assemble the resident layout in `dev_in`.
6. Store `RifeState`, set `enabled=true`, `loaded=true`, append the stats string to the steps log.

### 7.5 `load_model` output selection for RIFE
With `offload_copy=false` MIGraphX exposes a 4D scratch param; the current "last 4D non-input wins"
+ `out_h%in_h` heuristic can latch onto scratch. For RIFE, select the output by NAME
(`main:#output_0`) or by `channel==3`, and assert `C==3, H==ph, W==pw`. Add a RIFE-aware path in
`load_model` (or a dedicated loader).

### 7.6 ABI
Implement `aji_rife_factor` (→`1`+num/den **iff `enabled && loaded`**, else 0),
`aji_rife_before_upscale` (→`enabled && before_upscale`), `aji_infer_rife`. `aji_rife_factor` must
report independent of `c->active` (a RIFE-only chain has `active=false`; setup_rife at `:659` runs
before the active computation `:666`, which is correct).

### 7.7 `aji_infer_rife(a, b, t, out)` per-frame (synchronous)
1. Validate: `a/b/out` formats equal ∈ {NV12, P010}; dims equal `w,h`.
2. CPU scene-detect on raw unpadded luma → `sum/(pw·ph) > scd_threshold` ⇒ return `AJI_SCENE`.
3. Pre-fill `pin_in` ch0-5 region with BT.709(studio-black) so pad borders are black; CPU
   BT.709+bilinear YUV→RGB of A→ch0-2, B→ch3-5 into the centered `pw×ph` window.
4. Fill ch6 with `(_Float16)t`; copy ch7-10 from the host const-template into `pin_in`.
5. fp32→fp16 already in `pin_in`; H2D the full 11-ch `pin_in`→`dev_in` (simple baseline; the
   resident-ch7-10 + partial-H2D-of-ch0-6 optimization is available later since the layout is
   contiguous). `eval` (bind `dev_in`/`dev_out`) + `hipDeviceSynchronize` under `gpu_eval_mtx`;
   D2H `dev_out`→`pin_out`.
6. CPU BT.709+bilinear RGB→YUV → crop the centered `w×h` window into `out`.
7. Return `AJI_OK`.

**Exit criteria:** a conf chain with `chain_N_rife=yes` + a `rife_model` plays with fps doubling,
visually correct interpolation, and SCDetect skipping interpolation across hard cuts — on ROCm,
no filter change. (Validation is manual in-player; see §10.)

## 8. Phase B — 444 + GPU-resident (nice-to-have)

### 8.1 HIP kernels (mechanical CUDA→HIP from `kernels.cu` into `aji_rocm_color.hip`)
`__half→_Float16`, `cudaStream_t→hipStream_t`, atomics/`__shared__`/`__syncthreads` unchanged.
`GRID(w,h,d)` = block `(32,8)`, grid `((w+31)/32,(h+7)/8,d)` (`kernels.cu:432-436`).

| launcher | from | notes |
|----------|------|-------|
| `aji_gpu_pre444` | `k_pre444` `kernels.cu:278-316` | planar u16 YUV→RGB fp16, pure matrix |
| `aji_gpu_post444` | `k_post444` `kernels.cu:236-276` | **NEW kernel**; reuses only the `k_out_y_uvdiff` matrix EXPRESSION; writes u16 to ALL 3 planes unconditionally; hardcode `qdiv=1, qmax=65535` (do NOT read `csp.qdiv/qmax`; do NOT branch the store on `is_p010`) |
| `aji_gpu_rife_consts` | `k_rife_consts` `kernels.cu:513-535` | ch7-10, once at setup; follow with a device-wide sync so non-blocking infer streams see the consts |
| `aji_gpu_fill_f16` | `k_fill_f16` `kernels.cu:537-552` | ch6 timestep; **1-D grid, block 256** |
| `aji_gpu_fill_plane` | `k_fill_plane` `kernels.cu:554-577` | templated u8/u16 border black |
| `aji_gpu_scd_diff` | `k_scd_diff` `kernels.cu:579-626` | single-kernel reduce + `atomicAdd`; **block must be 32×8=256** (matches `__shared__ float partial[256]`); `norm` per format incl. the `format==YUV444P16 ? 1/65535 : 1/65472` u16 branch |

Reuse `aji_quant` (`aji_rocm_color.hip:18-22`), `aji_mirr` (`:13-17`), the `k_out_y_uvdiff` matrix
(`:45-59`). Launcher convention (matching `aji_gpu_out_color` `:97-111`): `extern "C"`, device
pointers as `void*`, stream as `void*`, colorspace as `aji_color_csp` **by value** (NOT
`kernels.cu`'s `aji_csp`, which lacks `qdiv/qmax/is_p010`; set `is_p010=0` for 444), `void` return
+ caller-side `hipDeviceSynchronize`; prototypes in `aji_rocm_color.h`.

### 8.2 Engine 444 output writer
Lift the **output** rejection at `aji_rocm.cpp:966` (leave the input rejection `:965`). Add a
`gpu_post444` (3 tight full-res u16 planes → `out->plane[0..2]`) as a third branch in `gpu_post`
or a separate function. The `GpuColor` scratch (`Un/Vn/hu/hv`) is sized for 4:2:0 chroma-resample;
444 needs no downsample scratch — add a 444 variant to `GpuColor`/`gpu_color_ensure`. **Cover BOTH
paths:** the `run_chain_gpu` device path AND the CPU `gpu_post` fallback (`gpu_color_eligible` is
only true for single-model no-resize, `:671-672`), or 444 silently breaks on multi-step chains.

### 8.3 Filter
Drop `!p->is_sw` at `vf_animejanai.c:456` so the sw path can set `out_fmt=AJI_FMT_YUV444P16`
(`:461`). The `out_params` bit-encoding fixup `:462-469` running on sw is benign (comment `:466`).
No API version bump: widening a format's backend support is additive (precedent: `aji.h:194`,
`rife_before_upscale` added without a bump). Do NOT bump `AJI_API_VERSION` (7) — the dispatcher's
`params->api_version != AJI_API_VERSION` check (`aji_dispatch.cpp:175`) would reject an unbumped
filter. Optionally soften the `aji.h:51` "TensorRT backend only" note for `YUV444P16`.

### 8.4 Device-resident `aji_infer_rife` 444 path (mirrors `aji_trt.cpp:1607-1815`)
Add to `RifeState` (Phase B): `in_tensor` (`11*pw*ph*2`, **consts in ch7-10 written ONCE by
`aji_gpu_rife_consts`, must persist — not a reused scratch**), `out_tensor` (`3*pw*ph*2`),
`pad_a/pad_b/pad_o` (`3*py` each), `scd` float accumulator. Per frame: `aji_gpu_fill_plane` borders
once at setup; D2D stage the 3 YUV444 planes into the padded buffers (`hipMemcpy2DAsync` D2D, using
the `pad_t`/`pad_l` offsets, §4) → `aji_gpu_pre444` into `in_tensor` ch0-5 → `aji_gpu_fill_f16`
ch6 → `eval` binding the device tensors directly (skip pinned H2D/D2H) → `aji_gpu_post444` into
`pad_o` → D2D crop into `out->plane[0..2]`. The RGB tensor never crosses PCIe. Scene-detect via
`aji_gpu_scd_diff` + a per-frame D2H readback (one `hipStreamSynchronize`) — inherent, matches TRT
and the ABI (`aji.h:208-213`).

### 8.5 444 only on an active upscale chain
The filter's 444 gate (`:456-461`) is inside the `aji_active` branch; `render_interp` builds frames
with `format=p->out_fmt`. A RIFE-ONLY chain has `out_fmt = p->aji_fmt` (the NV12/P010 input format,
`:449`), so 444 cannot engage without an active upscale chain. Document this as a known limitation
(444 RIFE requires upscale+RIFE, not RIFE-only).

## 9. Parity traps (verified — must preserve)

- BT.709 hardcoded for RIFE color regardless of source matrix — do NOT use the source matrix.
- Scene-detect divisor = PADDED area `pw·ph`; diff summed only over unpadded `w×h`; raw container
  luma values.
- 444 quant `qmax=65535, qdiv=1` (NOT P010's `qdiv=64`); `k_post444` hardcodes them.
- 444 has no chroma siting — do NOT carry `gpu_post`'s forced-LEFT siting into the 444 path.
- Timestep: `(_Float16)t` (RNE), matching TRT; DML truncates (≤1 ULP).
- Const channels 7-10 filled ONCE; only ch0-6 change per frame.

## 10. Testing — no runnable ROCm harness exists today

`harness.c`/`encode.c` are CUDA-only (`cuda_runtime.h`, `cudaMalloc`, `cudaStream_t`) and cannot
run on the AMD box; there is **no RIFE parity fixture** in-repo. So:
- **Phase A primary:** manual in-player play test (exit criteria §7) — fps doubling, interp
  quality, SCDetect on hard cuts. Test a fractional factor (e.g. 5/2, non-0.5 `t`) to exercise the
  ch6 cast at arbitrary `t`.
- **Parity (optional but recommended):** add a host-memory/ROCm-capable harness (or a
  `--backend-agnostic` variant that mallocs HOST planes — `aji_rocm` wants host pointers), OR dump
  interpolated frames from the in-player path and diff offline against a TRT/DML golden.
- **Phase B:** re-run with 444; confirm full-res chroma (no 4:2:0 round-trip loss), output matches
  the reference, and the per-frame sync does not stutter steady state. Spot-check a second variant
  (`_ensemble`) compiles.
- Build: new kernels ride the existing `hipcc` TU — `enable_language(HIP)` (`CMakeLists.txt:174`),
  `aji_rocm_color.hip` already in the source list (`:175`), `gfx1201` (`:172`). `kernels.cu` is
  CUDA-only and NOT linked into `aji_rocm`. **No CMake change needed.**

## 11. Risks & open questions

- Spot-check the default shipped `rife_v4.x.onnx` input is named `input`, 11-ch, output 3-ch
  (probe confirmed v4.14). The §7.4 hard gate catches mismatches.
- `compile_mxr`/`load_model` assume input name `input` — RIFE relies on that (probe-confirmed);
  the channel-count parameterization + by-name output selection (§7.5) cover the rest.
- Async-build ordering of upscale vs RIFE through the single build slot; the cascade + `loaded`
  gating (§7.4/§7.6) define the contract, but exercise it on a cold-cache upscale+RIFE chain.
- `_lite`/`_ensemble` variants in scope; `_heavy` is not.
- 8× factor cap is filter-side (`vf:430`); the engine need not enforce it but should not error on
  >8× (the filter silently downgrades). No engine clamp required.
- Parity oracle creation (§10) is unscoped effort if automated parity is wanted.

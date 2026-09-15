# Handoff: RIFE on ncnn-Vulkan (aji_vk) — integrate into the main project

**To:** the agent that owns the main project folder (`/home/nathan/Projects/animejanai-inference`,
trunk branch `linux-vulkan-backend` — the line that already has RIFE-on-ROCm merged).
**From:** the aji_vk (ncnn-Vulkan) line of work.
**Status:** RIFE now runs on the ncnn-Vulkan backend at parity with ROCm/TRT, fully GPU-resident,
wired end-to-end (engine + player + Manager + packaging) and **verified live**.

**Integration model — READ THIS FIRST:** the project is **off GitHub** and will be re-integrated onto a
**fresh fork** once testing is done. **Commit history will NOT survive** — local commits are only
checkpoints/rollbacks. So **do NOT plan a `git merge` of these branches.** Integrate by **porting the
changed files** (copy + apply the specific edits below) into the main tree / the new ncnn fork. The branch
names below are just *where the source currently sits*, not merge targets.

---

## 0. TL;DR — what you're integrating

1. A **new ncnn Vulkan layer, `GridSample` (gridsample_vulkan)** — ncnn never had one; RIFE's optical-flow
   warps fell back to CPU without it. Lives in the **ncnn fork** (`/home/nathan/Projects/ncnn-vk`), not this repo.
2. The **aji_vk RIFE engine implementation** (`src/aji_vk.cpp`) — `setup_rife` + `aji_infer_rife` + accessors,
   mirroring `aji_rocm`'s RifeState but executing on ncnn-Vulkan. Reuses `rife_cpu.{h,cpp}` **verbatim**.
2b. A **RIFE ONNX→ncnn conversion pipeline** (`tools/rife_ncnn/`) — needed because vsmlrt RIFE doesn't
    convert naively (see Discoveries).
3. **Packaging**: stage ncnn RIFE models + seed the conf; bundle the rebuilt libncnn.

Everything is proven on an RX 9070 XT (RADV/gfx1201). RIFE on aji_vk is actually **~2.4× faster than ROCm**
on the full RIFE+upscale pipeline (RIFE is warp-heavy → favors ncnn; opposite of the conv-only upscaler).

---

## 1. Where the source currently sits (copy FROM these)

| source location | path | files to port |
|---|---|---|
| **main tree (dest)** | `/home/nathan/Projects/animejanai-inference` | already has rife_cpu.{h,cpp}, aji_rocm RIFE, ABI (aji.h RIFE fns), harnesses |
| **aji_vk work** | `/home/nathan/Projects/aji-vk-integration` | `src/aji_vk.cpp`, `CMakeLists.txt`, `tools/rife_ncnn/` (and the committed aji_vk backend + dispatcher routing if the new fork doesn't have them yet) |
| **ncnn fork** | `/home/nathan/Projects/ncnn-vk` | `src/layer/vulkan/gridsample_vulkan.{cpp,h}` + `src/layer/vulkan/shader/gridsample.comp` → copy into whatever ncnn tree the new fork builds against |

The exact files that carry the RIFE work (just copy these / apply their edits — ignore git status, it's throwaway):
```
src/aji_vk.cpp          # the RIFE implementation (also has the committed aji_vk backend + GPU in-color)
CMakeLists.txt          # adds src/rife_cpu.cpp to the aji_vk target + -mf16c -mavx2 -mfma; defines the aji_vk target
src/aji_dispatch.cpp    # routes backend 'vulkan'/'ncnn' -> aji_vk stem (a small diff vs the RIFE-ROCm dispatcher)
tools/rife_ncnn/        # the conversion pipeline + harnesses (new dir)
```
`src/rife_cpu.{h,cpp}` are **already in the main tree** (identical copies) — don't re-port them; just make sure
the aji_vk CMake target compiles `rife_cpu.cpp`. The ncnn `gridsample_vulkan` files are NEW (ncnn has no such layer).

---

## 2. Discoveries (the WHY — don't re-derive these)

1. **vsmlrt RIFE does NOT convert to ncnn naively** — two blockers, both solved in `tools/rife_ncnn`:
   - **fp16 scalar constants crash pnnx** (`unknown constant scalar type 10` → `map::at`). Fix: cast the
     ONNX fp16→fp32 first (`fp16_to_fp32.py`); pnnx `fp16=1` re-quantizes weights afterward.
   - **The batch=2 grid_sample island.** vsmlrt stacks frame0/frame1 on the BATCH axis around `grid_sample`
     (a ~47-of-275-tensor island: the `encode` block + the whole warp pyramid run at batch=2, collapsing
     to batch=1 only at the final blend). **ncnn has no batch dim** → pnnx assumes batch=1 and silently
     drops half the data → garbage `c=1` output. Fix: `debatch.py` splits the island into two batch=1
     streams (boundary-in Reshape→two channel Slices; internal nodes duplicated `_a`/`_b` sharing weights;
     boundary-out Reshape→Concat; Split-axis-0→Identity). **Bit-exact** (99 dB vs the original ONNX via onnxruntime).
2. **ncnn had NO Vulkan GridSample** (CPU-only; verified absent locally AND upstream `git log --all`). RIFE
   has 4 warps (8 after de-batch). Without a GPU kernel they round-trip GPU→host→GPU each → not "in-GPU" +
   slow. We wrote `gridsample_vulkan` (mirrors the CPU layer exactly). Result: **0 CPU-fallback layers,
   all warps GPU-native.**
3. **RIFE model-version speed is counter-intuitive on a matrix-core GPU:** `_lite` is SLOWER than full
   (narrow channels underutilize the WMMA-16×16×16 coopmat tiles); v4.10–v4.22 FULL are architecturally
   identical (same speed); v4.26 is bigger/slower. **rife_v4.14 full = the speed+quality sweet spot**
   (8.0 ms@640×512 model-only vs lite 9.2, 4.26 9.1).
4. **ncnn `output_names()[0]` is WRONG for the de-batched RIFE net** — it exposes 5 outputs (intermediate
   flow blobs + the final RGB `out0`); index [0] is an intermediate (`c=1`). **Use the blob named `out0`.**
5. **pyncnn gotcha** (test-only): `ncnn.Mat(np_array)` wraps the numpy buffer; a temporary → freed → garbage
   1e30 output (NOT a model bug). Keep a persistent contiguous array alive across `extract()`.
6. **`_ensemble` RIFE models do NOT convert** (pnnx fails on all of them). 27 of 47 convert (all full + lite,
   v4.7–v4.26). Acceptable — ensemble is a rare refinement; the Manager dropdown lists only what converts.
7. **VFR sources** (e.g. old DivX anime): anime **OPs are native 30 fps** stored VFR (dup-frames dropped) —
   NOT telecine (that's the episodes at 24). For offline transcode, normalize **VFR→CFR at decode** before
   RIFE, output at exactly 2× that CFR, mux original audio → A/V stays synced. (Live playback in mpv handles
   timing itself.)
8. **Custom profiles are pre-seeded conf `slot_N` sections** — the conf editor (both this port AND upstream,
   it's a `// TODO`) has Add-Model/Add-Chain but NO "Add Profile" button. A normal install seeds slot_1..9;
   `assemble_vk_package.sh` wrote a 1-slot conf → only 1 custom profile showed. Fix = seed 9 slots.

---

## 3. Changes to integrate (detailed)

### 3a. ncnn fork — `gridsample_vulkan` (NEW ncnn layer)
Files (in `/home/nathan/Projects/ncnn-vk`):
- `src/layer/vulkan/gridsample_vulkan.h`, `gridsample_vulkan.cpp`
- `src/layer/vulkan/shader/gridsample.comp`
Auto-registers via the existing `ncnn_add_layer(GridSample)` (just drop the files + re-run cmake). Buffer
storage, `support_vulkan_packing=false` (elempack=1); covers bilinear/nearest/bicubic × zeros/border/reflection
× align_corner × permute_fusion, dims==3 (the image case RIFE uses). **Gotcha:** the specialization vector
must cover ALL 13 declared constants (4 modes + 9 shapes); leave the 9 shape spec-consts 0 so `psc()` uses
push constants → one pipeline for every (dynamic) resolution.
**Upstream-mergeable** — worth a PR to Tencent/ncnn when back online (it's a genuine ncnn gap).

### 3b. Engine — `src/aji_vk.cpp` (the RIFE implementation)
In `aji-vk-integration/src/aji_vk.cpp`. Mirrors `aji_rocm`'s structure (steal it as the reference):
- `RifeState` struct on `aji_ctx` (`model`, `Geom g`, num/den, scd_threshold, before_upscale, enabled,
  loaded, `assembly` 11×plane fp32 staging, `reset()`); plus `rife_model_dir` read in `aji_create` from
  `p->rife_model_dir`.
- `load_rife_model()` — like `load_model` but no 3-ch scale probe (input is 11-ch); **`out_name = "out0"`**
  (see Discovery 4).
- `setup_rife()` — mirror `aji_rocm::setup_rife` but **drop the MIGraphX async-build cascade** (ncnn loads
  instantly → only LOADED/DISABLED). Loads `<rife_model_dir>/<name>.param`, `rife_cpu::fill_consts` once.
- `build_plan` hook — replace the old "RIFE not supported" log line with `setup_rife(...)`; `rife.reset()`
  at the top of build_plan for reconfigure safety.
- `aji_infer_rife()` — `rife_cpu::scene_detect` → `yuv420_to_rgb_planes` ×2 + timestep → `ncnn::Mat(pw,ph,11)`
  per-channel memcpy → `Extractor::input/extract("out0")` → `rgb_planes_to_yuv420`. ncnn does the
  fp32→fp16 cast / H2D / GPU eval (warps via gridsample_vulkan) / D2H internally (no pinned buffers).
- Real `aji_rife_factor` / `aji_rife_before_upscale` accessors (gate on `rife.enabled && rife.loaded`).
- `CMakeLists.txt`: add `src/rife_cpu.cpp` to the `aji_vk` target + `-mf16c -mavx2 -mfma`.
- `rife_cpu.{h,cpp}` are **already in trunk** — do NOT re-add; my copies are identical. RIFE color stays CPU
  (Phase A, == ROCm shipped); only the model+warps are GPU.

### 3c. Conversion pipeline — `tools/rife_ncnn/`
`fp16_to_fp32.py`, `debatch.py` (the key IP), `rife_io.py`, `validate_parity.py`, `batch_convert.sh`
(converts all non-ensemble v4.x → .param/.bin), plus harnesses: `gs_bench.cpp` (model-only + residency check),
`rife_harness_vk.cpp` (A8/A9 gate), `vk_rife_transcode.cpp` (offline transcode; = trunk `rocm_rife_transcode`,
backend-agnostic), `README.md`. Requires local `pnnx` (in `/home/nathan/Projects/animejanai-linux/.venv/bin`)
+ onnx/onnxruntime/ncnn(pyncnn) in that venv.

### 3d. Packaging
- `package_inference.sh` already bundles the gridsample_vulkan `libncnn` (it copies whatever `libaji_vk.so`
  links from `ncnn-vk/build-fast`) — **just rebuild ncnn-vk first** so build-fast has gridsample_vulkan.
- Deployed RIFE models: a REAL `animejanai/rife/` dir with **both** the ncnn `.param/.bin` (engine loads
  these) AND the `.onnx` (Manager `RifeOnDisk`/`RifeModels` + the lua `check_components` detect via `*.onnx`).
- `assemble_vk_package.sh` TODO: it symlinks the stock `rife/` (only `.onnx`) and writes a **1-slot** conf —
  change it to (i) stage the converted `.param/.bin` alongside the `.onnx`, (ii) seed **9** conf slots
  (copy the stock conf) instead of the 1-slot heredoc.
- Slot model (FYI): built-ins `!`/`@`/`#` = slots 1001/1002/1003 (HD, NO rife); **`Ctrl+1..9` = conf
  `slot_1..9` = custom profiles where RIFE lives**; `Ctrl+0` = off. RIFE only runs on a custom profile →
  the Manager "Set as Default Profile" writes `[global] default_slot=N`.

---

## 4. Integration plan (file-level port — NO git merge)

1. **ncnn fork:** copy the 3 `gridsample_vulkan` files into the ncnn tree the new fork builds against; rebuild
   (`cmake . && ninja ncnn` in the build dir). They auto-register via the existing `ncnn_add_layer(GridSample)`.
   This is the libncnn the product links + bundles — it MUST have gridsample_vulkan or the warps fall back to CPU.
2. **Engine — copy the new file + apply edits to shared files:**
   - **Copy** `tools/rife_ncnn/` wholesale.
   - **Copy** `src/aji_vk.cpp` (it's the complete aji_vk backend incl. the RIFE impl — if the new fork already
     has an older aji_vk.cpp, replace it; otherwise this is the file). `rife_cpu.{h,cpp}` already present — leave.
   - **Apply** to `CMakeLists.txt`: add `src/rife_cpu.cpp` to the `aji_vk` target's sources and
     `-mf16c -mavx2 -mfma` to its compile options (and the whole `aji_vk` target block if absent).
   - **Apply** to `src/aji_dispatch.cpp`: route backend token `vulkan`/`ncnn` → the `aji_vk` stem (alongside the
     existing `rocm`→`aji_rocm`). These are small, localized edits — do them by hand, don't merge branches.
3. **Convert + stage models:** run `tools/rife_ncnn/batch_convert.sh` → 27 ncnn models; stage into the package
   `animejanai/rife/` (`.param/.bin` + `.onnx` symlinks).
4. **Build the package** via the (fixed) `package_inference.sh` + `assemble_vk_package.sh`.

---

## 5. Testing procedure (verify the integration — run top to bottom)

Run from `tools/rife_ncnn/` with the venv on PATH. Each step gates the next.

1. **Conversion + numerical parity** (proves the de-batch is faithful):
   ```
   PY=/home/nathan/Projects/animejanai-linux/.venv/bin/python
   $PY fp16_to_fp32.py <rife>/rife_v4.14.onnx rife_v4.14_fp32.onnx
   $PY debatch.py rife_v4.14_fp32.onnx rife_v4.14_db.onnx          # prints "checker OK"
   /home/.../.venv/bin/pnnx rife_v4.14_db.onnx inputshape=[1,11,256,256]f32 inputshape2=[1,11,512,384]f32 fp16=1
   $PY validate_parity.py
   ```
   **Expect:** ncnn-cpu vs onnx-fp32 ≈ **99 dB** (bit-exact); ncnn-vulkan vs onnx-fp16 ≈ **66–69 dB**.
   PITFALL: use realistic frames (random input is pathological); keep a persistent input buffer (Discovery 5).
2. **GPU residency + model speed** (proves gridsample_vulkan is live):
   ```
   g++ -O2 -std=c++17 gs_bench.cpp -I<ncnn>/src -I<ncnn>/build-fast/src -L<ncnn>/build-fast/src \
       -lncnn -lvulkan -fopenmp -Wl,-rpath,<ncnn>/build-fast/src -o gs_bench
   ./gs_bench rife_v4.14_db.ncnn.param rife_v4.14_db.ncnn.bin 640 512 150
   ```
   **Expect:** `CPU-fallback=0  GridSample=8(gpu=8)  model≈8 ms/frame`. If `CPU-fallback>0` → libncnn lacks
   gridsample_vulkan (rebuild ncnn-vk + re-bundle).
3. **A8/A9 engine gate** (the SAME gate ROCm passes) — build `rife_harness_vk.cpp` against `libaji_vk.so`
   (backend=vulkan, rife-model-dir → the converted models):
   **Expect:** `A8 PASS` (factor 2/1, before_upscale=1), `A9 PASS` (identity mean|diff|≤4, motion differs from
   both inputs, scene-cut → AJI_SCENE), `RIFE HARNESS: PASS`. Bench ≈ **13.9 ms/call @640×512** (vs ROCm 27.3).
4. **Offline transcode** (full RIFE+upscale): `vk_rife_transcode` on a clip → output frame count doubles
   (2N−1), output is a valid image. For VFR sources, decode with `-vf fps=<native CFR>` first (Discovery 7).
5. **Live in the player** (the real path):
   ```
   ~/AnimeJaNai-Linux-vk/mpv-upscale-2x_animejanai-vk/run-animejanai --vo=null --length=4 <clip>
   ```
   With a custom profile active (set `default_slot=N` to a rife slot, or press Ctrl+N), check
   `animejanai/currentanimejanai.log`:
   **Expect:** `slot N chain M: WxH -> 2Wx2H [model ...; RIFE rife_v4.xx 2/1 interp ... pre-upscale; ]`.
6. **Manager** (configure RIFE): launch `AnimeJaNaiManager`. With `.onnx` present in `animejanai/rife/`, the
   per-profile **RIFE toggle is enabled** and the model dropdown lists the converted versions; toggling it
   writes `chain_N_rife=yes` + `rife_model=...`. "Set as Default Profile" writes `default_slot`.

---

## 6. Known gaps / TODOs (not blocking, but track them)

- **`_ensemble` RIFE** doesn't convert (pnnx) — 27/47 models available; ensemble omitted.
- **Manager has no "Add Custom Profile" button** (upstream `// TODO`) — you can only edit the 9 pre-seeded
  conf slots. A proper fix is an Add-Profile command in `AnimeJaNaiConfEditor` (front-end work).
- **`assemble_vk_package.sh`** still writes a 1-slot conf + symlinks the stock `.onnx`-only rife dir — fix to
  seed 9 slots + stage the ncnn `.param/.bin` (§3d).
- **RIFE color is CPU** (Phase A, matches ROCm shipped). GPU-resident 4:4:4 color (Phase B) is the
  `AJI_VK_GPUINCOLOR` foundation — a later optimization, not parity-blocking.
- **`.onnx` are symlinks** to the base package in the current deployment — copy them if the base may move.
- **Upstream the ncnn gridsample_vulkan** to Tencent/ncnn when back online.

---

## 7. Receipts (measured, RX 9070 XT / RADV gfx1201)
- de-batched ONNX vs original: 99.0 dB (bit-exact). ncnn-cpu vs onnx-fp32: 99.0 dB. ncnn-vulkan vs onnx-fp16
  (TRT/ROCm ref): 68.7 dB. Generalizes across resolutions.
- gridsample_vulkan: 0 CPU-fallback, 8/8 warps GPU-native; 9× faster than ncnn-CPU.
- RIFE-alone: aji_vk 13.9 ms vs ROCm 27.3 ms @640×512 (same harness).
- Full RIFE+upscale (X OP, 640×480→1280×960, 30→60): aji_vk **1.19× real-time / 75 out-fps** vs ROCm ~0.5×.
- Live player + Manager verified on `~/AnimeJaNai-Linux-vk/mpv-upscale-2x_animejanai-vk`.

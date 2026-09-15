# ncnn fork patch (aji_vk)

`aji_vk` needs ncnn built from this tree's patch on top of upstream Tencent/ncnn
`f6f734f` (2026-06-17, "vulkan int8 kernels (#6751)"):

```sh
git clone https://github.com/Tencent/ncnn && cd ncnn && git checkout f6f734f
git am ../animejanai-inference/tools/ncnn_patch/*.patch
cmake -S . -B build-fast -G Ninja -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=ON \
      -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF        # macOS: add -DNCNN_SIMPLEVK=OFF -DVulkan_LIBRARY=/opt/homebrew/lib/libvulkan.dylib
ninja -C build-fast ncnn ncnnoptimize
```

What the patch adds (all needed at runtime by the shipped `.param` files):

- **GridSample Vulkan layer** — RIFE warps run on the GPU (`tools/rife_ncnn`).
- **Fused winograd43 convolution** (`convolution_pack4_3x3s1d1_winograd43_fused.comp`): one
  dispatch per 3x3 conv with the winograd transforms staged in shared memory instead of two
  ~150 MB VRAM intermediates per layer. 3x faster on RDNA2 (no cooperative matrix), faster than
  ncnn's coopmat path on RDNA4 and Ada too. Default on; `NCNN_WINO_FUSED=0` reverts to stock ncnn.
- **SpabTail layer** — the SPAN block tail `(res + x) * tanh(conv1x1(x))` as one pass. Model files
  are rewritten with `tools/ncnn_fuse/fuse_spab.py` (run `ncnnoptimize in.param in.bin opt.param opt.bin 0`
  first for Conv+ReLU fusion); those `.param` files load only with this ncnn.
- `NCNN_DISPATCH_THRESHOLD` env override of ncnn's mid-forward command-buffer flush (profiling).

Regenerating the patch from the fork working tree (`~/Projects/ncnn-vk`, branch `animejanai`):
`git format-patch -o tools/ncnn_patch f6f734f..animejanai`.

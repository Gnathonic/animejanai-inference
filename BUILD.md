# Build guide

## TensorRT backend (`libaji.so`, CUDA)

Needs CUDA toolkit 13.x and TensorRT 11.x (default path:
`~/sdk/tensorrt/usr`, override with `-DAJI_TRT_ROOT=`; on a system TensorRT
install from NVIDIA's apt repo, pass `-DAJI_TRT_ROOT=/usr`).

```sh
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S .
cmake --build build -j
```

The build is skipped automatically if no CUDA toolchain is found.

## ROCm/MIGraphX backend (`libaji_rocm.so`, AMD)

### Build-host requirements

- ROCm SDK including `hiprtc` development headers
  (`/opt/rocm/include/hip/hiprtc.h`; override root with `-DAJI_ROCM_ROOT=`).
- MIGraphX development headers and `libmigraphx_c`.
- A plain C++ compiler (g++); **no HIP compiler or GPU arch list is needed
  at build time.**

The color kernels (`src/aji_rocm_color_device.hip`) are embedded into
`libaji_rocm.so` as a C string at build time and JIT-compiled per-GPU with
hipRTC at runtime. The library contains zero embedded device code and is
arch-agnostic — it runs on any AMD GPU whose ROCm install is compatible.

```sh
cmake -B build -S . -DAJI_ROCM_ROOT=/opt/rocm
cmake --build build -j
# verify no device code is baked in:
roc-obj-ls build/libaji_rocm.so   # expected: "No kernel section found"
```

### Target-machine runtime requirements

The installed tarball needs these shared libraries present on the system:

| Library | Purpose |
|---|---|
| `libhiprtc.so` | JIT-compile color kernels at first run |
| `libmigraphx_c.so.3` | Model inference |
| `libamdhip64.so.7` | HIP device management |

These are part of a standard ROCm install. Bundling ROCm is out of scope.

### JIT code-object cache

On first run, hipRTC compiles the color kernels for the GPU arch reported by
`hipGetDeviceProperties().gcnArchName` and writes a code object to:

```
animejanai/cache/aji_color.<gcnArchName>.<srcHash>.co
```

`<srcHash>` is a hash of the embedded source and compile flags; a kernel
change automatically invalidates stale objects. On subsequent runs the code
object is loaded directly. If a cached object fails to load it is deleted and
recompiled (self-healing). The cache directory is created on demand.

Override the cache directory with the environment variable
`AJI_ROCM_CACHE_DIR`. The default location is resolved at runtime via
`dladdr` to `<dir-of-libaji_rocm.so>/../cache/` — the `animejanai/cache/`
directory in the installed layout.

Code objects in `cache/` are per-machine and per-GPU. They are **not**
shipped in the tarball and must not be committed to the repository.

### Color parity test

```sh
./build/aji_rocm_color_test
# expected: selftest OK, maxdiff=0 for NV12 and P010
```

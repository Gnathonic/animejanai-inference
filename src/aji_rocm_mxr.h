// Shared MIGraphX engine-compile used by BOTH the in-process sync path
// (aji_rocm.cpp, async_build=0 CLI/benchmark) AND the out-of-process helper
// (aji_rocm_compile.cpp) that the async path forks. One implementation keeps the
// compiled .mxr byte-identical across the two paths.
//
// WHY out-of-process for async: MIGraphX prog.compile() runs the MLIR backend on
// its OWN pool of parallel-compile threads IN-PROCESS. If the player quits while an
// async compile is in flight, detaching that thread leaves those compile threads
// running INTO process-exit global-destructor teardown (libmigraphx/HIP/libstdc++
// globals destroyed underneath them) -> vtable dispatch through freed state ->
// SIGSEGV / "pure virtual method called" / hang. Running the compile in a child
// process (like aji_trt's trtexec) means the player has NO in-process MIGraphX
// compile threads: quit just SIGKILLs the child and the player exits cleanly.
#pragma once
#include <migraphx/migraphx.hpp>
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

// Opt-in exhaustive kernel tuning (AJI_ROCM_EXHAUSTIVE_TUNE=1): MIGraphX
// benchmarks every candidate kernel config instead of the heuristic pick.
// Default OFF — compile time rises from ~1-2 min to many minutes per engine
// for an uncertain (0-15%) inference gain; the SPAN convs already route
// through self-tuning MLIR. Read here (not just in the compile) so the cache
// KEY changes too: a tuned engine must never silently reuse an untuned .mxr.
inline bool mxr_exhaustive_tune() {
    const char* e = getenv("AJI_ROCM_EXHAUSTIVE_TUNE");
    return e && *e && *e != '0';
}

// The .mxr cache key: <onnx>.<w>x<h>.c<channels>.dev.mlir.fp16[.exh].mxr
// The GPU the engine is compiled for ("gfx1030", "gfx1201", ...). A saved .mxr embeds
// device code for exactly one arch, so it is part of the cache key: an engine compiled on
// one AMD GPU fails to load ("Failed to call function") on another, which bites when a
// model dir moves between machines or the GPU is upgraded.
inline const std::string& mxr_gpu_arch() {
    static const std::string arch = [] {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return std::string("gfxunknown");
        std::string a = prop.gcnArchName;            // "gfx1030:sramecc+:xnack-"
        return a.substr(0, a.find(':'));
    }();
    return arch;
}
inline std::string mxr_cache_path(const std::string& onnx, int w, int h, int channels = 3) {
    return onnx + "." + std::to_string(w) + "x" + std::to_string(h)
           + ".c" + std::to_string(channels) + "." + mxr_gpu_arch() + ".dev.mlir.fp16"
           + (mxr_exhaustive_tune() ? ".exh" : "") + ".mxr";
}
inline bool mxr_cached(const std::string& onnx, int w, int h, int channels = 3) {
    std::ifstream probe(mxr_cache_path(onnx, w, h, channels), std::ios::binary);
    return probe.good();
}

// Parse + fp16-quantize + GPU-compile one .onnx at a FIXED input shape and save the
// engine. Touches NO ctx state, so it is safe on a worker thread OR in a helper
// process. Writes to a temp file then atomically renames into place, so an
// interrupted compile (player quit mid-build) never leaves a half-written .mxr that
// would later fail to load. Returns false + *errout on error.
inline bool aji_rocm_compile_mxr(const std::string& onnx_path, int in_w, int in_h,
                                 std::string* errout, int in_channels = 3) {
    try {
        // MLIR (rocMLIR) is the DEFAULT conv codegen on RDNA and is REQUIRED for correctness:
        // disabling it falls back to a MIOpen conv solver that reads uninitialized workspace at
        // 4K and yields non-deterministic, evenly-spaced column static (confirmed: MLIR-off
        // differs every run, MLIR-on is bit-identical). It costs ~70s more compile (one-time,
        // behind the progress bar) and ZERO inference fps (measured 32.4 vs 32.5ms). So leave
        // MLIR on (the 2.15 default); MIGRAPHX_DISABLE_MLIR=1 in the env still forces it off for
        // experiments. (Dynamic-shape compile is still impossible: the SPAN reflect-pad preamble
        // has non-constant pads MIGraphX can't parse dynamically.)
        migraphx::onnx_options oo;
        oo.set_input_parameter_shape("input", {1, (size_t)in_channels, (size_t)in_h, (size_t)in_w});
        auto prog = migraphx::parse_onnx(onnx_path.c_str(), oo);
        migraphx::quantize_fp16(prog);
        migraphx::compile_options co; co.set_offload_copy(false);  // device-resident
        if (mxr_exhaustive_tune())
            co.set_exhaustive_tune_flag(true);     // opt-in; see mxr_exhaustive_tune()
        prog.compile(migraphx::target("gpu"), co);
        const std::string cache = mxr_cache_path(onnx_path, in_w, in_h, in_channels);
        const std::string tmp = cache + ".tmp." + std::to_string(in_w) + "x" + std::to_string(in_h);
        migraphx::save(prog, tmp.c_str());
        if (std::rename(tmp.c_str(), cache.c_str()) != 0) {
            std::remove(tmp.c_str());
            throw std::runtime_error("could not rename engine into place");
        }
        return true;
    } catch (const std::exception& e) {
        if (errout) *errout = std::string("migraphx compile failed (") + onnx_path + "): " + e.what();
        return false;
    }
}

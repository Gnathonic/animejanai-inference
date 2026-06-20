cmake_minimum_required(VERSION 3.24)
project(animejanai-inference LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA is optional. The TensorRT backend (aji_trt) and its CUDA harness/parity tools
# build only when a CUDA toolchain is present; the ROCm/MIGraphX backend (aji_rocm)
# needs no CUDA either, so this project configures on a plain
# Linux box with just a C/C++ compiler.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Ship kernels for every supported GPU generation (Turing..Blackwell), NOT
        # "native": native builds only for the build machine's GPU, so the shipped
        # binary dies with "no kernel image is available for execution on the device"
        # on every other architecture (e.g. built on a 5090/sm120 => fails on a
        # 4090/sm89). The trailing -virtual emits PTX so future archs JIT-compile.
        # Keep in sync with the TensorRT builder packs (sm75/80/86/89/90/100/120).
        # Local dev can still override with -DCMAKE_CUDA_ARCHITECTURES=native.
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 100-real 120-real 120-virtual)
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(AJI_HAVE_CUDA ON)

    # TensorRT location (default: userland dpkg -x extract on Linux; on Windows
    # pass -DAJI_TRT_ROOT=<dir with nvinfer_10.dll, nvinfer_10.lib, include/>).
    if(WIN32)
        set(AJI_TRT_ROOT "" CACHE PATH "TensorRT root")
        set(AJI_TRT_INCLUDE "${AJI_TRT_ROOT}/include")
        set(AJI_TRT_LIB "${AJI_TRT_ROOT}")
        set(AJI_NVINFER "nvinfer_10" CACHE STRING
            "nvinfer import library name (nvinfer_10 / nvinfer_11)")
    else()
        set(AJI_TRT_ROOT "$ENV{HOME}/sdk/tensorrt/usr" CACHE PATH "TensorRT root")
        set(AJI_TRT_INCLUDE "${AJI_TRT_ROOT}/include/x86_64-linux-gnu")
        set(AJI_TRT_LIB "${AJI_TRT_ROOT}/lib/x86_64-linux-gnu")
        set(AJI_NVINFER nvinfer)
    endif()
else()
    message(STATUS "No CUDA toolchain found; skipping the TensorRT backend (aji_trt) "
                   "and CUDA tools. The ROCm/MIGraphX backend (aji_rocm) still builds.")
endif()

# aji: thin dispatcher (what the mpv filter loads as lib=). Parses the
# conf's [global] backend key and forwards the C ABI to the sibling
# backend library. No CUDA/TRT/ORT dependencies of its own.
add_library(aji SHARED src/aji_dispatch.cpp src/aji_conf.cpp)
target_include_directories(aji PUBLIC include)
target_link_libraries(aji PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(aji PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)

if(AJI_HAVE_CUDA)
# aji_trt: TensorRT/CUDA backend (the former aji.dll).
add_library(aji_trt SHARED src/aji_trt.cpp src/aji_conf.cpp src/kernels.cu)
target_include_directories(aji_trt
    PUBLIC include
    PRIVATE ${AJI_TRT_INCLUDE})
target_link_directories(aji_trt PRIVATE ${AJI_TRT_LIB})
target_link_libraries(aji_trt PRIVATE ${AJI_NVINFER} CUDA::cuda_driver CUDA::cudart)
set_target_properties(aji_trt PROPERTIES
    BUILD_RPATH "${AJI_TRT_LIB}"
    INSTALL_RPATH "${AJI_TRT_LIB}"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)
if(NOT WIN32)
    # backends export the same aji_* names as the dispatcher; bind their
    # internal references locally so the global (dispatcher) symbols
    # can't interpose
    target_link_options(aji_trt PRIVATE -Wl,-Bsymbolic)
endif()

add_executable(aji_harness src/harness.c)
target_link_libraries(aji_harness PRIVATE aji CUDA::cudart)
add_dependencies(aji_harness aji_trt)
set_target_properties(aji_harness PROPERTIES
    BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")

# aji_encode: offline file->file upscale/interpolate CLI. aji_harness's
# libaji call sequence + real libav demux/decode/encode/mux. TensorRT-only
# (v1). Built only when the ffmpeg/libav development libraries are present;
# on Windows pass -DAJI_FFMPEG_ROOT=<dir with include/ and lib/>.
if(WIN32)
    set(AJI_FFMPEG_ROOT "" CACHE PATH "ffmpeg dev root (include/, lib/)")
    if(AJI_FFMPEG_ROOT)
        add_executable(aji_encode src/encode.c src/aji_encode.manifest)
        target_include_directories(aji_encode PRIVATE "${AJI_FFMPEG_ROOT}/include")
        target_link_directories(aji_encode PRIVATE "${AJI_FFMPEG_ROOT}/lib")
        target_link_libraries(aji_encode PRIVATE aji CUDA::cudart CUDA::cuda_driver
            avformat avcodec avutil avfilter swscale)
        add_dependencies(aji_encode aji_trt)
        set_target_properties(aji_encode PROPERTIES
            BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")
    else()
        message(STATUS "AJI_FFMPEG_ROOT not set; skipping aji_encode target")
    endif()
else()
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFMPEG IMPORTED_TARGET
            libavformat libavcodec libavutil libavfilter libswscale)
    endif()
    if(FFMPEG_FOUND)
        add_executable(aji_encode src/encode.c)
        target_link_libraries(aji_encode PRIVATE
            aji CUDA::cudart CUDA::cuda_driver PkgConfig::FFMPEG)
        add_dependencies(aji_encode aji_trt)
        set_target_properties(aji_encode PROPERTIES
            BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")
    else()
        message(STATUS "ffmpeg/libav not found via pkg-config; skipping aji_encode target")
    endif()
endif()
endif()  # AJI_HAVE_CUDA

# aji_dml: DirectML backend (Windows only). Needs the ONNX Runtime
# DirectML headers (Microsoft.ML.OnnxRuntime.DirectML nupkg) and the
# DirectML header (Microsoft.AI.DirectML nupkg); onnxruntime.dll and
# DirectML.dll are loaded at runtime from the shim's own directory, so
# nothing is import-linked.
if(WIN32)
    set(AJI_ORT_ROOT "" CACHE PATH
        "Microsoft.ML.OnnxRuntime.DirectML package root")
    set(AJI_DML_ROOT "" CACHE PATH
        "Microsoft.AI.DirectML package root")
    add_library(aji_dml SHARED src/aji_dml.cpp src/aji_conf.cpp)
    target_include_directories(aji_dml
        PUBLIC include
        PRIVATE "${AJI_ORT_ROOT}/build/native/include"
                "${AJI_DML_ROOT}/include")
    target_link_libraries(aji_dml PRIVATE d3d12 d3d11 dxgi d3dcompiler)
    set_target_properties(aji_dml PROPERTIES CXX_VISIBILITY_PRESET hidden)

    add_executable(aji_harness_dml src/harness_dml.cpp)
    target_link_libraries(aji_harness_dml PRIVATE aji d3d11)
    add_dependencies(aji_harness_dml aji_dml)
endif()

if(AJI_HAVE_CUDA)
# Parity test tool: compiles the kernels directly (the shared lib hides
# the launcher symbols), no TensorRT dependency.
add_executable(aji_kernel_test src/kernel_test.c src/kernels.cu)
target_include_directories(aji_kernel_test PRIVATE include src)
target_link_libraries(aji_kernel_test PRIVATE CUDA::cudart)
endif()

# RIFE self-test harness: standalone CPU unit tests (no GPU required).
add_executable(rife_selftest src/rife_selftest.cpp src/rife_cpu.cpp)
target_include_directories(rife_selftest PRIVATE src include)
# rife_cpu.cpp pulls resample.h which uses f16c/AVX intrinsics; match aji_rocm's CXX flags.
target_compile_options(rife_selftest PRIVATE -mf16c -mavx2 -mfma)

# The ncnn-Vulkan backend (aji_vk) is retired: ~7-9x slower than MIGraphX on these
# models, so it didn't stand up as the AMD path. aji_rocm (below) replaces it; NVIDIA
# on Linux is served by aji_trt (TensorRT).
find_package(OpenMP)

# aji_rocm: AMD ROCm/MIGraphX backend (the Linux/AMD real-time path; ~7-9x faster than
# the retired ncnn-Vulkan aji_vk, TensorRT-class). Loaded by the dispatcher as
# libaji_rocm.so. Runs the SAME fp16 .onnx as aji_trt/aji_dml on MIGraphX; CPU color via
# resample.h (OpenMP). Needs MIGraphX (libmigraphx_c) + ROCm at AJI_ROCM_ROOT.
set(AJI_ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm install root (migraphx headers + libs)")
find_library(AJI_MIGRAPHX_LIB NAMES migraphx_c PATHS "${AJI_ROCM_ROOT}/lib")
find_library(AJI_HIP_LIB NAMES amdhip64 PATHS "${AJI_ROCM_ROOT}/lib")
find_package(OpenMP)
if(AJI_MIGRAPHX_LIB AND AJI_HIP_LIB AND EXISTS "${AJI_ROCM_ROOT}/include/migraphx/migraphx.hpp")
    # GPU-resident color (Phase B): aji_rocm_color_device.hip holds the YUV<->RGB +
    # chroma resample kernels; they are JIT-compiled per-GPU with hipRTC at runtime
    # (src/aji_rocm_color.cpp), so libaji_rocm.so embeds NO device code and runs on any
    # AMD arch. Embed the device source as a string for the JIT loader.
    find_library(AJI_HIPRTC_LIB NAMES hiprtc PATHS "${AJI_ROCM_ROOT}/lib")
    set(AJI_COLOR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/aji_rocm_color_device.hip)
    set(AJI_COLOR_GEN ${CMAKE_CURRENT_BINARY_DIR}/gen/aji_rocm_color_device_src.h)
    add_custom_command(
        OUTPUT ${AJI_COLOR_GEN}
        COMMAND ${CMAKE_COMMAND} -DIN=${AJI_COLOR_SRC} -DOUT=${AJI_COLOR_GEN}
                -DVAR=AJI_ROCM_COLOR_SRC -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
        DEPENDS ${AJI_COLOR_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
        COMMENT "Embedding aji_rocm_color device source")
    # rife_cpu.cpp (RIFE CPU color/scene/geometry) joins the library; the device color
    # is JIT (hipRTC), so there is no enable_language(HIP) / gfx arch list.
    add_library(aji_rocm SHARED src/aji_rocm.cpp src/aji_rocm_color.cpp src/aji_conf.cpp src/rife_cpu.cpp ${AJI_COLOR_GEN})
    target_include_directories(aji_rocm
        PUBLIC include
        PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/gen "${AJI_ROCM_ROOT}/include")
    target_compile_options(aji_rocm PRIVATE -mf16c -mavx2 -mfma)
    target_compile_definitions(aji_rocm PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(aji_rocm PRIVATE ${AJI_MIGRAPHX_LIB} ${AJI_HIP_LIB} ${AJI_HIPRTC_LIB})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aji_rocm PRIVATE OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(aji_rocm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        BUILD_RPATH "${AJI_ROCM_ROOT}/lib"
        INSTALL_RPATH "$ORIGIN:${AJI_ROCM_ROOT}/lib")
    if(NOT WIN32)
        target_link_options(aji_rocm PRIVATE -Wl,-Bsymbolic)
    endif()

    # Characterization test for the GPU out-color kernels (golden regression).
    add_executable(aji_rocm_color_test test/rocm_color_test.cpp src/aji_rocm_color.cpp ${AJI_COLOR_GEN})
    target_include_directories(aji_rocm_color_test PRIVATE include src ${CMAKE_CURRENT_BINARY_DIR}/gen "${AJI_ROCM_ROOT}/include")
    target_compile_options(aji_rocm_color_test PRIVATE -mf16c -mavx2 -mfma)
    target_compile_definitions(aji_rocm_color_test PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(aji_rocm_color_test PRIVATE ${AJI_HIP_LIB} ${AJI_HIPRTC_LIB})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aji_rocm_color_test PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # RIFE end-to-end gate (tasks A8/A9): links libaji_rocm.so, drives the engine
    # through a RIFE-only conf chain, waits for the first-run compile, and asserts
    # the engine loads, aji_rife_factor reports the configured factor, aji_infer_rife
    # interpolates a valid frame, and a hard cut returns AJI_SCENE. Host frames only
    # (aji_rocm planes are host pointers) so it needs no resample.h / f16c flags.
    add_executable(rife_harness test/rife_harness.cpp)
    target_include_directories(rife_harness PRIVATE include src)
    target_link_libraries(rife_harness PRIVATE aji_rocm)

    # Clean engine-level RIFE+upscale pipeline benchmark: one chain that BOTH
    # upscales (2x SD Compact) AND interpolates (rife_v4.26, before_upscale), driven
    # exactly like the player's filter (aji_infer_rife + aji_infer/aji_wait), to
    # characterize "RIFE 2x at 480p" output fps without mpv noise.
    add_executable(rife_pipeline_bench test/rife_pipeline_bench.cpp)
    target_include_directories(rife_pipeline_bench PRIVATE include src)
    target_link_libraries(rife_pipeline_bench PRIVATE aji_rocm)

    # rocm_rife_transcode: minimal OFFLINE RIFE+upscale transcode FILTER. Reuses
    # rife_pipeline_bench's conf/create/configure + RIFE-first flow, but reads raw
    # NV12 from stdin and writes upscaled NV12 to stdout (ffmpeg does decode+encode
    # around it) — the ROCm replacement for the CUDA-only aji_encode CLI.
    add_executable(rocm_rife_transcode test/rocm_rife_transcode.cpp)
    target_include_directories(rocm_rife_transcode PRIVATE include src)
    target_link_libraries(rocm_rife_transcode PRIVATE aji_rocm)
else()
    message(STATUS "MIGraphX (${AJI_ROCM_ROOT} libmigraphx_c) not found; skipping aji_rocm.")
endif()

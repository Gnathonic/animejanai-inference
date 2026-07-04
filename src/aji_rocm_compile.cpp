// aji_rocm_compile — out-of-process MIGraphX engine compiler.
//
// The aji_rocm backend's async_build path forks/execs this helper instead of
// compiling MIGraphX in-process, so a player quit mid-compile just SIGKILLs the
// child; the player never has in-process MIGraphX compile threads racing
// process-exit teardown. It writes the .mxr next to the .onnx (temp-then-rename,
// so a killed compile leaves no half-written engine), exactly like the in-process
// sync path — both share aji_rocm_mxr.h, so the compiled engine is byte-identical.
//
// Usage: aji_rocm_compile <onnx_path> <in_w> <in_h> <in_channels>
// Exit:  0 success (.mxr written), 1 compile error, 2 bad args.
#include "aji_rocm_mxr.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <onnx_path> <in_w> <in_h> <in_channels>\n",
                     argv[0] ? argv[0] : "aji_rocm_compile");
        return 2;
    }
    const std::string onnx = argv[1];
    const int w  = std::atoi(argv[2]);
    const int h  = std::atoi(argv[3]);
    const int ch = std::atoi(argv[4]);
    if (w <= 0 || h <= 0 || ch <= 0) {
        std::fprintf(stderr, "aji_rocm_compile: bad shape %dx%d c%d\n", w, h, ch);
        return 2;
    }
    std::string err;
    if (!aji_rocm_compile_mxr(onnx, w, h, &err, ch)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    return 0;
}

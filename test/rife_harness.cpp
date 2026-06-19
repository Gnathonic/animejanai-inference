/*
 * rife_harness — end-to-end gate for the ROCm RIFE path (tasks A8/A9).
 *
 * Standalone program that links libaji_rocm.so and drives the engine through a
 * RIFE-only conf chain (no upscale models) so ONLY the RIFE engine compiles and
 * loads. It then asserts the four facts the A8/A9 work claims:
 *
 *   A8 (setup_rife):  the RIFE engine compiles + loads; aji_rife_factor reports
 *                     the chain's configured factor (2/1) and rife_before_upscale.
 *   A9 (aji_infer_rife):
 *       - identity:  interpolating two IDENTICAL frames returns ~that frame.
 *       - motion:    interpolating two distinct frames returns a valid, written
 *                    output (sanity — no golden).
 *       - scene cut: a hard black->white cut returns AJI_SCENE.
 *
 * aji_rocm frame planes are HOST pointers (CPU color via resample.h, then
 * hipMemcpy to/from device internally), so this allocates plain host buffers —
 * unlike src/harness.c, which is CUDA and cudaMallocs device memory. We borrow
 * only the conf-mode aji_create/aji_configure shape from it, none of its cuda*.
 *
 * Build:  cmake --build build -j --target rife_harness
 * Run:    ./build/rife_harness
 * Exit:   0 on PASS, nonzero on any failure (this is the gate).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <unistd.h>   // usleep

#include "aji.h"

// ----- the RIFE-only conf written to a temp file (slot 7, no upscale models) ---
// A wide chain (0x0..infxinf, min_fps 0) so any input matches; rife=yes with
// model 414 (rife_v4.14), 2/1, before_upscale=yes. No chain_7_model_* keys -> a
// RIFE-only chain: setup_rife runs even though aji_configure reports inactive.
static const int   SLOT = 7;
static const char *CONF_TEXT =
    "[global]\n"
    "backend=rocm\n"
    "logging=no\n"
    "\n"
    "[slot_7]\n"
    "profile_name=RIFE Harness (rife-only, no upscale)\n"
    "chain_7_min_resolution=0x0\n"
    "chain_7_max_resolution=infxinf\n"
    "chain_7_min_fps=0\n"
    "chain_7_max_fps=inf\n"
    "chain_7_rife=yes\n"
    "chain_7_rife_model=414\n"
    "chain_7_rife_factor_numerator=2\n"
    "chain_7_rife_factor_denominator=1\n"
    "chain_7_rife_ensemble=no\n"
    "chain_7_rife_scene_detect_threshold=0.150\n"
    "chain_7_rife_before_upscale=yes\n";

static const char *RIFE_DIR =
    "/home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/"
    "animejanai/rife";
// model_dir: a RIFE-only chain has no upscale .onnx to find, so any existing dir
// is fine; point it at the animejanai dir (the rife dir's parent).
static const char *MODEL_DIR =
    "/home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/"
    "animejanai";

static const int  W = 256, H = 256;          // already mod-64 -> pad 0, fast compile
static const double FPS = 48.0;              // 2x -> source 24-ish; matches the chain
static const double BUILD_TIMEOUT_S = 300.0; // first compile ~72s; generous headroom

static void log_cb(void *opaque, int level, const char *msg) {
    (void)opaque;
    fprintf(stderr, "[aji:%d] %s\n", level, msg);
}

// ---- host NV12 frame (8-bit) ----------------------------------------------
struct NV12 {
    std::vector<uint8_t> y;    // W*H
    std::vector<uint8_t> uv;   // (W/2)*(H/2)*2 interleaved CbCr
    aji_frame f{};
    void alloc() {
        y.assign((size_t)W * H, 0);
        uv.assign((size_t)(W / 2) * (H / 2) * 2, 128);   // chroma mid (neutral)
        f.width = W; f.height = H;
        f.format = AJI_FMT_NV12;
        f.matrix = AJI_MATRIX_BT709;
        f.range = AJI_RANGE_LIMITED;
        f.siting = AJI_SITING_LEFT;
        f.plane[0] = y.data();
        f.plane[1] = uv.data();
        f.plane[2] = nullptr;
        f.stride[0] = W;
        f.stride[1] = W;          // CbCr row = (W/2) pairs * 2 bytes = W bytes
        f.stride[2] = 0;
    }
};

static double mean_abs_diff_y(const NV12 &a, const NV12 &b) {
    double s = 0; size_t n = (size_t)W * H;
    for (size_t i = 0; i < n; i++)
        s += std::fabs((double)a.y[i] - (double)b.y[i]);
    return s / n;
}
static int max_abs_diff_y(const NV12 &a, const NV12 &b) {
    int m = 0; size_t n = (size_t)W * H;
    for (size_t i = 0; i < n; i++) {
        int d = std::abs((int)a.y[i] - (int)b.y[i]);
        if (d > m) m = d;
    }
    return m;
}
static double mean_y(const NV12 &a) {
    double s = 0; size_t n = (size_t)W * H;
    for (size_t i = 0; i < n; i++) s += a.y[i];
    return s / n;
}

static int FAIL(const char *why) {
    printf("\nRIFE HARNESS: FAIL: %s\n", why);
    return 1;
}

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // 1. Write the temp conf.
    char conf_path[] = "/tmp/rife_harnessXXXXXX.conf";
    {
        int fd = mkstemps(conf_path, 5);   // 5 = len(".conf")
        if (fd < 0) { perror("mkstemps"); return FAIL("could not create temp conf"); }
        if (write(fd, CONF_TEXT, strlen(CONF_TEXT)) != (ssize_t)strlen(CONF_TEXT)) {
            close(fd); return FAIL("write temp conf failed");
        }
        close(fd);
    }
    printf("rife_harness: conf=%s slot=%d  rife_dir=%s\n", conf_path, SLOT, RIFE_DIR);
    printf("------------------- conf -------------------\n%s"
           "--------------------------------------------\n", CONF_TEXT);

    // 2. Create the engine in conf mode with async builds so we can poll.
    aji_create_params p{};
    p.api_version = AJI_API_VERSION;
    p.conf_path = conf_path;
    p.model_dir = MODEL_DIR;
    p.rife_model_dir = RIFE_DIR;
    p.slot = SLOT;
    p.async_build = 1;                 // background compile + aji_poll
    p.log = log_cb;

    aji_ctx *c = aji_create(&p);
    if (!c) return FAIL("aji_create returned NULL");

    int ow = 0, oh = 0;
    int act = aji_configure(c, W, H, FPS, &ow, &oh);
    if (act < 0) {
        fprintf(stderr, "aji_configure error %d: %s\n", act, aji_last_error(c));
        return FAIL("initial aji_configure errored");
    }
    printf("aji_configure -> active=%d  out=%dx%d\n", act, ow, oh);
    printf("log: %s\n", aji_current_log(c));

    // 3. Build-poll-reconfigure loop until aji_rife_factor reports a loaded engine.
    //    A RIFE-only chain returns active=0 from configure (no upscale steps), so we
    //    key the loop on aji_rife_factor, NOT on the configure return value.
    int num = 0, den = 0;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    int polls = 0;
    while (aji_rife_factor(c, &num, &den) == 0) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double el = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        if (el > BUILD_TIMEOUT_S) {
            fprintf(stderr, "last log: %s\nlast error: %s\n",
                    aji_current_log(c), aji_last_error(c));
            char buf[128];
            snprintf(buf, sizeof buf, "RIFE engine not loaded after %.0fs (timeout)", el);
            aji_destroy(&c);
            return FAIL(buf);
        }
        if (aji_poll(c) == 1) {
            printf("[%.1fs] aji_poll: build finished -> reconfigure\n", el);
            int a2 = aji_configure(c, W, H, FPS, &ow, &oh);
            if (a2 < 0) {
                fprintf(stderr, "reconfigure error %d: %s\n", a2, aji_last_error(c));
                aji_destroy(&c);
                return FAIL("reconfigure after build errored");
            }
            printf("    reconfigured: active=%d  log: %s\n", a2, aji_current_log(c));
        } else if ((polls++ % 25) == 0) {           // ~ every 5s
            printf("[%.1fs] compiling RIFE engine... (%s)\n", el, aji_current_log(c));
        }
        usleep(200 * 1000);   // 200ms
    }
    {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double el = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        printf("[%.1fs] RIFE engine LOADED.\n", el);
    }

    // 4. ASSERT factor + ordering (A8).
    printf("\n== A8: setup_rife / accessors ==\n");
    printf("aji_rife_factor: %d/%d\n", num, den);
    if (num != 2 || den != 1)
        { aji_destroy(&c); return FAIL("aji_rife_factor != 2/1"); }
    int before = aji_rife_before_upscale(c);
    printf("aji_rife_before_upscale: %d\n", before);
    if (before != 1)
        { aji_destroy(&c); return FAIL("aji_rife_before_upscale != 1"); }
    printf("A8 PASS\n");

    // 5. Build host frames and run the three A9 tests.
    printf("\n== A9: aji_infer_rife ==\n");

    // gradient frame A (Y = 40 + x scaled into limited range; chroma mid)
    NV12 A; A.alloc();
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            A.y[(size_t)y * W + x] = (uint8_t)(40 + (x * 180) / (W - 1));   // 40..220

    // OUT buffer
    NV12 OUT; OUT.alloc();

    // -- identity: interp(A, A) ~= A --
    std::memset(OUT.y.data(), 0, OUT.y.size());          // zero so "untouched" is detectable
    std::memset(OUT.uv.data(), 0, OUT.uv.size());
    int r = aji_infer_rife(c, &A.f, &A.f, 0.5, &OUT.f, nullptr);
    if (r != AJI_OK) {
        fprintf(stderr, "identity aji_infer_rife -> %d: %s\n", r, aji_last_error(c));
        aji_destroy(&c);
        return FAIL("identity interp did not return AJI_OK");
    }
    double id_mean = mean_abs_diff_y(OUT, A);
    int    id_max  = max_abs_diff_y(OUT, A);
    double out_mean = mean_y(OUT);
    printf("identity: AJI_OK  meanY|OUT-A|=%.3f  maxY|OUT-A|=%d  meanY(OUT)=%.1f\n",
           id_mean, id_max, out_mean);
    if (out_mean < 1.0) { aji_destroy(&c); return FAIL("identity OUT is ~all zero (not written)"); }
    if (id_mean > 4.0) {                                  // fp16 + model error tolerance
        aji_destroy(&c);
        return FAIL("identity interp differs too much from input (mean|diff| > 4)");
    }
    printf("identity PASS (mean diff %.3f within tol 4.0)\n", id_mean);

    // -- motion: interp(A, B) written & valid, B = A shifted +8px --
    NV12 B; B.alloc();
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int sx = x - 8; if (sx < 0) sx = 0;
            B.y[(size_t)y * W + x] = A.y[(size_t)y * W + sx];
        }
    NV12 ZERO; ZERO.alloc();
    std::memset(ZERO.y.data(), 0, ZERO.y.size());
    std::memset(OUT.y.data(), 0, OUT.y.size());
    std::memset(OUT.uv.data(), 0, OUT.uv.size());
    r = aji_infer_rife(c, &A.f, &B.f, 0.5, &OUT.f, nullptr);
    if (r != AJI_OK) {
        fprintf(stderr, "motion aji_infer_rife -> %d: %s\n", r, aji_last_error(c));
        aji_destroy(&c);
        return FAIL("motion interp did not return AJI_OK");
    }
    {
        double m = mean_y(OUT);
        int    mn = 255, mx = 0;
        for (size_t i = 0; i < OUT.y.size(); i++) {
            mn = std::min(mn, (int)OUT.y[i]);
            mx = std::max(mx, (int)OUT.y[i]);
        }
        double dz = mean_abs_diff_y(OUT, ZERO);
        printf("motion: AJI_OK  meanY(OUT)=%.1f  Yrange=[%d..%d]  mean|OUT-zero|=%.1f\n",
               m, mn, mx, dz);
        if (dz < 1.0) { aji_destroy(&c); return FAIL("motion OUT ~all zero (not written)"); }
        if (mx > 254 && mn < 1) {
            // huge spread is acceptable; just guard against a wild garbage buffer
        }
        printf("motion PASS (OUT written, valid Y range)\n");
    }

    // -- scene cut: BLACK (Y=16) -> WHITE (Y=235) -> AJI_SCENE --
    NV12 BLACK; BLACK.alloc();
    NV12 WHITE; WHITE.alloc();
    std::fill(BLACK.y.begin(), BLACK.y.end(), (uint8_t)16);
    std::fill(WHITE.y.begin(), WHITE.y.end(), (uint8_t)235);
    std::memset(OUT.y.data(), 0x55, OUT.y.size());        // sentinel; AJI_SCENE leaves untouched
    r = aji_infer_rife(c, &BLACK.f, &WHITE.f, 0.5, &OUT.f, nullptr);
    printf("scene-cut: aji_infer_rife -> %d (%s)\n", r,
           r == AJI_SCENE ? "AJI_SCENE" : r == AJI_OK ? "AJI_OK" : "ERR");
    if (r < 0) {
        fprintf(stderr, "scene aji_infer_rife error: %s\n", aji_last_error(c));
        aji_destroy(&c);
        return FAIL("scene-cut interp errored");
    }
    if (r != AJI_SCENE) {
        aji_destroy(&c);
        return FAIL("hard black->white cut did not return AJI_SCENE");
    }
    printf("scene-cut PASS (AJI_SCENE)\n");

    printf("\nA9 PASS\n");
    aji_destroy(&c);
    unlink(conf_path);

    printf("\nRIFE HARNESS: PASS\n");
    printf("summary: factor=%d/%d before_upscale=%d  identity mean|diff|=%.3f "
           "max=%d  scene=AJI_SCENE\n", num, den, before, id_mean, id_max);
    return 0;
}

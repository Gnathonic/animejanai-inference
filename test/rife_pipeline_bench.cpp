/*
 * rife_pipeline_bench — clean ENGINE-level benchmark of the full RIFE-2x-before-
 * upscale pipeline, exactly the per-output-frame work the player's filter does,
 * but with NO mpv (no testsrc decode, no --vo=null, no OOM). Characterizes
 * "RIFE 2x at 480p" reliably.
 *
 * Deployment chain mirrored (animejanai.conf slot_2 chain_2): a SINGLE chain that
 * BOTH upscales (2x SD Compact) AND interpolates (rife_v4.26, factor 2/1,
 * before_upscale=yes). 640x480 source -> 1280x960 output.
 *
 * Per INPUT PAIR (prev_src, cur_src) the player produces TWO output frames:
 *   1. aji_infer_rife(prev_src, cur_src, 0.5) -> interp_src  (640x480, source res)
 *   2. upscale interp_src -> 1280x960   (the interpolated frame)
 *   3. upscale cur_src    -> 1280x960   (the real frame)
 * i.e. 1 RIFE call + 2 upscale calls per pair = 2 output frames.
 *
 * Upscale uses the engine's normal async path (aji_infer = submit, aji_wait =
 * collect), the same submit/collect the filter drives. We measure TWO upscale
 * strategies:
 *   - PIPELINED: submit both, then wait both (mirrors the new filter's
 *     upscale_batch; the worker overlaps eval of #2 with in-color of #2 /
 *     out-color of #1).
 *   - SYNCHRONOUS: submit #1, wait #1, submit #2, wait #2 (naive one-at-a-time).
 *
 * Reports: avg RIFE-call ms, avg upscale-call ms, total ms per input pair, and
 * OUTPUT FPS (= 2 / total_per_pair). Run cold then warm; warm = steady state.
 *
 * Set AJI_RIFE_PROFILE=1 / AJI_ROCM_TIMING=1 to see the per-phase RIFE/color
 * breakdown interleaved (proves whether the upscale call is eval-bound ~8ms or
 * color-bound >>8ms).
 *
 * Build:  cmake --build build -j --target rife_pipeline_bench
 * Run:    ./build/rife_pipeline_bench
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include <unistd.h>   // usleep

#include "aji.h"

// ----- conf: ONE chain that upscales (SD Compact) AND rifes (426, 2/1, before) --
// Matches deployment slot_2 chain_2: max-res 1280x720, fps 0..31 -> 640x480@24
// selects it. A single upscale model + rife_before_upscale=yes.
static const int   SLOT = 2;
static const char *CONF_TEXT =
    "[global]\n"
    "backend=rocm\n"
    "logging=no\n"
    "default_slot=2\n"
    "\n"
    "[slot_2]\n"
    "profile_name=RIFE Pipeline Bench (SD Compact + rife426 before)\n"
    "chain_1_min_resolution=0x0\n"
    "chain_1_max_resolution=1280x720\n"
    "chain_1_min_fps=0\n"
    "chain_1_max_fps=31\n"
    "chain_1_model_1_resize_height_before_upscale=0\n"
    "chain_1_model_1_resize_factor_before_upscale=100\n"
    "chain_1_model_1_name=2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo\n"
    "chain_1_rife=yes\n"
    "chain_1_rife_model=426\n"
    "chain_1_rife_factor_numerator=2\n"
    "chain_1_rife_factor_denominator=1\n"
    "chain_1_rife_scene_detect_threshold=0.150\n"
    "chain_1_rife_ensemble=no\n"
    "chain_1_rife_before_upscale=yes\n";

static const char *RIFE_DIR =
    "/home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/"
    "animejanai/rife";
static const char *MODEL_DIR =
    "/home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/"
    "animejanai/onnx";

static const int    W = 640, H = 480;          // source; RIFE pads to 640x512, upscale -> 1280x960
static const double FPS = 24.0;                // matches chain_2 (0..31), 2x -> 48 output
static const double BUILD_TIMEOUT_S = 600.0;   // cached -> instant; headroom if a compile slips in

static void log_cb(void *opaque, int level, const char *msg) {
    (void)opaque;
    fprintf(stderr, "[aji:%d] %s\n", level, msg);
}

// ---- host NV12 frame (8-bit), W x H luma ----------------------------------
struct NV12 {
    int w, h;
    std::vector<uint8_t> y, uv;
    aji_frame f{};
    void alloc(int W_, int H_) {
        w = W_; h = H_;
        y.assign((size_t)w * h, 0);
        uv.assign((size_t)(w / 2) * (h / 2) * 2, 128);
        f.width = w; f.height = h;
        f.format = AJI_FMT_NV12;
        f.matrix = AJI_MATRIX_BT709;
        f.range = AJI_RANGE_LIMITED;
        f.siting = AJI_SITING_LEFT;
        f.plane[0] = y.data();  f.plane[1] = uv.data();  f.plane[2] = nullptr;
        f.stride[0] = w;  f.stride[1] = w;  f.stride[2] = 0;
    }
};

// Fill frame f with a moving diagonal gradient at "phase" p (so consecutive
// frames differ but stay correlated -> scene-detect stays false).
static void fill_moving(NV12 &n, int p) {
    for (int y = 0; y < n.h; y++)
        for (int x = 0; x < n.w; x++)
            n.y[(size_t)y * n.w + x] =
                (uint8_t)(40 + (((x + y + p) % (n.w)) * 180) / (n.w - 1));   // 40..220, drifts
    // mild chroma drift too
    int cw = n.w / 2, ch = n.h / 2;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            n.uv[(size_t)(y * cw + x) * 2 + 0] = (uint8_t)(120 + ((x + p) % 16));
            n.uv[(size_t)(y * cw + x) * 2 + 1] = (uint8_t)(130 + ((y + p) % 16));
        }
}

static int FAIL(const char *why) {
    printf("\nRIFE PIPELINE BENCH: FAIL: %s\n", why);
    return 1;
}

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // 1. Write the temp conf.
    char conf_path[] = "/tmp/rife_pipeline_benchXXXXXX.conf";
    {
        int fd = mkstemps(conf_path, 5);
        if (fd < 0) { perror("mkstemps"); return FAIL("temp conf"); }
        if (write(fd, CONF_TEXT, strlen(CONF_TEXT)) != (ssize_t)strlen(CONF_TEXT)) {
            close(fd); return FAIL("write temp conf"); }
        close(fd);
    }
    printf("rife_pipeline_bench: conf=%s slot=%d\n", conf_path, SLOT);
    printf("  source=%dx%d fps=%.0f  -> upscale 2x -> %dx%d ; RIFE-2x before upscale\n",
           W, H, FPS, W * 2, H * 2);
    printf("------------------- conf -------------------\n%s"
           "--------------------------------------------\n", CONF_TEXT);

    // 2. Create the engine (conf mode, async build so we can poll if a compile slips in).
    aji_create_params p{};
    p.api_version = AJI_API_VERSION;
    p.conf_path = conf_path;
    p.model_dir = MODEL_DIR;
    p.rife_model_dir = RIFE_DIR;
    p.slot = SLOT;
    p.async_build = 1;
    p.log = log_cb;

    aji_ctx *c = aji_create(&p);
    if (!c) return FAIL("aji_create returned NULL");

    int ow = 0, oh = 0;
    int act = aji_configure(c, W, H, FPS, &ow, &oh);
    if (act < 0) { fprintf(stderr, "configure err %d: %s\n", act, aji_last_error(c));
                   return FAIL("initial aji_configure errored"); }
    printf("aji_configure -> active=%d  out=%dx%d\n", act, ow, oh);

    // 3. Poll until BOTH the upscale chain is active AND RIFE is loaded. A chain with
    //    rife_before_upscale builds the upscale engine and the rife engine on background
    //    threads; we reconfigure on each aji_poll() until active==1 and rife loaded.
    int num = 0, den = 0;
    clk::time_point t0 = clk::now();
    int polls = 0;
    bool sawCompile = false;
    while (!(act == 1 && aji_rife_factor(c, &num, &den) == 1)) {
        double el = ms(t0, clk::now()) / 1000.0;
        if (el > BUILD_TIMEOUT_S) {
            fprintf(stderr, "last log: %s\nlast error: %s\n",
                    aji_current_log(c), aji_last_error(c));
            aji_destroy(&c);
            return FAIL("engine(s) not ready before timeout");
        }
        if (aji_poll(c) == 1) {
            sawCompile = true;
            printf("[%.1fs] aji_poll: a build finished -> reconfigure\n", el);
            act = aji_configure(c, W, H, FPS, &ow, &oh);
            if (act < 0) { fprintf(stderr, "reconfigure err %d: %s\n", act, aji_last_error(c));
                           aji_destroy(&c); return FAIL("reconfigure errored"); }
            printf("    reconfigured: active=%d out=%dx%d\n", act, ow, oh);
        } else if ((polls++ % 25) == 0) {
            printf("[%.1fs] waiting for engine(s)... active=%d rife_loaded=%d (%s)\n",
                   el, act, aji_rife_factor(c, &num, &den), aji_current_log(c));
        }
        usleep(200 * 1000);
    }
    printf("[%.1fs] engines READY%s.\n", ms(t0, clk::now()) / 1000.0,
           sawCompile ? "  (NOTE: a COMPILE happened — engine was NOT fully cached)"
                      : "  (all cached, no compile)");

    // 4. Confirm the chain shape matches the deployment.
    printf("\n== chain assertions ==\n");
    printf("active(upscale)=%d  out=%dx%d (expect %dx%d)\n", act, ow, oh, W * 2, H * 2);
    printf("aji_rife_factor=%d/%d  before_upscale=%d\n", num, den, aji_rife_before_upscale(c));
    if (act != 1)        { aji_destroy(&c); return FAIL("chain not active (no upscale model)"); }
    if (ow != W * 2 || oh != H * 2) { aji_destroy(&c); return FAIL("output dims != 2x source"); }
    if (num != 2 || den != 1)       { aji_destroy(&c); return FAIL("rife factor != 2/1"); }
    if (aji_rife_before_upscale(c) != 1) { aji_destroy(&c); return FAIL("rife not before_upscale"); }
    printf("chain OK: 1 upscale model (2x) + RIFE 2x before upscale.\n");

    // 5. Build a small ring of distinct moving source frames so each pair really
    //    interpolates (no scene cut) and the upscale sees fresh content.
    const int NSRC = 8;
    std::vector<NV12> src(NSRC);
    for (int i = 0; i < NSRC; i++) { src[i].alloc(W, H); fill_moving(src[i], i * 3); }
    NV12 interp; interp.alloc(W, H);                 // RIFE output (source res)
    NV12 up_interp; up_interp.alloc(W * 2, H * 2);   // upscaled interpolated frame
    NV12 up_cur;    up_cur.alloc(W * 2, H * 2);      // upscaled real frame

    // Helper: one input pair, RIFE then upscale-both. Strategy selects pipelined vs sync.
    // Accumulates the RIFE-call ms and the (summed) upscale ms (submit->wait) into refs.
    // Returns total ms for the pair. scene_skips counts AJI_SCENE (should be ~0).
    auto do_pair = [&](int prevIdx, int curIdx, bool pipelined,
                       double &rife_ms_acc, double &up_ms_acc, int &scene_skips,
                       bool &ok) -> double {
        ok = true;
        const aji_frame *prev = &src[prevIdx].f;
        const aji_frame *cur  = &src[curIdx].f;

        clk::time_point p0 = clk::now();
        // --- RIFE: interpolate at SOURCE res (640x480) ---
        clk::time_point r0 = clk::now();
        int rr = aji_infer_rife(c, prev, cur, 0.5, &interp.f, nullptr);
        clk::time_point r1 = clk::now();
        rife_ms_acc += ms(r0, r1);
        const aji_frame *interp_in = &interp.f;
        if (rr == AJI_SCENE) { scene_skips++; interp_in = prev; }   // dup prev like the filter
        else if (rr != AJI_OK) { fprintf(stderr, "rife err %d: %s\n", rr, aji_last_error(c));
                                 ok = false; return 0; }

        // --- upscale BOTH interp and cur, 640x480 -> 1280x960 ---
        clk::time_point u0 = clk::now();
        if (pipelined) {
            // submit both, then wait both: worker overlaps eval(#2) with color of the pair
            int s1 = aji_infer(c, interp_in, &up_interp.f, nullptr);
            int s2 = aji_infer(c, cur,       &up_cur.f,    nullptr);
            if (s1 != AJI_OK || s2 != AJI_OK) { fprintf(stderr, "infer submit err %d/%d: %s\n",
                                                        s1, s2, aji_last_error(c)); ok = false; return 0; }
            uint64_t tk = aji_flush(c, nullptr);   // newest ticket covers both (FIFO)
            int w = aji_wait(c, tk);
            if (w != AJI_OK) { fprintf(stderr, "wait err %d: %s\n", w, aji_last_error(c));
                               ok = false; return 0; }
        } else {
            // synchronous one-at-a-time
            int s1 = aji_infer(c, interp_in, &up_interp.f, nullptr);
            if (s1 != AJI_OK) { ok = false; return 0; }
            int w1 = aji_wait(c, aji_flush(c, nullptr));
            int s2 = aji_infer(c, cur, &up_cur.f, nullptr);
            if (s2 != AJI_OK) { ok = false; return 0; }
            int w2 = aji_wait(c, aji_flush(c, nullptr));
            if (w1 != AJI_OK || w2 != AJI_OK) { ok = false; return 0; }
        }
        clk::time_point u1 = clk::now();
        up_ms_acc += ms(u0, u1);

        return ms(p0, clk::now());
    };

    // Runs `iters` input pairs of the given strategy after `warm` warm-up pairs.
    auto run_phase = [&](const char *label, bool pipelined, int warm, int iters) {
        printf("\n== %s : %d warm-up + %d measured input pairs ==\n", label, warm, iters);
        double rife_acc = 0, up_acc = 0, total_acc = 0;
        int scene_skips = 0; bool ok = true;
        // warm-up (not timed into the averages)
        for (int i = 0; i < warm; i++) {
            double junk_r = 0, junk_u = 0; int junk_s = 0; bool wok = true;
            do_pair(i % NSRC, (i + 1) % NSRC, pipelined, junk_r, junk_u, junk_s, wok);
            if (!wok) { printf("  warm-up pair %d failed\n", i); return; }
        }
        clk::time_point all0 = clk::now();
        for (int i = 0; i < iters; i++) {
            double pair_ms = do_pair(i % NSRC, (i + 1) % NSRC, pipelined,
                                     rife_acc, up_acc, scene_skips, ok);
            if (!ok) { printf("  pair %d failed\n", i); return; }
            total_acc += pair_ms;
        }
        double wall_ms = ms(all0, clk::now());

        double avg_rife  = rife_acc / iters;
        double avg_up    = up_acc / iters;          // both upscales (the pair's upscale time)
        double avg_up1   = avg_up / 2.0;            // per single upscale call
        double avg_total = total_acc / iters;       // RIFE + 2 upscales per pair
        double out_fps   = 1000.0 * 2.0 / avg_total;        // 2 output frames per pair
        double out_fps_w = 1000.0 * (2.0 * iters) / wall_ms;// wall-clock check

        printf("  scene_skips=%d/%d\n", scene_skips, iters);
        printf("  avg RIFE call           : %.2f ms\n", avg_rife);
        printf("  avg upscale (both, /pair): %.2f ms  (%.2f ms/call x2)\n", avg_up, avg_up1);
        printf("  avg TOTAL per input pair : %.2f ms  (= RIFE + 2 upscales, 2 output frames)\n",
               avg_total);
        printf("  OUTPUT FPS               : %.1f fps  (wall-clock check: %.1f fps)\n",
               out_fps, out_fps_w);
    };

    // 6. COLD then WARM. The first measured phase (small) is "cold"; the second (large)
    //    is steady-state warm. Then the synchronous variant for comparison.
    printf("\n################ COLD (first pass) ################");
    run_phase("PIPELINED upscale (COLD)", true, 2, 20);

    printf("\n################ WARM (steady state) ################");
    run_phase("PIPELINED upscale (WARM)", true, 10, 150);

    printf("\n################ SYNC (one-at-a-time, for comparison) ################");
    run_phase("SYNCHRONOUS upscale (WARM)", false, 10, 100);

    aji_destroy(&c);
    unlink(conf_path);

    printf("\nRIFE PIPELINE BENCH: DONE\n");
    return 0;
}

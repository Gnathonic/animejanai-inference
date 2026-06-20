/*
 * rocm_rife_transcode — minimal ROCm OFFLINE RIFE+upscale transcode filter.
 *
 * A raw-frame stdin->stdout filter that drives the libaji_rocm engine (the
 * AMD/ROCm path) through the SAME conf/slot/model-dirs as rife_pipeline_bench:
 * a single chain that BOTH upscales (2x SD Compact) AND interpolates
 * (rife_v4.26, factor 2/1, rife_before_upscale=yes). 640x480 source -> 1280x960
 * output at 2x the input frame rate.
 *
 * The real transcode CLI aji_encode (src/encode.c) is CUDA/NVDEC/NVENC-only and
 * won't build on this AMD box, so this tool replaces it for ROCm: ffmpeg does
 * decode+encode around it; this filter is just engine glue.
 *
 *   ffmpeg ... -f rawvideo -pix_fmt nv12 - \
 *     | rocm_rife_transcode W H FPS \
 *     | ffmpeg -f rawvideo -pix_fmt nv12 -s OUTWxOUTH -r OUTFPS -i - ... OUT.mp4
 *
 * stdin : raw NV12 frames, each W*H*3/2 bytes (Y plane W*H, then interleaved
 *         CbCr W*H/2).
 * stdout: raw NV12 frames, each out_w*out_h*3/2 bytes, IN TIME ORDER, mirroring
 *         the player's grid (RIFE-first flow). For 2x: per new input frame we
 *         emit upscale(interp(prev,cur,0.5)) then upscale(cur). The very first
 *         frame emits just upscale(frame0).
 *
 * The frame construction (format NV12, matrix BT709, range LIMITED, siting LEFT,
 * host plane pointers) matches what rife_pipeline_bench feeds the engine and the
 * engine validated.
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>    // std::swap
#include <vector>

#include <unistd.h>

#include "aji.h"

static const int SLOT = 2;
static const char *CONF_TEXT =
    "[global]\n"
    "backend=rocm\n"
    "logging=no\n"
    "default_slot=2\n"
    "\n"
    "[slot_2]\n"
    "profile_name=RIFE Transcode (SD Compact + rife426 before)\n"
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

static const double BUILD_TIMEOUT_S = 600.0;

// stderr-only logging so stdout stays a clean rawvideo stream.
static void log_cb(void *opaque, int level, const char *msg) {
    (void)opaque;
    fprintf(stderr, "[aji:%d] %s\n", level, msg);
}

// ---- host NV12 frame (8-bit) -----------------------------------------------
struct NV12 {
    int w = 0, h = 0;
    std::vector<uint8_t> y, uv;   // y = w*h, uv = (w/2)*(h/2)*2 interleaved CbCr
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
    size_t y_bytes()  const { return (size_t)w * h; }
    size_t uv_bytes() const { return (size_t)(w / 2) * (h / 2) * 2; }
    size_t total()    const { return y_bytes() + uv_bytes(); }
};

// Read exactly n bytes from fd, looping over short reads. Returns:
//   n  -> full frame read
//   0  -> clean EOF at a frame boundary (no bytes read)
//  <0  -> partial frame at EOF (truncated input) — treated as end
static ssize_t read_full(int fd, uint8_t *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r < 0) {
            if (errno == EINTR) continue;
            perror("read(stdin)");
            return -1;
        }
        if (r == 0) {                 // EOF
            return got == 0 ? 0 : -(ssize_t)got;
        }
        got += (size_t)r;
    }
    return (ssize_t)n;
}

// Write exactly n bytes to fd, looping over short writes.
static int write_full(int fd, const uint8_t *buf, size_t n) {
    size_t put = 0;
    while (put < n) {
        ssize_t w = write(fd, buf + put, n - put);
        if (w < 0) {
            if (errno == EINTR) continue;
            perror("write(stdout)");
            return -1;
        }
        put += (size_t)w;
    }
    return 0;
}

static int FAIL(const char *why) {
    fprintf(stderr, "rocm_rife_transcode: FATAL: %s\n", why);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s W H FPS   (raw NV12 in on stdin -> raw NV12 out on stdout)\n",
                argv[0]);
        return 2;
    }
    const int    W   = atoi(argv[1]);
    const int    H   = atoi(argv[2]);
    const double FPS = atof(argv[3]);
    if (W <= 0 || H <= 0 || (W & 1) || (H & 1) || FPS <= 0)
        return FAIL("bad W/H/FPS (W,H must be positive even ints, FPS>0)");

    fprintf(stderr, "rocm_rife_transcode: in=%dx%d fps=%.3f slot=%d\n", W, H, FPS, SLOT);

    // 1. Write the temp conf (same chain as rife_pipeline_bench).
    char conf_path[] = "/tmp/rocm_rife_transcodeXXXXXX.conf";
    {
        int fd = mkstemps(conf_path, 5);
        if (fd < 0) { perror("mkstemps"); return FAIL("temp conf"); }
        size_t L = strlen(CONF_TEXT);
        if (write(fd, CONF_TEXT, L) != (ssize_t)L) { close(fd); return FAIL("write temp conf"); }
        close(fd);
    }

    // 2. Create the engine (conf mode, async build).
    aji_create_params p{};
    p.api_version = AJI_API_VERSION;
    p.conf_path = conf_path;
    p.model_dir = MODEL_DIR;
    p.rife_model_dir = RIFE_DIR;
    p.slot = SLOT;
    p.async_build = 1;
    p.log = log_cb;

    aji_ctx *c = aji_create(&p);
    if (!c) { unlink(conf_path); return FAIL("aji_create returned NULL"); }

    int ow = 0, oh = 0;
    int act = aji_configure(c, W, H, FPS, &ow, &oh);
    if (act < 0) {
        fprintf(stderr, "configure err %d: %s\n", act, aji_last_error(c));
        aji_destroy(&c); unlink(conf_path); return FAIL("initial aji_configure errored");
    }

    // 3. Poll until upscale chain active AND RIFE loaded (mirror the bench's wait).
    int num = 0, den = 0;
    int polls = 0;
    bool sawCompile = false;
    double elapsed = 0;
    while (!(act == 1 && aji_rife_factor(c, &num, &den) == 1)) {
        if (elapsed > BUILD_TIMEOUT_S) {
            fprintf(stderr, "last log: %s\nlast error: %s\n",
                    aji_current_log(c), aji_last_error(c));
            aji_destroy(&c); unlink(conf_path);
            return FAIL("engine(s) not ready before timeout");
        }
        if (aji_poll(c) == 1) {
            sawCompile = true;
            fprintf(stderr, "[%.1fs] aji_poll: a build finished -> reconfigure\n", elapsed);
            act = aji_configure(c, W, H, FPS, &ow, &oh);
            if (act < 0) {
                fprintf(stderr, "reconfigure err %d: %s\n", act, aji_last_error(c));
                aji_destroy(&c); unlink(conf_path); return FAIL("reconfigure errored");
            }
        } else if ((polls++ % 25) == 0) {
            fprintf(stderr, "[%.1fs] waiting for engine(s)... active=%d rife=%d (%s)\n",
                    elapsed, act, aji_rife_factor(c, &num, &den), aji_current_log(c));
        }
        usleep(200 * 1000);
        elapsed += 0.2;
    }

    const int rife_before = aji_rife_before_upscale(c);
    fprintf(stderr, "[%.1fs] engines READY%s. out=%dx%d  rife_factor=%d/%d before_upscale=%d\n",
            elapsed, sawCompile ? " (a COMPILE happened — not fully cached)" : " (all cached)",
            ow, oh, num, den, rife_before);

    // 4. Validate the chain shape (must match deployment / bench).
    if (act != 1)                  { aji_destroy(&c); unlink(conf_path); return FAIL("chain not active"); }
    if (ow != W * 2 || oh != H * 2){ aji_destroy(&c); unlink(conf_path); return FAIL("output dims != 2x source"); }
    if (num < 1 || den < 1)        { aji_destroy(&c); unlink(conf_path); return FAIL("rife factor invalid"); }
    if (rife_before != 1)          { aji_destroy(&c); unlink(conf_path); return FAIL("rife not before_upscale"); }

    // 5. Frame buffers. Two source slots (prev/cur, ping-ponged), one interp
    //    (source res), one upscaled output.
    NV12 srcA, srcB, interp, up;
    srcA.alloc(W, H); srcB.alloc(W, H);
    interp.alloc(W, H);
    up.alloc(ow, oh);

    const size_t in_frame_bytes  = srcA.total();        // W*H*3/2
    const size_t out_frame_bytes = up.total();          // ow*oh*3/2
    fprintf(stderr, "rocm_rife_transcode: in_frame=%zuB out_frame=%zuB  grid=%d/%d\n",
            in_frame_bytes, out_frame_bytes, num, den);

    const int IN_FD = 0, OUT_FD = 1;

    // Upscale a source-res frame `in` -> `up`, synchronously, and write `up` to stdout.
    auto upscale_and_emit = [&](const aji_frame *in) -> int {
        int s = aji_infer(c, in, &up.f, nullptr);
        if (s != AJI_OK) { fprintf(stderr, "aji_infer err %d: %s\n", s, aji_last_error(c)); return -1; }
        int w = aji_wait(c, aji_flush(c, nullptr));
        if (w != AJI_OK) { fprintf(stderr, "aji_wait err %d: %s\n", w, aji_last_error(c)); return -1; }
        return write_full(OUT_FD, up.y.data(), up.y_bytes()) == 0 &&
               write_full(OUT_FD, up.uv.data(), up.uv_bytes()) == 0 ? 0 : -1;
    };

    // 6. Stream. RIFE-first flow, walking the num/den grid per input frame.
    NV12 *prev = &srcA, *cur = &srcB;
    uint64_t in_count = 0, out_count = 0, scene_skips = 0;

    while (true) {
        NV12 *dst = (in_count == 0) ? prev : cur;
        // read Y then UV (one NV12 frame).
        ssize_t ry = read_full(IN_FD, dst->y.data(), dst->y_bytes());
        if (ry == 0) break;                              // clean EOF at boundary
        if (ry < 0) { fprintf(stderr, "truncated Y at frame %llu — stopping\n",
                              (unsigned long long)in_count); break; }
        ssize_t ru = read_full(IN_FD, dst->uv.data(), dst->uv_bytes());
        if (ru <= 0) { fprintf(stderr, "truncated UV at frame %llu — stopping\n",
                              (unsigned long long)in_count); break; }

        if (in_count == 0) {
            // First frame: just upscale it -> 1 output frame. prev already holds it.
            if (upscale_and_emit(&prev->f) != 0) { aji_destroy(&c); unlink(conf_path); return FAIL("emit frame0"); }
            out_count++;
            in_count++;
            continue;
        }

        // Subsequent frame in `cur`: walk the grid between prev and cur.
        // For each integer k in (0, num/den), the fractional time is t=k/(num/den)
        // = k*den/num in (0,1) -> an interpolated frame; then the integer point
        // (k == num/den) emits the real upscaled cur. With factor 2/1 the grid is
        // {0.5 (interp), 1.0 (cur)} = upscale(interp(prev,cur,0.5)), upscale(cur).
        // General num/den: emit (num/den - 1) interpolated frames then cur. We
        // require den==1 here for the integer grid the player uses; den>1 chains
        // aren't part of this conf, so assert it.
        const int steps = num / den;                     // den==1 for this conf
        for (int k = 1; k < steps; k++) {
            double t = (double)k / (double)steps;        // in (0,1)
            int rr = aji_infer_rife(c, &prev->f, &cur->f, t, &interp.f, nullptr);
            const aji_frame *interp_in = &interp.f;
            if (rr == AJI_SCENE) {
                // Scene change: engine contract says emit a duplicate of `a` (prev).
                scene_skips++;
                interp_in = &prev->f;
            } else if (rr != AJI_OK) {
                fprintf(stderr, "aji_infer_rife err %d: %s\n", rr, aji_last_error(c));
                aji_destroy(&c); unlink(conf_path); return FAIL("rife infer");
            }
            if (upscale_and_emit(interp_in) != 0) { aji_destroy(&c); unlink(conf_path); return FAIL("emit interp"); }
            out_count++;
        }
        // Integer point: the real cur frame, upscaled.
        if (upscale_and_emit(&cur->f) != 0) { aji_destroy(&c); unlink(conf_path); return FAIL("emit cur"); }
        out_count++;

        in_count++;
        std::swap(prev, cur);                            // cur becomes next prev
    }

    fprintf(stderr,
            "rocm_rife_transcode: DONE  in_frames=%llu out_frames=%llu scene_skips=%llu\n",
            (unsigned long long)in_count, (unsigned long long)out_count,
            (unsigned long long)scene_skips);

    aji_destroy(&c);
    unlink(conf_path);
    return 0;
}

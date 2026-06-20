/*
 * rocm_rife_transcode — ROCm OFFLINE RIFE+upscale transcode filter (CONFIG MATRIX).
 *
 * A raw-frame stdin->stdout filter that drives the libaji_rocm engine (the
 * AMD/ROCm path) through a caller-supplied conf/slot/model-dirs. Generalized
 * from the original PRE-only NV12 tool to cover the full config matrix:
 *
 *   - PRE  (rife_before_upscale=yes): RIFE-first. Interpolate the SOURCE pair
 *          at INPUT res, then upscale each result to output res.
 *   - POST (rife_before_upscale=no) : upscale-first. Upscale each input, then
 *          interpolate the UPSCALED pair at OUTPUT res.
 *   - NO-RIFE (no rife in chain)    : upscale each input 1:1 (the A/B control).
 *
 * The real transcode CLI aji_encode (src/encode.c) is CUDA/NVDEC/NVENC-only and
 * won't build on this AMD box, so this tool replaces it for ROCm: ffmpeg does
 * decode+encode around it; this filter is just engine glue.
 *
 *   ffmpeg ... -f rawvideo -pix_fmt <nv12|p010le> - \
 *     | rocm_rife_transcode --conf C --model-dir D --rife-model-dir R \
 *                           --format <nv12|p010le> [--slot N] W H FPS \
 *     | ffmpeg -f rawvideo -pix_fmt <nv12|p010le> -s OUTWxOUTH -r OUTFPS -i - ... OUT.mp4
 *
 * stdin : raw NV12 (W*H*3/2 bytes) or P010LE (W*H*3 bytes) frames.
 * stdout: raw frames in the same pixel format at out_w x out_h, IN TIME ORDER.
 *
 * Frame construction (matrix BT709, range LIMITED, siting LEFT, host plane
 * pointers) matches what rife_pipeline_bench feeds the engine.
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

static const double BUILD_TIMEOUT_S = 600.0;

// stderr-only logging so stdout stays a clean rawvideo stream.
static void log_cb(void *opaque, int level, const char *msg) {
    (void)opaque;
    fprintf(stderr, "[aji:%d] %s\n", level, msg);
}

// ---- host 4:2:0 frame (NV12 8-bit OR P010LE 10-bit-in-16) ------------------
// NV12 : Y = w*h bytes (8-bit), UV = (w/2)*(h/2)*2 bytes (interleaved CbCr).
// P010 : Y = w*h*2 bytes (16-bit LE), UV = (w/2)*(h/2)*2*2 bytes (16-bit LE
//        interleaved CbCr). Engine wants the P010 format constant with 16-bit
//        strides; the 10 bits live in the MSBs but ffmpeg's rawvideo p010le is
//        already MSB-aligned so we pass bytes through verbatim.
struct Frame {
    int w = 0, h = 0;
    int fmt = AJI_FMT_NV12;       // AJI_FMT_NV12 or AJI_FMT_P010
    int bpp = 1;                  // bytes per sample (1 for NV12, 2 for P010)
    std::vector<uint8_t> y, uv;
    aji_frame f{};
    void alloc(int W_, int H_, int format) {
        w = W_; h = H_; fmt = format;
        bpp = (fmt == AJI_FMT_P010) ? 2 : 1;
        y.assign((size_t)w * h * bpp, 0);
        // chroma neutral: 8-bit 128, 16-bit MSB-aligned ~0x8000.
        uv.assign((size_t)(w / 2) * (h / 2) * 2 * bpp, 0);
        if (bpp == 1) {
            for (auto &b : uv) b = 128;
        } else {
            uint16_t *p = (uint16_t *)uv.data();
            size_t n = uv.size() / 2;
            for (size_t i = 0; i < n; i++) p[i] = 0x8000;
        }
        f.width = w; f.height = h;
        f.format = fmt;
        f.matrix = AJI_MATRIX_BT709;
        f.range = AJI_RANGE_LIMITED;
        f.siting = AJI_SITING_LEFT;
        f.plane[0] = y.data();  f.plane[1] = uv.data();  f.plane[2] = nullptr;
        f.stride[0] = (ptrdiff_t)w * bpp;
        f.stride[1] = (ptrdiff_t)w * bpp;     // interleaved CbCr: 2 * (w/2) * bpp = w*bpp
        f.stride[2] = 0;
    }
    size_t y_bytes()  const { return (size_t)w * h * bpp; }
    size_t uv_bytes() const { return (size_t)(w / 2) * (h / 2) * 2 * bpp; }
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
    const char *conf_path = nullptr;
    const char *model_dir = nullptr;
    const char *rife_model_dir = nullptr;
    const char *format_str = nullptr;
    int slot = 2;

    // Parse options, collecting positional W H FPS.
    std::vector<const char *> pos;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--conf"))                 conf_path = (++i < argc) ? argv[i] : nullptr;
        else if (!strcmp(a, "--model-dir"))       model_dir = (++i < argc) ? argv[i] : nullptr;
        else if (!strcmp(a, "--rife-model-dir"))  rife_model_dir = (++i < argc) ? argv[i] : nullptr;
        else if (!strcmp(a, "--format"))          format_str = (++i < argc) ? argv[i] : nullptr;
        else if (!strcmp(a, "--slot"))            slot = (++i < argc) ? atoi(argv[i]) : slot;
        else pos.push_back(a);
    }

    if (!conf_path || !model_dir || !rife_model_dir || !format_str || pos.size() < 3) {
        fprintf(stderr,
                "usage: %s --conf <path> --model-dir <dir> --rife-model-dir <dir> "
                "--format <nv12|p010le> [--slot N] W H FPS\n"
                "  (raw frames in on stdin -> raw frames out on stdout)\n",
                argv[0]);
        return 2;
    }

    int format;
    if (!strcmp(format_str, "nv12"))         format = AJI_FMT_NV12;
    else if (!strcmp(format_str, "p010le"))  format = AJI_FMT_P010;
    else return FAIL("bad --format (expected nv12 or p010le)");

    const int    W   = atoi(pos[0]);
    const int    H   = atoi(pos[1]);
    const double FPS = atof(pos[2]);
    if (W <= 0 || H <= 0 || (W & 1) || (H & 1) || FPS <= 0)
        return FAIL("bad W/H/FPS (W,H must be positive even ints, FPS>0)");

    fprintf(stderr, "rocm_rife_transcode: in=%dx%d fps=%.3f slot=%d format=%s conf=%s\n",
            W, H, FPS, slot, format_str, conf_path);

    // Create the engine (conf mode, async build). --conf passed straight through.
    aji_create_params p{};
    p.api_version = AJI_API_VERSION;
    p.conf_path = conf_path;
    p.model_dir = model_dir;
    p.rife_model_dir = rife_model_dir;
    p.slot = slot;
    p.async_build = 1;
    p.log = log_cb;

    aji_ctx *c = aji_create(&p);
    if (!c) return FAIL("aji_create returned NULL");

    int ow = 0, oh = 0;
    int act = aji_configure(c, W, H, FPS, &ow, &oh);
    if (act < 0) {
        fprintf(stderr, "configure err %d: %s\n", act, aji_last_error(c));
        aji_destroy(&c); return FAIL("initial aji_configure errored");
    }

    // Poll until upscale chain active. RIFE may or may not be present; if a
    // rife factor is configured we also wait for it to load.
    int num = 0, den = 0;
    int polls = 0;
    bool sawCompile = false;
    double elapsed = 0;
    auto ready = [&]() -> bool {
        if (act != 1) return false;
        // If RIFE is configured (factor != 1) wait for it; else active is enough.
        int rn = 0, rd = 0;
        int rf = aji_rife_factor(c, &rn, &rd);
        if (rf == 1) { num = rn; den = rd; return true; }       // rife loaded
        // rf != 1 means either no-rife OR rife not yet loaded. We can't tell
        // those apart cheaply, so give the build loop a bounded grace period;
        // once a compile has settled and the chain is active with no pending
        // build, treat rf!=1 as NO-RIFE.
        return false;
    };
    bool no_rife = false;
    while (!ready()) {
        if (elapsed > BUILD_TIMEOUT_S) {
            fprintf(stderr, "last log: %s\nlast error: %s\n",
                    aji_current_log(c), aji_last_error(c));
            aji_destroy(&c);
            return FAIL("engine(s) not ready before timeout");
        }
        int pr = aji_poll(c);
        if (pr == 1) {
            sawCompile = true;
            fprintf(stderr, "[%.1fs] aji_poll: a build finished -> reconfigure\n", elapsed);
            act = aji_configure(c, W, H, FPS, &ow, &oh);
            if (act < 0) {
                fprintf(stderr, "reconfigure err %d: %s\n", act, aji_last_error(c));
                aji_destroy(&c); return FAIL("reconfigure errored");
            }
        } else if (pr == 0 && act == 1) {
            // No pending build and chain active, but rife factor still != 1:
            // this is a NO-RIFE chain. Stop waiting.
            int rn = 0, rd = 0;
            if (aji_rife_factor(c, &rn, &rd) != 1) { no_rife = true; break; }
        } else if ((polls++ % 25) == 0) {
            fprintf(stderr, "[%.1fs] waiting for engine(s)... active=%d (%s)\n",
                    elapsed, act, aji_current_log(c));
        }
        usleep(200 * 1000);
        elapsed += 0.2;
    }

    const int rife_before = no_rife ? -1 : aji_rife_before_upscale(c);
    const char *mode = no_rife ? "NO-RIFE" : (rife_before == 1 ? "PRE" : "POST");
    fprintf(stderr,
            "[%.1fs] engines READY%s. out=%dx%d  rife_factor=%d/%d before_upscale=%d MODE=%s\n",
            elapsed, sawCompile ? " (a COMPILE happened — not fully cached)" : " (all cached)",
            ow, oh, num, den, rife_before, mode);

    // Validate chain shape: active, integer scale derived from configure.
    if (act != 1) { aji_destroy(&c); return FAIL("chain not active"); }
    if (ow <= 0 || oh <= 0 || (ow % W) != 0 || (oh % H) != 0) {
        aji_destroy(&c); return FAIL("output dims not a positive integer multiple of source");
    }
    if (!no_rife && (num < 1 || den < 1)) { aji_destroy(&c); return FAIL("rife factor invalid"); }
    // Integer factors only: the step loop below uses steps = num/den, so a rational
    // factor (e.g. 5/2) would silently truncate to a 2x stream with the wrong frame
    // count and an fps that no longer matches the downstream -r. Fail loudly, matching
    // the reference offline encoder (src/encode.c rejects rden != 1 || rnum % rden != 0).
    if (!no_rife && (den != 1 || num % den != 0)) {
        aji_destroy(&c);
        return FAIL("rational RIFE factor (den != 1) not supported; integer factors only");
    }

    const int IN_FD = 0, OUT_FD = 1;

    // Write a frame (Y then UV) to stdout.
    auto emit_frame = [&](const Frame &fr) -> int {
        return write_full(OUT_FD, fr.y.data(), fr.y_bytes()) == 0 &&
               write_full(OUT_FD, fr.uv.data(), fr.uv_bytes()) == 0 ? 0 : -1;
    };

    // Upscale `in` -> `dst` synchronously.
    auto upscale = [&](const aji_frame *in, Frame &dst) -> int {
        int s = aji_infer(c, in, &dst.f, nullptr);
        if (s != AJI_OK) { fprintf(stderr, "aji_infer err %d: %s\n", s, aji_last_error(c)); return -1; }
        int w = aji_wait(c, aji_flush(c, nullptr));
        if (w != AJI_OK) { fprintf(stderr, "aji_wait err %d: %s\n", w, aji_last_error(c)); return -1; }
        return 0;
    };

    // Read one input frame into `dst`. Returns 1 read, 0 clean EOF, -1 truncated.
    auto read_frame = [&](Frame &dst, uint64_t idx) -> int {
        ssize_t ry = read_full(IN_FD, dst.y.data(), dst.y_bytes());
        if (ry == 0) return 0;
        if (ry < 0) { fprintf(stderr, "truncated Y at frame %llu — stopping\n",
                              (unsigned long long)idx); return -1; }
        ssize_t ru = read_full(IN_FD, dst.uv.data(), dst.uv_bytes());
        if (ru <= 0) { fprintf(stderr, "truncated UV at frame %llu — stopping\n",
                              (unsigned long long)idx); return -1; }
        return 1;
    };

    uint64_t in_count = 0, out_count = 0, scene_skips = 0;

    if (no_rife) {
        // ---- NO-RIFE: upscale each input 1:1 -----------------------------
        Frame src, up;
        src.alloc(W, H, format);
        up.alloc(ow, oh, format);
        fprintf(stderr, "rocm_rife_transcode: in_frame=%zuB out_frame=%zuB (NO-RIFE 1:1)\n",
                src.total(), up.total());
        while (true) {
            int r = read_frame(src, in_count);
            if (r == 0) break;
            if (r < 0) break;
            if (upscale(&src.f, up) != 0) { aji_destroy(&c); return FAIL("upscale (no-rife)"); }
            if (emit_frame(up) != 0)      { aji_destroy(&c); return FAIL("emit (no-rife)"); }
            in_count++; out_count++;
        }
    } else if (rife_before == 1) {
        // ---- PRE: RIFE-first. Interp SOURCE pair, upscale each result. ----
        Frame srcA, srcB, interp, up;
        srcA.alloc(W, H, format); srcB.alloc(W, H, format);
        interp.alloc(W, H, format);
        up.alloc(ow, oh, format);
        fprintf(stderr, "rocm_rife_transcode: in_frame=%zuB out_frame=%zuB grid=%d/%d (PRE)\n",
                srcA.total(), up.total(), num, den);

        auto upscale_and_emit = [&](const aji_frame *in) -> int {
            if (upscale(in, up) != 0) return -1;
            return emit_frame(up);
        };

        Frame *prev = &srcA, *cur = &srcB;
        const int steps = num / den;
        while (true) {
            Frame *dst = (in_count == 0) ? prev : cur;
            int r = read_frame(*dst, in_count);
            if (r == 0) break;
            if (r < 0) break;

            if (in_count == 0) {
                if (upscale_and_emit(&prev->f) != 0) { aji_destroy(&c); return FAIL("emit frame0"); }
                out_count++; in_count++;
                continue;
            }
            for (int k = 1; k < steps; k++) {
                double t = (double)k / (double)steps;
                int rr = aji_infer_rife(c, &prev->f, &cur->f, t, &interp.f, nullptr);
                const aji_frame *interp_in = &interp.f;
                if (rr == AJI_SCENE) { scene_skips++; interp_in = &prev->f; }
                else if (rr != AJI_OK) {
                    fprintf(stderr, "aji_infer_rife err %d: %s\n", rr, aji_last_error(c));
                    aji_destroy(&c); return FAIL("rife infer (PRE)");
                }
                if (upscale_and_emit(interp_in) != 0) { aji_destroy(&c); return FAIL("emit interp (PRE)"); }
                out_count++;
            }
            if (upscale_and_emit(&cur->f) != 0) { aji_destroy(&c); return FAIL("emit cur (PRE)"); }
            out_count++;
            in_count++;
            std::swap(prev, cur);
        }
    } else {
        // ---- POST: upscale-first. Interp the UPSCALED pair at OUTPUT res. --
        Frame src, upA, upB, interp;
        src.alloc(W, H, format);
        upA.alloc(ow, oh, format); upB.alloc(ow, oh, format);
        interp.alloc(ow, oh, format);
        fprintf(stderr, "rocm_rife_transcode: in_frame=%zuB out_frame=%zuB grid=%d/%d (POST)\n",
                src.total(), upA.total(), num, den);

        Frame *up_prev = &upA, *up_cur = &upB;
        const int steps = num / den;
        while (true) {
            int r = read_frame(src, in_count);
            if (r == 0) break;
            if (r < 0) break;

            if (in_count == 0) {
                // first input -> upscale -> emit -> keep as up_prev.
                if (upscale(&src.f, *up_prev) != 0) { aji_destroy(&c); return FAIL("upscale frame0 (POST)"); }
                if (emit_frame(*up_prev) != 0)      { aji_destroy(&c); return FAIL("emit frame0 (POST)"); }
                out_count++; in_count++;
                continue;
            }
            // later input -> up_cur = upscale(f).
            if (upscale(&src.f, *up_cur) != 0) { aji_destroy(&c); return FAIL("upscale cur (POST)"); }
            for (int k = 1; k < steps; k++) {
                double t = (double)k / (double)steps;
                int rr = aji_infer_rife(c, &up_prev->f, &up_cur->f, t, &interp.f, nullptr);
                const Frame *interp_in = &interp;
                if (rr == AJI_SCENE) { scene_skips++; interp_in = up_prev; }   // emit dup of up_prev
                else if (rr != AJI_OK) {
                    fprintf(stderr, "aji_infer_rife err %d: %s\n", rr, aji_last_error(c));
                    aji_destroy(&c); return FAIL("rife infer (POST)");
                }
                if (emit_frame(*interp_in) != 0) { aji_destroy(&c); return FAIL("emit interp (POST)"); }
                out_count++;
            }
            // emit the real upscaled cur.
            if (emit_frame(*up_cur) != 0) { aji_destroy(&c); return FAIL("emit cur (POST)"); }
            out_count++;
            in_count++;
            std::swap(up_prev, up_cur);
        }
    }

    fprintf(stderr, "rocm_rife_transcode: MODE=%s\n", mode);
    fprintf(stderr,
            "rocm_rife_transcode: DONE  in_frames=%llu out_frames=%llu scene_skips=%llu\n",
            (unsigned long long)in_count, (unsigned long long)out_count,
            (unsigned long long)scene_skips);

    aji_destroy(&c);
    return 0;
}

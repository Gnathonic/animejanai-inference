/*
 * aji_host_harness — backend-agnostic CLI harness for the host-plane (software) path.
 *
 * Unlike src/harness.c (CUDA device frames, TensorRT), this needs no CUDA: it dlopens
 * the dispatcher (libaji.so / .dylib / aji.dll), lets [global] backend= in the conf pick
 * the backend (vulkan/rocm), and feeds raw NV12/P010 frames from a file as HOST planes —
 * exactly what the mpv filter's software path does. Use it to verify a backend on a box
 * with no player (macOS/MoltenVK bring-up, headless CI), and for ms/frame numbers.
 *
 *   aji_host_harness --lib <libaji> --conf animejanai.conf --model-dir DIR [--slot N]
 *                    --input in.raw --width W --height H [--format nv12|p010]
 *                    [--matrix 601|709|2020] [--range limited|full] [--fps F]
 *                    [--frames N] [--inflight N] [--output out.raw] [--rife-dir DIR]
 *   --inflight N keeps N frames submitted before waiting the oldest (the player pipelines
 *   this way; default 1 = one frame at a time, a throughput lower bound).
 *
 *   ffmpeg -i clip.mkv -frames:v 24 -pix_fmt nv12 -f rawvideo in.raw
 *
 * Exit 0 iff a chain was active and every frame inferred; prints in->out geometry and
 * mean ms/frame (wall, submit+wait per frame, i.e. no pipelining — a lower bound on fps).
 */
#include "aji.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void log_cb(void *o, int level, const char *m) { (void)o; fprintf(stderr, "[aji%s] %s\n", level == 0 ? " ERROR" : "", m); }

#define LOAD(name) name##_f = (typeof(name##_f))dlsym(lib, #name); if (!name##_f) { fprintf(stderr, "missing symbol %s\n", #name); return 2; }

int main(int argc, char **argv)
{
    const char *libpath = NULL, *conf = NULL, *model_dir = NULL, *rife_dir = NULL, *input = NULL, *output = NULL;
    int w = 0, h = 0, slot = 1002, frames = 0, inflight = 1, format = AJI_FMT_NV12, matrix = AJI_MATRIX_BT709, range = AJI_RANGE_LIMITED;
    double fps = 23.976;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--lib")) libpath = v; else if (!strcmp(a, "--conf")) conf = v;
        else if (!strcmp(a, "--model-dir")) model_dir = v; else if (!strcmp(a, "--rife-dir")) rife_dir = v;
        else if (!strcmp(a, "--input")) input = v; else if (!strcmp(a, "--output")) output = v;
        else if (!strcmp(a, "--width")) w = atoi(v); else if (!strcmp(a, "--height")) h = atoi(v);
        else if (!strcmp(a, "--slot")) slot = atoi(v); else if (!strcmp(a, "--frames")) frames = atoi(v);
        else if (!strcmp(a, "--fps")) fps = atof(v); else if (!strcmp(a, "--inflight")) inflight = atoi(v) > 1 ? atoi(v) : 1;
        else if (!strcmp(a, "--format")) format = !strcmp(v, "p010") ? AJI_FMT_P010 : AJI_FMT_NV12;
        else if (!strcmp(a, "--matrix")) matrix = !strcmp(v, "601") ? AJI_MATRIX_BT601 : !strcmp(v, "2020") ? AJI_MATRIX_BT2020 : AJI_MATRIX_BT709;
        else if (!strcmp(a, "--range")) range = !strcmp(v, "full") ? AJI_RANGE_FULL : AJI_RANGE_LIMITED;
        else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
        i++;
    }
    if (!libpath || !conf || !model_dir || !input || !w || !h) { fprintf(stderr, "usage: see header comment\n"); return 2; }

    void *lib = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    aji_ctx *(*aji_create_f)(const aji_create_params *);
    int (*aji_configure_f)(aji_ctx *, int, int, double, int *, int *);
    int (*aji_infer_f)(aji_ctx *, const aji_frame *, const aji_frame *, void *);
    uint64_t (*aji_flush_f)(aji_ctx *, void *);
    int (*aji_wait_f)(aji_ctx *, uint64_t);
    const char *(*aji_current_log_f)(aji_ctx *);
    const char *(*aji_last_error_f)(aji_ctx *);
    void (*aji_destroy_f)(aji_ctx **);
    LOAD(aji_create) LOAD(aji_configure) LOAD(aji_infer) LOAD(aji_flush) LOAD(aji_wait)
    LOAD(aji_current_log) LOAD(aji_last_error) LOAD(aji_destroy)
    const char *(*probe)(const char *) = (const char *(*)(const char *))dlsym(lib, "aji_backend_probe");
    fprintf(stderr, "backend probe: %s\n", probe ? probe(conf) : "(symbol absent)");

    aji_create_params p = { .api_version = AJI_API_VERSION, .conf_path = conf, .model_dir = model_dir,
                            .slot = slot, .rife_model_dir = rife_dir, .async_build = 0, .log = log_cb };
    aji_ctx *c = aji_create_f(&p);
    if (!c) { fprintf(stderr, "aji_create failed\n"); return 1; }
    int ow = 0, oh = 0;
    int r = aji_configure_f(c, w, h, fps, &ow, &oh);
    if (r < 0) { fprintf(stderr, "aji_configure: %d %s\n", r, aji_last_error_f(c)); return 1; }
    fprintf(stderr, "%s\n", aji_current_log_f(c));
    if (r == 0) { fprintf(stderr, "no chain active for %dx%d slot %d\n", w, h, slot); return 1; }
    printf("configured: %dx%d -> %dx%d (slot %d)\n", w, h, ow, oh, slot); fflush(stdout);

    const int bps = format == AJI_FMT_P010 ? 2 : 1;
    const size_t in_y = (size_t)w * h * bps, in_uv = (size_t)w * (h / 2) * bps;
    const size_t out_y = (size_t)ow * oh * bps, out_uv = (size_t)ow * (oh / 2) * bps;
    FILE *fi = fopen(input, "rb"), *fo = output ? fopen(output, "wb") : NULL;
    if (!fi) { perror(input); return 2; }
    /* ring of in-flight frames: each keeps its own in/out planes until its ticket is waited */
    unsigned char **ib = malloc(sizeof *ib * inflight), **ob = malloc(sizeof *ob * inflight);
    uint64_t *tk = calloc(inflight, sizeof *tk);
    for (int i = 0; i < inflight; i++) { ib[i] = malloc(in_y + in_uv); ob[i] = malloc(out_y + out_uv); }
    int n = 0, done = 0; double total = 0, first = 0, t_first_done = 0, t_start = 0;
    #define WAIT_SLOT(i) do { \
        r = aji_wait_f(c, tk[i]); \
        if (r < 0) { fprintf(stderr, "aji_wait frame %d: %d %s\n", done, r, aji_last_error_f(c)); return 1; } \
        tk[i] = 0; if (fo) fwrite(ob[i], 1, out_y + out_uv, fo); \
        if (done == 0) { first = now_ms() - t_start; t_first_done = now_ms(); } \
        done++; } while (0)
    for (;;) {
        int i = n % inflight;
        if (tk[i]) WAIT_SLOT(i);
        if ((frames && n >= frames) || fread(ib[i], 1, in_y + in_uv, fi) != in_y + in_uv) break;
        aji_frame in = { w, h, format, matrix, range, AJI_SITING_LEFT, { ib[i], ib[i] + in_y, NULL }, { w * bps, w * bps, 0 } };
        aji_frame out = { ow, oh, format, matrix, range, AJI_SITING_LEFT, { ob[i], ob[i] + out_y, NULL }, { ow * bps, ow * bps, 0 } };
        if (n == 0) t_start = now_ms();
        r = aji_infer_f(c, &in, &out, NULL);
        if (r < 0) { fprintf(stderr, "aji_infer frame %d: %d %s\n", n, r, aji_last_error_f(c)); return 1; }
        tk[i] = aji_flush_f(c, NULL);
        if (!tk[i]) tk[i] = 1;   /* sync engines return a constant; still mark the slot pending */
        n++;
    }
    for (int k = 0; k < inflight; k++) { int i = (n + k) % inflight; if (tk[i]) WAIT_SLOT(i); }
    if (fo) fclose(fo);
    total = now_ms() - t_first_done;
    printf("frames: %d  first: %.1f ms  mean(inflight %d, after first): %.2f ms/frame  (%.1f fps)\n",
           n, first, inflight, n > 1 ? total / (n - 1) : 0, n > 1 ? 1000.0 * (n - 1) / total : 0);
    aji_destroy_f(&c);
    return n > 0 ? 0 : 1;
}

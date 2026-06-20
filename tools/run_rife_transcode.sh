#!/usr/bin/env bash
# run_rife_transcode.sh — wire ffmpeg decode -> rocm_rife_transcode -> ffmpeg encode.
#
# ffmpeg decodes INPUT to raw NV12 on a pipe; rocm_rife_transcode upscales+RIFE-
# interpolates (libaji_rocm, ROCm/MIGraphX) and writes raw NV12; a second ffmpeg
# encodes that to OUTPUT.mp4 at the doubled frame rate. No mpv/display in the path.
#
# Usage: run_rife_transcode.sh [INPUT] [OUTPUT] [W] [H] [FPS]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${HERE}/build/rocm_rife_transcode"

INPUT="${1:-/tmp/clip.mp4}"
OUTPUT="${2:-/home/nathan/rife-test-deploy/rife_transcode_out.mp4}"
W="${3:-640}"
H="${4:-480}"
FPS="${5:-24}"

# Engine: SD Compact 2x + RIFE 2/1 before_upscale -> out = 2x dims, 2x fps.
OUTW=$(( W * 2 ))
OUTH=$(( H * 2 ))
OUTFPS=$(awk "BEGIN{printf \"%g\", ${FPS}*2}")

if [[ ! -f "$INPUT" ]]; then
    echo "[run] INPUT $INPUT missing — creating a 640x480 testsrc clip" >&2
    ffmpeg -y -f lavfi -i "testsrc=size=${W}x${H}:rate=${FPS}:duration=4" \
        -c:v libx264 -pix_fmt yuv420p "$INPUT"
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "[run] INPUT=$INPUT  OUTPUT=$OUTPUT" >&2
echo "[run] in=${W}x${H}@${FPS}  ->  out=${OUTW}x${OUTH}@${OUTFPS}" >&2

# decode -> filter -> encode. set -o pipefail surfaces any stage failure.
ffmpeg -v error -i "$INPUT" -f rawvideo -pix_fmt nv12 - \
  | "$BIN" "$W" "$H" "$FPS" \
  | ffmpeg -v error -f rawvideo -pix_fmt nv12 -s "${OUTW}x${OUTH}" -r "$OUTFPS" -i - \
        -c:v libx264 -pix_fmt yuv420p -y "$OUTPUT"

echo "[run] wrote $OUTPUT" >&2

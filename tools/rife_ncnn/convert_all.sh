#!/bin/bash
# Batch-convert the upstream RIFE fp16 ONNX pack (rife-fp16-1.7z) to ncnn
# .param/.bin pairs for the aji_vk (ncnn-Vulkan) backend.
#
# Per-model recipe (proven bit-exact 2026-06-20, see README.md):
#   1. fp16 -> fp32          (pnnx's onnx frontend crashes on fp16 scalar consts)
#   2. de-batch              (the batch=2 grid_sample island; ncnn has no batch dim)
#   3. pnnx -> ncnn fp16=1   (re-quantizes weights to fp16)
#   4. validate_one.py       (>= 40 dB vs onnxruntime-fp32 on realistic frames)
# Only validated conversions are copied to OUT. _heavy models are skipped: no
# rife_model code maps to them on any backend (they are unselectable product-wide).
#
# Usage: convert_all.sh [PACK_DIR] [OUT_DIR]     env: JOBS=<parallel> (default 3)
# Idempotent: models already present in OUT are skipped, so it resumes cleanly.
set -u
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/nathan/Projects/animejanai-linux/.venv/bin/python}"
PNNX="${PNNX:-/home/nathan/Projects/animejanai-linux/.venv/bin/pnnx}"
PACK="${1:-$HOME/.cache/animejanai-assets/rife-pack}"
OUT="${2:-$HOME/.cache/animejanai-assets/rife-ncnn}"
JOBS="${JOBS:-3}"
mkdir -p "$OUT"

convert_one() {
    local onnx="$1" name work rc
    name="$(basename "$onnx" .onnx)"
    case "$name" in *_heavy*) echo "SKIP $name (heavy: unselectable)"; return 0;; esac
    if [ -f "$OUT/$name.param" ] && [ -f "$OUT/$name.bin" ]; then
        echo "SKIP $name (already converted)"; return 0
    fi
    work="$(mktemp -d "${TMPDIR:-/tmp}/rifecvt.$name.XXXX")"
    {
        "$PY" "$SELF/fp16_to_fp32.py" "$onnx" "$work/${name}_fp32.onnx" &&
        "$PY" "$SELF/debatch.py" "$work/${name}_fp32.onnx" "$work/${name}_db.onnx" &&
        ( cd "$work" && "$PNNX" "${name}_db.onnx" \
              "inputshape=[1,11,256,256]f32" "inputshape2=[1,11,512,384]f32" \
              fp16=1 optlevel=2 ) &&
        [ -f "$work/${name}_db.ncnn.param" ] && [ -f "$work/${name}_db.ncnn.bin" ] &&
        "$PY" "$SELF/validate_one.py" "$work/${name}_fp32.onnx" \
              "$work/${name}_db.ncnn.param" "$work/${name}_db.ncnn.bin" &&
        cp "$work/${name}_db.ncnn.param" "$OUT/$name.param" &&
        cp "$work/${name}_db.ncnn.bin"   "$OUT/$name.bin"
    } > "$OUT/$name.log" 2>&1
    rc=$?
    rm -rf "$work"
    if [ $rc -eq 0 ]; then echo "OK   $name"; else echo "FAIL $name (see $OUT/$name.log)"; fi
    return 0  # one failure must not kill the batch
}
export -f convert_one
export SELF PY PNNX OUT

ls "$PACK"/*.onnx | sort | xargs -P "$JOBS" -I{} bash -c 'convert_one "$@"' _ {}
echo "== done: $(ls "$OUT"/*.param 2>/dev/null | wc -l) pairs in $OUT ($(grep -l 'RESULT PASS' "$OUT"/*.log 2>/dev/null | wc -l) validated PASS)"

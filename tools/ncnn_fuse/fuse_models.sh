#!/bin/bash
# Fuse a directory of ncnn upscale models for aji_vk: Conv+ReLU (ncnnoptimize) + SPAB tails (fuse_spab.py).
#   fuse_models.sh <src-dir with .param/.bin> <dst-dir> [ncnnoptimize path]
# The originals are left untouched in <src-dir>; run on the plain converter output, never on already-fused files.
set -e
SRC=$1; DST=$2; OPT=${3:-$HOME/Projects/ncnn-vk/build-fast/tools/ncnnoptimize}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DST"; T=$(mktemp -d)
for p in "$SRC"/*.param; do
  m=$(basename "$p" .param)
  if grep -q "^SpabTail" "$p"; then echo "$m: already fused, skipping"; continue; fi
  "$OPT" "$SRC/$m.param" "$SRC/$m.bin" "$T/$m.param" "$T/$m.bin" 0 > /dev/null 2>&1
  python3 "$HERE/fuse_spab.py" "$T/$m.param" "$T/$m.bin" "$DST/$m.param" "$DST/$m.bin"
done
rm -rf "$T"

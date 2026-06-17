#!/bin/bash
# Build libaji_vk.so against a vendored ncnn (Vulkan) build tree.
# Set NCNN_BUILD to your ncnn build dir (containing src/libncnn.so + src/*.h).
set -e
NCNN_BUILD="${NCNN_BUILD:-/tmp/ncnn-src}"
# $ORIGIN first: the shipped dist/lib/libaji_vk.so finds its co-located
# libncnn.so.1 with no LD_LIBRARY_PATH; the build-tree path is the dev fallback.
g++ -O2 -shared -fPIC src/aji_vk.cpp src/aji_conf.cpp -Iinclude -Isrc \
  -I"$NCNN_BUILD/src" -I"$NCNN_BUILD/build/src" \
  "$NCNN_BUILD/build/src/libncnn.so" -lvulkan -fopenmp -lpthread \
  -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$NCNN_BUILD/build/src" -o libaji_vk.so
echo "built libaji_vk.so"

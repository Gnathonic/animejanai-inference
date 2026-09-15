#!/bin/bash
# Produce several RIFE+upscale demo clips through the full in-GPU aji_vk pipeline.
set -u
VK=/home/nathan/Projects/aji-vk-integration
NR=/home/nathan/Projects/ncnn-vk/build-fast/src
ONNX=/home/nathan/AnimeJaNai-Linux-vk/mpv-upscale-2x_animejanai-vk/animejanai/onnx
T="$VK/tools/rife_ncnn/vk_rife_transcode"
CLIP="/home/nathan/Downloads/msedge_VpwT3PXcoB-00.00.00.000-00.00.10.144.mp4"
OUT=/tmp/rife_demos
SD=2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo
HD=2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16
export LD_LIBRARY_PATH="$NR:$VK/build-vk"

mkconf() { # name model rife before_upscale -> writes $OUT/$1.conf
  local f="$OUT/$1.conf"
  {
    echo "[global]"; echo "backend=vulkan"; echo "logging=no"
    echo "[slot_2]"; echo "profile_name=$1"
    echo "chain_2_min_resolution=0x0"; echo "chain_2_max_resolution=infxinf"
    echo "chain_2_min_fps=0"; echo "chain_2_max_fps=inf"
    echo "chain_2_model_1_name=$2"
    if [ "$3" = yes ]; then
      echo "chain_2_rife=yes"; echo "chain_2_rife_model=414"
      echo "chain_2_rife_factor_numerator=2"; echo "chain_2_rife_factor_denominator=1"
      echo "chain_2_rife_ensemble=no"; echo "chain_2_rife_scene_detect_threshold=0.150"
      echo "chain_2_rife_before_upscale=$4"
    fi
  } > "$f"; echo "$f"
}

run() { # tag conf W H secs srcfps outfps outW outH
  local tag=$1 conf=$2 W=$3 H=$4 secs=$5 sfps=$6 ofps=$7 oW=$8 oH=$9
  echo "=== $tag : ${W}x${H}@${sfps} -> ${oW}x${oH}@${ofps} (${secs}s) ==="
  local t0=$(date +%s.%N)
  ffmpeg -nostdin -v error -i "$CLIP" -t "$secs" -vf "scale=${W}:${H},format=nv12" -r "$sfps" -f rawvideo -pix_fmt nv12 - 2>/dev/null \
   | "$T" --conf "$conf" --model-dir "$ONNX" --rife-model-dir /tmp/rife_models --format nv12 --slot 2 "$W" "$H" "$sfps" 2>"$OUT/$tag.log" \
   | ffmpeg -nostdin -v error -f rawvideo -pix_fmt nv12 -s "${oW}x${oH}" -r "$ofps" -i - -c:v libx264 -crf 18 -pix_fmt yuv420p -y "$OUT/$tag.mp4" 2>/dev/null
  local rc=${PIPESTATUS[1]}
  local t1=$(date +%s.%N)
  local stats=$(grep -E 'DONE|READY' "$OUT/$tag.log" | tr '\n' ' ')
  printf "  rc=%s  wall=%.1fs  %s\n" "$rc" "$(echo "$t1-$t0"|bc)" "$stats"
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 "$OUT/$tag.mp4" 2>/dev/null | sed 's/^/  out: /'
}

c1=$(mkconf sd_pre   "$SD" yes yes)
c2=$(mkconf hd_pre   "$HD" yes yes)
c3=$(mkconf sd_post  "$SD" yes no)
c4=$(mkconf sd_norife "$SD" no  no)

run demo_sd_pre    "$c1" 480 246 6 30 60  960 492
run demo_hd_pre    "$c2" 960 490 4 30 60 1920 980
run demo_sd_post   "$c3" 480 246 4 30 60  960 492
run demo_sd_norife "$c4" 480 246 6 30 30  960 492
echo "=== demos in $OUT ===" && ls -la "$OUT"/demo_*.mp4

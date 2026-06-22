# RIFE -> ncnn conversion (proven bit-exact, 2026-06-20)

Convert a vsmlrt `rife_v4.x.onnx` to ncnn for the aji_vk (ncnn-Vulkan) backend.

```bash
PY=/home/nathan/Projects/animejanai-linux/.venv/bin/python
RIFE=/home/nathan/AnimeJaNai-Linux/mpv-upscale-2x_animejanai-v0.4.3-linux/animejanai/rife
# 1) fp16 -> fp32 (pnnx onnx frontend crashes on fp16 scalar constants)
$PY fp16_to_fp32.py $RIFE/rife_v4.14.onnx rife_v4.14_fp32.onnx
# 2) de-batch the batch=2 grid_sample island (ncnn has no batch dim)
$PY debatch.py rife_v4.14_fp32.onnx rife_v4.14_db.onnx
# 3) pnnx -> ncnn (fp16=1 re-quantizes weights)
/home/nathan/Projects/animejanai-linux/.venv/bin/pnnx rife_v4.14_db.onnx \
    inputshape=[1,11,256,256]f32 inputshape2=[1,11,512,384]f32 fp16=1 optlevel=2
# 4) validate vs onnxruntime (realistic frames; persistent input buffer)
$PY validate_parity.py
```
Result: ncnn(fp16) vs onnx-fp16 (TRT/ROCm ref) = 68.3 dB; bit-exact vs onnx-fp32.
GAP: ncnn lacks a Vulkan GridSample -> 8 warps fall back to CPU (see memory rife-ncnn-conversion).

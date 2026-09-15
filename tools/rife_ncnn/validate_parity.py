"""Validate a converted ncnn RIFE model vs onnxruntime on realistic frames.

Gotchas baked in:
- REALISTIC input (random is pathological: fp16-vs-fp32 only ~23 dB).
- PERSISTENT input buffer (ncnn.Mat wraps numpy; a temporary -> freed -> garbage 1e30).
Usage: run from a dir containing the onnx + .ncnn.param/.bin (edit names below).
"""
import numpy as np, onnxruntime as ort, ncnn
from rife_io import make_input, stats

ONNX_FP16 = "rife_v4.14.onnx"          # TRT/ROCm reference (may be absent; skipped if so)
ONNX_FP32 = "rife_v4.14_fp32.onnx"
NCNN_PARAM, NCNN_BIN = "rife_v4.14_db.ncnn.param", "rife_v4.14_db.ncnn.bin"

def onnx_run(path, dt, xb):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return np.asarray(s.run(None, {s.get_inputs()[0].name: xb.astype(dt)})[0], np.float32)[0]

def ncnn_run(xin, vulkan=False, fp16=True):
    net = ncnn.Net(); net.opt.use_vulkan_compute = vulkan
    for f in ('use_fp16_packed','use_fp16_storage','use_fp16_arithmetic'): setattr(net.opt, f, fp16)
    net.load_param(NCNN_PARAM); net.load_model(NCNN_BIN)
    ex = net.create_extractor(); mat = ncnn.Mat(xin); ex.input("in0", mat)
    _, out = ex.extract("out0"); r = np.array(out, np.float32).copy(); del mat, net
    return r

for res, shift in ((256, 3), (512, 4)):
    x = make_input(res, res, shift=shift); xb = x[None]
    xin = np.ascontiguousarray(x, np.float32)
    o32 = onnx_run(ONNX_FP32, np.float32, xb)
    print(f"--- {res}x{res} ---")
    stats(ncnn_run(xin, vulkan=False), o32, "ncnn-cpu  vs onnx-fp32")
    stats(ncnn_run(xin, vulkan=True),  o32, "ncnn-vulkan vs onnx-fp32")
    try:
        stats(ncnn_run(xin, vulkan=False), onnx_run(ONNX_FP16, np.float16, xb), "ncnn-cpu  vs onnx-fp16 (ref)")
    except Exception:
        pass

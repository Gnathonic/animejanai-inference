#!/usr/bin/env python
"""Parity-validate one converted RIFE ncnn model vs its fp32 ONNX (CPU).

Usage: validate_one.py <model_fp32.onnx> <model.ncnn.param> <model.ncnn.bin>

Passes (exit 0) when ncnn-cpu (fp16 opts, matching shipped inference precision)
vs onnxruntime-fp32 reaches >= 40 dB at both 256x256 and 512x512 on realistic
optical-flow frames (rife_io.make_input; random input is pathological).
The proven-good v4.14 reference measures ~68 dB; structural conversion failures
(debatch/gridsample miswiring) collapse to <20 dB, so 40 dB separates cleanly.
"""
import sys, os
import numpy as np
import onnxruntime as ort
import ncnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rife_io import make_input, stats

onnx_fp32, param, binf = sys.argv[1], sys.argv[2], sys.argv[3]
THRESH = 40.0

sess = ort.InferenceSession(onnx_fp32, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

worst = 99.0
for res, shift in ((256, 3), (512, 4)):
    x = make_input(res, res, shift=shift)
    ref = np.asarray(sess.run(None, {iname: x[None].astype(np.float32)})[0],
                     np.float32)[0]
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    for f in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic"):
        setattr(net.opt, f, True)
    net.load_param(param)
    net.load_model(binf)
    ex = net.create_extractor()
    xin = np.ascontiguousarray(x, np.float32)  # persistent buffer (see validate_parity.py)
    mat = ncnn.Mat(xin)
    ex.input("in0", mat)
    _, out = ex.extract("out0")
    r = np.array(out, np.float32).copy()
    del mat, ex, net
    p = stats(r, ref, f"{res}x{res} ncnn-cpu vs onnx-fp32")
    worst = min(worst, p)

ok = worst >= THRESH
print(f"RESULT {'PASS' if ok else 'FAIL'} worst={worst:.1f} dB (gate {THRESH})")
sys.exit(0 if ok else 1)

import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, AttributeProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT

def cvt_tensor(t):
    if t.data_type == F16:
        arr = numpy_helper.to_array(t).astype(np.float32)
        nt = numpy_helper.from_array(arr, t.name)
        t.CopyFrom(nt)

# initializers
for init in g.initializer:
    cvt_tensor(init)

# graph inputs/outputs/value_info
for vi in list(g.input) + list(g.output) + list(g.value_info):
    if vi.type.tensor_type.elem_type == F16:
        vi.type.tensor_type.elem_type = F32

# node attributes (Constant value tensors, Cast 'to', embedded tensors)
for node in g.node:
    if node.op_type == 'Cast':
        for a in node.attribute:
            if a.name == 'to' and a.i == F16:
                a.i = F32
    for a in node.attribute:
        if a.type == AttributeProto.TENSOR:
            cvt_tensor(a.t)
        elif a.type == AttributeProto.TENSORS:
            for t in a.tensor:
                cvt_tensor(t)

onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst)

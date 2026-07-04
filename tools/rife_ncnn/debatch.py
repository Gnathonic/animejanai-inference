"""De-batch the RIFE batch=2 island into two batch=1 streams (exact-weight preserving).

The vsmlrt RIFE export stacks frame0/frame1 on the BATCH axis around grid_sample
(channel<->batch Reshapes). ncnn has no batch dim, so we split that island into two
batch=1 streams: boundary-in Reshape([1,2C,H,W]->[2,C,H,W]) -> two channel Slices;
every internal batch=2 node -> _a/_b duplicate (shared weights); boundary-out
Reshape([2,C,H,W]->[1,2C,H,W]) -> Concat(_a,_b,axis=1); Split(axis=0) -> Identity aliases.

_ensemble exports add two topology variants (both handled generically):
  - island entry via Concat(axis=0) of two batch=1 streams -> Identity aliases
    (stream a = input[0], stream b = input[1]);
  - rank-1 length-2 constants (Split sizes) whose lead dim == 2 are misclassified
    as batch2 by the lead-dim test and get consumed by KEPT batch1 nodes; their
    original producers are re-kept alongside the __a/__b duplicates (the
    classification itself is deliberately unchanged so proven non-ensemble
    output stays byte-identical).
"""
import sys, numpy as np, onnx
from onnx import helper, TensorProto, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph

# --- classify via shape inference on a concrete copy (model stays dynamic) ---
mc = onnx.load(src)
d = mc.graph.input[0].type.tensor_type.shape.dim
d[0].dim_value, d[1].dim_value, d[2].dim_value, d[3].dim_value = 1, 11, 64, 64
mc = shape_inference.infer_shapes(mc, strict_mode=False, data_prop=True)
shp = {}
for v in list(mc.graph.value_info) + list(mc.graph.output) + list(mc.graph.input):
    dd = v.type.tensor_type.shape.dim
    shp[v.name] = [(x.dim_value if x.HasField('dim_value') else None) for x in dd]
def lead(n):  s = shp.get(n); return s[0] if s else None
def chan(n):  s = shp.get(n); return s[1] if s and len(s) > 1 else None
def is_b2(n): return lead(n) == 2

nodes = list(g.node)
prod = {o: n for n in nodes for o in n.output}

# node sets
b2_nodes = [n for n in nodes if any(is_b2(o) for o in n.output)]
b2_tensors = {o for n in b2_nodes for o in n.output if is_b2(o)}

def is_boundary_in(n):   # Reshape: batch1 data input -> batch2 output (channel->batch)
    return n.op_type == "Reshape" and is_b2(n.output[0]) and not is_b2(n.input[0])
def is_boundary_out(n):  # Reshape: batch2 input -> batch1 output (batch->channel)
    return n.op_type == "Reshape" and is_b2(n.input[0]) and lead(n.output[0]) == 1
def is_split0(n):        # Split axis=0 of a batch2 tensor -> batch1 halves
    if n.op_type != "Split" or not is_b2(n.input[0]): return False
    ax = next((a.i for a in n.attribute if a.name == "axis"), 0)
    return ax == 0
def is_concat_in(n):     # Concat axis=0 of two batch1 streams -> batch2 (ensemble entry)
    if n.op_type != "Concat" or not is_b2(n.output[0]): return False
    s = shp.get(n.output[0])
    if not s or len(s) < 3: return False          # real batch tensor, not a 1-D shape vector
    ax = next((a.i for a in n.attribute if a.name == "axis"), None)
    return ax == 0 and all(i not in b2_tensors for i in n.input)

# sanity: the whole a/b machinery assumes two streams; no rank>=3 tensor may have lead>2
weird = [(t, s) for t, s in shp.items() if s and len(s) >= 3 and s[0] not in (None, 1, 2)]
if weird:
    print("WARN: rank>=3 tensors with batch>2 (unsupported topology):", weird[:8])

# boundary-in nodes have batch2 OUTPUT (in b2_nodes); boundary-out/split0 have batch1 output (NOT in b2_nodes)
bin_nodes  = [n for n in b2_nodes if is_boundary_in(n)]
cin_nodes  = [n for n in b2_nodes if is_concat_in(n)]
internal   = [n for n in b2_nodes if not is_boundary_in(n) and not is_concat_in(n)]
bout_nodes = [n for n in nodes if is_boundary_out(n)]
split0     = [n for n in nodes if is_split0(n)]

print(f"batch2 nodes={len(b2_nodes)} | boundary_in={len(bin_nodes)} concat_in={len(cin_nodes)} "
      f"boundary_out={len(bout_nodes)} split0={len(split0)} internal={len(internal)}")

def a(t): return t + "__a"
def b(t): return t + "__b"

new_nodes = []
inits = []

# keep all non-island nodes as-is; rebuild batch2 nodes + boundary-out + split0
island = set(id(n) for n in b2_nodes) | set(id(n) for n in bout_nodes) | set(id(n) for n in split0)
keep = [n for n in nodes if id(n) not in island]

# boundary-in -> two channel Slices on input[0]
for n in bin_nodes:
    out = n.output[0]; C = chan(out); src_in = n.input[0]
    assert C is not None, f"no chan for {out}"
    for nm, lo, hi in ((a(out), 0, C), (b(out), C, 2*C)):
        s, e, ax = f"{nm}_s", f"{nm}_e", f"{nm}_ax"
        inits.append(helper.make_tensor(s, TensorProto.INT64, [1], [lo]))
        inits.append(helper.make_tensor(e, TensorProto.INT64, [1], [hi]))
        inits.append(helper.make_tensor(ax, TensorProto.INT64, [1], [1]))
        new_nodes.append(helper.make_node("Slice", [src_in, s, e, ax], [nm], name=nm))

# Concat(axis=0) boundary-in -> Identity aliases (stream a = input[0], b = input[1])
for n in cin_nodes:
    out = n.output[0]
    assert len(n.input) == 2, f"concat-in {n.name}: {len(n.input)} inputs (only batch=2 supported)"
    new_nodes.append(helper.make_node("Identity", [n.input[0]], [a(out)], name="cinA_" + n.name))
    new_nodes.append(helper.make_node("Identity", [n.input[1]], [b(out)], name="cinB_" + n.name))

# internal batch2 nodes -> _a/_b duplicates (shared inits; batch1 inputs shared)
for n in internal:
    for tag in (a, b):
        ins = [tag(i) if i in b2_tensors else i for i in n.input]
        outs = [tag(o) for o in n.output]
        nn = helper.make_node(n.op_type, ins, outs, name=tag(n.name))
        nn.attribute.extend(n.attribute)
        new_nodes.append(nn)

# boundary-out Reshape -> Concat(_a,_b,axis=1) with ORIGINAL output name
for n in bout_nodes:
    src_in = n.input[0]
    new_nodes.append(helper.make_node("Concat", [a(src_in), b(src_in)], [n.output[0]],
                                      name="concat_" + n.name, axis=1))

# Split axis=0 -> Identity aliases (out0=_a, out1=_b)
for n in split0:
    src_in = n.input[0]
    new_nodes.append(helper.make_node("Identity", [a(src_in)], [n.output[0]], name="id_" + n.output[0]))
    if len(n.output) > 1:
        new_nodes.append(helper.make_node("Identity", [b(src_in)], [n.output[1]], name="id_" + n.output[1]))

# repair: kept nodes still consuming a (now-removed) "batch2" tensor. Typical case:
# rank-1 length-2 constants (Split sizes) misclassified as batch2 by the lead-dim
# test, whose consumers include kept batch1 nodes (ensemble models). Re-keep the
# ORIGINAL producer (alongside its __a/__b duplicates) when everything it consumes
# is still available outside the island — exact semantics, weights untouched.
produced_b1 = {o for n in bout_nodes for o in n.output} | {o for n in split0 for o in n.output}
def outside_ok(i):
    return i == "" or i not in b2_tensors or i in produced_b1
leaked = sorted({i for n in keep for i in n.input if i in b2_tensors and i not in produced_b1})
readd, seen, hard_leak = [], set(), []
for t in leaked:
    p = prod.get(t)
    if p is not None and id(p) not in seen and all(outside_ok(i) for i in p.input):
        readd.append(p); seen.add(id(p)); produced_b1.update(p.output)
    elif p is None or id(p) not in seen:
        hard_leak.append(t)
if readd:
    keep += readd
    print(f"re-kept {len(readd)} original island producers for kept consumers:",
          [n.name for n in readd[:8]])
if hard_leak:
    print("LEAK: kept nodes consume removed batch2 tensors:", hard_leak[:12])

# assemble: keep + new island, then topo-sort by output availability
all_nodes = keep + new_nodes
g.ClearField("node")
avail = {i.name for i in g.initializer} | {i.name for i in g.input} | {t.name for t in inits}
ordered = []; pending = list(all_nodes); guard = 0
while pending:
    guard += 1
    progress = False
    rest = []
    for n in pending:
        if all((i == "" or i in avail) for i in n.input):
            ordered.append(n); avail.update(n.output); progress = True
        else:
            rest.append(n)
    pending = rest
    if not progress:
        print("TOPO STALL, unresolved:", [(n.op_type, n.name, [i for i in n.input if i not in avail]) for n in pending[:8]])
        break
if pending:
    print("DROPPED", len(pending), "nodes; e.g.", [(n.op_type, n.name) for n in pending[:6]])
g.node.extend(ordered)
g.initializer.extend(inits)
# drop stale value_info (shapes changed)
del g.value_info[:]
onnx.save(m, dst)            # save BEFORE check so we can inspect on failure
try:
    onnx.checker.check_model(m)
    print("checker OK")
except Exception as e:
    print("checker FAIL:", str(e)[:200])
print("wrote", dst, "nodes:", len(ordered))

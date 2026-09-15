#!/usr/bin/env python3
"""Rewrite an ncnn .param/.bin pair, fusing each SPAN SPAB-block tail into one SpabTail layer.

Pattern (per block), matched structurally on the blob graph, not on names:
    Split       s        1 2 X  X_res X_att
    BinaryOp    add      2 1 R  X_res       A      0=0
    Convolution att      1 1 X_att          T      (1x1, num_input == num_output)
    TanH        tanh     1 1 T              U
    BinaryOp    mul      2 1 A  U           OUT    0=2
becomes
    SpabTail    tail     2 1 R  X           OUT    0=<oc> 1=<ic> 2=<bias> 3=<oc*ic>
The att convolution's weight bytes move to the SpabTail (same on-disk format: tag + weights + bias).
SpabTail is implemented in our ncnn fork (CPU reference + Vulkan pack4 kernel); the fused model
is aji_vk-only. Usage: fuse_spab.py in.param in.bin out.param out.bin
"""
import struct, sys

FP16_TAG = 0x01306B47

def align4(n): return (n + 3) & ~3

def parse_param(path):
    lines = open(path).read().splitlines()
    assert lines[0].strip() == '7767517', 'not an ncnn param (magic)'
    layers = []
    for ln in lines[2:]:
        if not ln.strip(): continue
        t = ln.split()
        typ, name, nin, nout = t[0], t[1], int(t[2]), int(t[3])
        ins = t[4:4 + nin]; outs = t[4 + nin:4 + nin + nout]; params = t[4 + nin + nout:]
        layers.append(dict(type=typ, name=name, ins=ins, outs=outs, params=params))
    return layers

def pdict(layer):
    d = {}
    for kv in layer['params']:
        k, v = kv.split('=', 1); d[int(k)] = v
    return d

WEIGHTLESS = {'Input', 'Split', 'Concat', 'ReLU', 'TanH', 'Sigmoid', 'BinaryOp', 'UnaryOp', 'Reorg', 'PixelShuffle',
              'Interp', 'Padding', 'Crop', 'Eltwise', 'Pooling', 'Reshape', 'Permute', 'Slice', 'Clip', 'Noop',
              'Softmax', 'HardSwish', 'HardSigmoid', 'Swish', 'Mish', 'GELU', 'ELU', 'Dropout', 'Flatten',
              'Squeeze', 'ExpandDims', 'MemoryData', 'Cast', 'Packing', 'GridSample', 'Convolution1D', 'CopyTo'}

def bin_chunk_size(layer, data, off):
    """bytes this layer consumes from the .bin starting at off."""
    t = layer['type']
    p = pdict(layer)
    if t in WEIGHTLESS:
        return 0
    if t == 'PReLU':
        return int(p.get(0, 0)) * 4
    if t in ('Convolution', 'ConvolutionDepthWise', 'Deconvolution', 'DeconvolutionDepthWise', 'InnerProduct', 'SpabTail'):
        if t == 'SpabTail':
            n = int(p.get(3, 0)); bias = int(p.get(2, 0)); nout = int(p.get(0, 0))
        elif t == 'InnerProduct':
            n = int(p.get(2, 0)); bias = int(p.get(1, 0)); nout = int(p.get(0, 0))
        else:
            n = int(p.get(6, 0)); bias = int(p.get(5, 0)); nout = int(p.get(0, 0))
        tag = struct.unpack_from('<I', data, off)[0]
        if tag == FP16_TAG: sz = 4 + align4(n * 2)
        elif tag == 0: sz = 4 + n * 4
        elif tag == 0x0002C056: sz = 4 + n * 4
        else: raise SystemExit('unsupported weight tag 0x%08X in %s (%s)' % (tag, layer['name'], t))
        if bias: sz += nout * 4
        return sz
    raise SystemExit('layer type %s (%s) has an unknown weight layout; refusing to rewrite the .bin' % (t, layer['name']))

def has_spab(layers):
    return any(l['type'] == 'TanH' for l in layers) and any(l['type'] == 'BinaryOp' and pdict(l).get(0, '0') == '2' for l in layers)

def main(pin, bin_in, pout, bin_out):
    layers = parse_param(pin)
    data = open(bin_in, 'rb').read()
    if not has_spab(layers):
        # nothing to fuse (e.g. Compact): pass the files through untouched
        import shutil
        shutil.copyfile(pin, pout); shutil.copyfile(bin_in, bin_out)
        print('%s: no SPAB tails, copied unchanged' % pout)
        return
    # slice the bin per layer (in file order)
    off = 0
    for l in layers:
        sz = bin_chunk_size(l, data, off); l['bin'] = data[off:off + sz]; off += sz
    assert off == len(data), 'bin size mismatch: consumed %d of %d' % (off, len(data))

    producer = {}
    for i, l in enumerate(layers):
        for o in l['outs']: producer[o] = i
    consumers = {}
    for i, l in enumerate(layers):
        for b in l['ins']: consumers.setdefault(b, []).append(i)

    remove = set(); replace = {}   # index of mul layer -> new SpabTail layer
    fused = 0
    for i, mul in enumerate(layers):
        if mul['type'] != 'BinaryOp' or pdict(mul).get(0, '0') != '2' or len(mul['ins']) != 2: continue
        a, u = mul['ins']
        add = layers[producer[a]] if a in producer else None
        tanh = layers[producer[u]] if u in producer else None
        if not add or not tanh: continue
        if add['type'] != 'BinaryOp' or pdict(add).get(0, '0') != '0' or len(add['ins']) != 2: continue
        if tanh['type'] != 'TanH': continue
        att = layers[producer[tanh['ins'][0]]]
        if att['type'] != 'Convolution': continue
        ap = pdict(att)
        if ap.get(1, '1') != '1' or ap.get(11, '1') != '1' or ap.get(2, '1') != '1' or ap.get(3, '1') != '1' or ap.get(4, '0') != '0': continue
        nout = int(ap[0]); nin = int(ap[6]) // nout
        if nin != nout: continue
        split = layers[producer[att['ins'][0]]]
        if split['type'] != 'Split' or len(split['outs']) != 2: continue
        x_res, x_att = split['outs']
        # add must be (R, X_res) in either order; att input must be the other split output
        if att['ins'][0] not in (x_res, x_att): continue
        other = x_res if att['ins'][0] == x_att else x_att
        if other not in add['ins']: continue
        r = add['ins'][0] if add['ins'][1] == other else add['ins'][1]
        # the split outputs must have no other consumers
        if any(len(consumers.get(b, [])) != 1 for b in split['outs']): continue
        if len(consumers.get(a, [])) != 1 or len(consumers.get(u, [])) != 1 or len(consumers.get(tanh['ins'][0], [])) != 1: continue
        X = split['ins'][0]
        tail = dict(type='SpabTail', name=mul['name'].replace('attmul', 'tail') if 'attmul' in mul['name'] else mul['name'] + '_tail',
                    ins=[r, X], outs=list(mul['outs']),
                    params=['0=%d' % nout, '1=%d' % nin, '2=%s' % ap.get(5, '0'), '3=%d' % (nout * nin)], bin=att['bin'])
        for l in (split, add, att, tanh): remove.add(layers.index(l))
        replace[i] = tail; fused += 1

    out_layers = []
    for i, l in enumerate(layers):
        if i in remove: continue
        out_layers.append(replace.get(i, l))
    blobs = set()
    for l in out_layers: blobs.update(l['ins']); blobs.update(l['outs'])
    with open(pout, 'w') as f:
        f.write('7767517\n%d %d\n' % (len(out_layers), len(blobs)))
        for l in out_layers:
            f.write('%-18s %-28s %d %d %s%s\n' % (l['type'], l['name'], len(l['ins']), len(l['outs']),
                    ' '.join(l['ins'] + l['outs']), (' ' + ' '.join(l['params'])) if l['params'] else ''))
    with open(bin_out, 'wb') as f:
        for l in out_layers: f.write(l['bin'])
    print('%s: fused %d SPAB tails, %d -> %d layers' % (pout, fused, len(layers), len(out_layers)))

if __name__ == '__main__':
    if len(sys.argv) != 5: raise SystemExit(__doc__)
    main(*sys.argv[1:])

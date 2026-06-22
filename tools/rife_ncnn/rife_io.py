"""Shared helpers: build a realistic RIFE 11-ch input + parity stats."""
import numpy as np

def make_input(PW, PH, shift=3.0, t=0.5, seed=0):
    """Two smooth, slightly-shifted frames -> realistic optical-flow input."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
    def frame(dx, dy):
        ch = []
        for k in range(3):
            f = 0.5 + 0.25*np.sin((xx+dx)*(0.03+0.01*k) + (yy+dy)*0.02 + k)
            f += 0.15*np.sin((xx+dx)*0.11 - (yy+dy)*0.07*(k+1))
            # a moving soft blob
            cxp, cyp = PW*0.4 + dx, PH*0.55 + dy
            f += 0.3*np.exp(-(((xx-cxp)**2+(yy-cyp)**2)/(2*(PW*0.18)**2)))
            ch.append(f)
        return np.clip(np.stack(ch), 0, 1).astype(np.float32)
    A = frame(0, 0)
    B = frame(shift, shift*0.5)
    x = np.zeros((11, PH, PW), dtype=np.float32)
    x[0:3] = A; x[3:6] = B; x[6] = t
    xs = (2.0*np.arange(PW)/(PW-1) - 1.0).astype(np.float32)
    ys = (2.0*np.arange(PH)/(PH-1) - 1.0).astype(np.float32)
    x[7] = np.broadcast_to(xs, (PH, PW))
    x[8] = np.broadcast_to(ys[:, None], (PH, PW))
    x[9] = 2.0/(PW-1); x[10] = 2.0/(PH-1)
    return x

def stats(a, b, label=""):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    if a.shape != b.shape:
        print(f"  {label:30s} SHAPE MISMATCH a={a.shape} b={b.shape}"); return -1
    d = np.abs(a-b); mse = float(np.mean((a-b)**2))
    rng_ = max(float(b.max())-float(b.min()), 1e-6)
    psnr = 99.0 if mse < 1e-12 else 20*np.log10(rng_) - 10*np.log10(mse)
    print(f"  {label:30s} shape={a.shape} maxdiff={float(d.max()):.5f} meandiff={float(d.mean()):.6f} PSNR={psnr:.1f} dB")
    return psnr

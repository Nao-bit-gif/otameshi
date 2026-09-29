"""Preview 10 motion ideas for the HATTORI symbol mark (the text is excluded).

Usage: python3 symbol_previews.py reference.png out_dir
Writes out_dir/NN_name.mp4 per idea and out_dir/all_ideas_grid.mp4.
"""
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 2
FPS = 30
CYCLE = 4.0              # each idea: ~3.3s of motion, then hold
EXACT0, EXACT1 = 3.3, 3.6
NF = int(CYCLE * FPS)
rng = np.random.default_rng(3)

# Crop around the symbol (reference pixels).
CX0, CY0, CW, CH = 300, 80, 350, 280
W, H = CW * SS, CH * SS


def clamp01(x):
    return np.clip(x, 0, 1)


def smooth(x):
    x = clamp01(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def out_cubic(x):
    return 1 - (1 - clamp01(x)) ** 3


def out_back(x, k=1.7):
    x = clamp01(x)
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


# ---------------------------------------------------------------- layers
ref_full = np.array(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32)
h0, w0, _ = ref_full.shape
SX0, SX1, SY0, SY1 = 355, 592, 100, 342           # symbol area

sat = ref_full.max(2) - ref_full.min(2)
core = (sat > 40).astype(np.uint8)
core[:SY0] = 0
core[SY1:] = 0
n, lab, st, cen = cv2.connectedComponentsWithStats(core, 8)
comps = [i for i in range(1, n) if st[i][4] > 500]
assert len(comps) == 5, len(comps)

plate_full = ref_full.copy()
for y in range(SY0, SY1):
    row = np.concatenate([ref_full[y, 20:SX0 - 5], ref_full[y, SX1 + 5:w0 - 20]])
    med = np.median(row, 0)
    noise = row[rng.integers(0, len(row), SX1 - SX0)] - med
    plate_full[y, SX0:SX1] = med + noise * 0.8

layers_small = []
for c in comps:
    m = (lab == c).astype(np.uint8)
    col = ref_full[cv2.erode(m, np.ones((3, 3), np.uint8)) > 0].mean(0)
    zone = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2) > 0
    d = ref_full - plate_full
    cb = col - plate_full
    a = (d * cb).sum(2) / np.maximum((cb * cb).sum(2), 1)
    a = np.where(zone, clamp01(a), 0).astype(np.float32)
    layers_small.append((a, col))


def crop_up(img, interp=cv2.INTER_CUBIC):
    c = img[CY0:CY0 + CH, CX0:CX0 + CW]
    return cv2.resize(c, (W, H), interpolation=interp)


REF = np.clip(crop_up(ref_full, cv2.INTER_LANCZOS4), 0, 255)
PLATE = np.clip(crop_up(plate_full, cv2.INTER_LANCZOS4), 0, 255)

layers = []
for a, col in layers_small:
    A = clamp01(crop_up(a))
    ys, xs = np.where(A > 0.5)
    layers.append(dict(a=A, col=col.astype(np.float32),
                       P=np.array([xs.mean(), ys.mean()], np.float32),
                       area=len(xs)))
# The dot is the orange (reddest) layer; the rest are blades.
dot_i = int(np.argmax([L["col"][0] - L["col"][2] for L in layers]))
DOT = layers[dot_i]
C = DOT["P"].copy()
blades = [L for i, L in enumerate(layers) if i != dot_i]
for L in blades:
    v = L["P"] - C
    L["psi"] = np.degrees(np.arctan2(v[1], v[0]))
    L["u"] = v / np.linalg.norm(v)
blades.sort(key=lambda L: L["psi"])               # clockwise on screen
for L in [DOT] + blades:
    L["rgb"] = L["col"]

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


# ---------------------------------------------------------------- helpers
def m3(M2):
    return np.vstack([M2, [0, 0, 1]]).astype(np.float64)


def rot(center, deg, s=1.0):
    return m3(cv2.getRotationMatrix2D((float(center[0]), float(center[1])), deg, s))


def tr(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


def axis_scale(P, deg, k):
    """Scale by k along direction `deg` (screen angle) around P."""
    R = rot(P, deg)
    Ri = rot(P, -deg)
    S = np.array([[k, 0, P[0] * (1 - k)], [0, 1, 0], [0, 0, 1]], np.float64)
    return Ri @ S @ R


I3 = np.eye(3)


def put(canvas, rgb, a, M=I3, op=1.0, add=None):
    if op <= 0:
        return
    if M is not I3:
        a = cv2.warpAffine(a, M[:2], (W, H), flags=cv2.INTER_LINEAR)
        if np.ndim(rgb) == 3:
            rgb = cv2.warpAffine(np.ascontiguousarray(rgb), M[:2], (W, H),
                                 flags=cv2.INTER_LINEAR)
    a = a * op
    col = rgb if add is None else rgb + add
    canvas *= 1 - a[..., None]
    canvas += col * a[..., None]


def glow(canvas, a, color, strength, sigma=10):
    g = cv2.GaussianBlur(a, (0, 0), sigma * SS / 2) * strength
    canvas += g[..., None] * np.float32(color)


# ---------------------------------------------------------------- metal
def brushed(h, w, blur=81):
    b = cv2.blur(rng.normal(0, 1, (h, w)).astype(np.float32), (blur, 1))
    f = cv2.blur(rng.normal(0, 1, (h, w)).astype(np.float32), (13, 1))
    return b / b.std() * 0.6 + f / f.std() * 0.4


def metal_of(L, spun=False):
    a = L["a"]
    if spun:  # concentric turned metal for the round dot
        r = np.hypot(XX - L["P"][0], YY - L["P"][1])
        ang = np.arctan2(YY - L["P"][1], XX - L["P"][0])
        ring = rng.normal(0, 1, 400).astype(np.float32)
        ring = np.convolve(ring, np.ones(3) / 3, "same")
        tex = 0.70 + 0.05 * ring[np.clip(r.astype(int), 0, 399)] \
            + 0.18 * np.cos(2 * (ang + np.pi / 4))
    else:
        yn = (YY - L["P"][1]) / 120
        tex = 0.70 + 0.16 * np.cos(yn * 3.0) + 0.05 * brushed(H, W)
    hf = cv2.GaussianBlur(a, (0, 0), 2.4)
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3)
    bevel = np.clip((gx * 0.6 + gy * 0.8) * 1.4, -0.35, 0.45)
    lum = np.clip(tex + bevel, 0, 1.2)
    return (lum[..., None] * np.float32([0.93, 0.97, 1.03]) * 255).astype(np.float32)


# ---------------------------------------------------------------- ideas
def i1_fly_in(t, cv):
    last = 0
    for k, L in enumerate(blades):
        t0 = 0.15 + k * 0.16
        u = (t - t0) / 0.7
        if u <= 0:
            continue
        e = out_back(u, 1.4)
        off = L["u"] * 430 * (1 - e)
        M = tr(*off) @ rot(L["P"], 220 * (1 - e))
        tl = t - (t0 + 0.7)
        flash = 90 * np.exp(-tl / 0.1) if tl > 0 else 0
        put(cv, L["rgb"], L["a"], M, smooth(u / 0.25), add=flash)
        last = t0 + 0.7
    u = (t - last - 0.02) / 0.45
    if u > 0:
        put(cv, DOT["rgb"], DOT["a"], rot(C, 0, max(out_back(u, 2.6), 1e-3)))


METAL = {id(L): metal_of(L, L is DOT) for L in [DOT] + blades}


def i2_metal(t, cv):
    tone = smooth((t - 2.35) / 0.6)
    su = (t - 1.75) / 0.6          # specular sweep
    spec = 0
    if 0 < su < 1:
        d = (XX + 0.5 * YY) - (-100 + su * (W + 500))
        spec = (np.exp(-(d / 30) ** 2) * 200)[..., None]
    items = [(L, 0.1 + k * 0.22, k) for k, L in enumerate(blades)] + [(DOT, 1.05, -1)]
    for L, t0, bi in items:
        u = (t - t0) / 0.42
        if u <= 0:
            continue
        e = out_back(u, 1.6)
        if L is DOT:
            M = rot(C, 0, 1 + 0.9 * (1 - e))
        else:
            side = 1 if bi % 2 else -1
            M = tr(*(L["u"] * 150 * (1 - e) + np.array([0, side * 40]) * (1 - e)))
        tl = t - (t0 + 0.42)
        flash = 110 * np.exp(-tl / 0.1) if tl > 0 else 0
        rgb = METAL[id(L)] * (1 - tone) + L["rgb"] * tone + spec * (1 - tone)
        put(cv, rgb, L["a"], M, smooth(u / 0.25), add=flash)


def i3_vortex(t, cv):
    u = (t - 0.1) / 2.1
    e = out_cubic(u)
    th = 630 * (1 - e)
    r = 2.4 * (1 - e)
    for L in blades:
        R = rot(C, th)
        v = R[:2, :2] @ (L["P"] - C) * r
        put(cv, L["rgb"], L["a"], tr(*v) @ R, smooth(u / 0.2))
    u2 = (t - 1.9) / 0.5
    if u2 > 0:
        put(cv, DOT["rgb"], DOT["a"], rot(C, 0, max(out_back(u2, 2.2), 1e-3)))


def i4_spin(t, cv):
    u = (t - 0.1) / 2.0
    e = out_cubic(u) ** 1.2
    th = -900 * (1 - e)
    tl = t - 2.1
    if tl > 0:  # tiny spring when it locks
        th += 5 * np.exp(-tl / 0.12) * np.sin(tl * 45)
    s = 0.25 + 0.75 * out_back(u, 1.1)
    M = rot(C, th, s)
    for L in blades + [DOT]:
        put(cv, L["rgb"], L["a"], M, smooth(u / 0.15))


def i5_fan(t, cv):
    b0 = blades[0]
    u0 = (t - 0.1) / 0.45
    s = out_back(u0, 2.0)
    for k in reversed(range(len(blades))):
        L = blades[k]
        u = (t - 0.75 - 0.08 * k) / 0.8
        e = out_back(u, 1.5)
        base = L["psi"] - b0["psi"]   # screen-cw degrees from blade 0
        M = rot(C, base * (1 - e)) @ rot(C, 0, max(s, 1e-3))
        put(cv, L["rgb"], L["a"], M, smooth(u0 / 0.3))
    u = (t - 1.75) / 0.45
    if u > 0:
        put(cv, DOT["rgb"], DOT["a"], rot(C, 0, max(out_back(u, 2.6), 1e-3)))


def i6_flip(t, cv):
    items = [(L, 0.15 + k * 0.3) for k, L in enumerate(blades)] + [(DOT, 1.45)]
    for L, t0 in items:
        u = (t - t0) / 0.75
        if u <= 0:
            continue
        phi = np.radians(90 * (1 - out_back(u, 1.3)))
        k = max(abs(np.cos(phi)), 0.02)
        axis = 0 if L is DOT else L["psi"] + 90
        M = axis_scale(L["P"], -axis, k)
        shade = 0.45 + 0.55 * np.cos(phi)
        spec = 140 * np.sin(2 * phi) ** 2 * (phi > 0)
        put(cv, L["rgb"] * shade, L["a"], M, 1.0, add=spec)


def i7_bloom(t, cv):
    items = []
    for k, L in enumerate(blades):
        u = (t - 0.55 - 0.1 * k) / 0.8
        if u > 0:
            e = out_back(u, 1.6)
            items.append((L, rot(C, -40 * (1 - e), max(e, 1e-3)), smooth(u / 0.2)))
    for L, M, op in items:
        put(cv, L["rgb"], L["a"], M, op)
    u = (t - 0.1) / 0.5
    if u > 0:
        s = out_back(u, 3.0)
        tl = t - 0.55
        pulse = 0.12 * np.exp(-tl / 0.15) * np.sin(tl * 30) if tl > 0 else 0
        put(cv, DOT["rgb"], DOT["a"], rot(C, 0, max(s + pulse, 1e-3)))


def _arc_setup(L):
    cnts, _ = cv2.findContours((L["a"] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    (cx, cy), _ = cv2.minEnclosingCircle(max(cnts, key=len))
    ang = (np.degrees(np.arctan2(YY - cy, XX - cx)) + 360) % 360
    occ = np.zeros(360, bool)
    occ[ang[L["a"] > 0.5].astype(int) % 360] = True
    # Largest empty gap -> the sweep starts at its end.
    best, bstart, run, rstart = 0, 0, 0, 0
    for i in range(720):
        if not occ[i % 360]:
            if run == 0:
                rstart = i
            run += 1
            if run > best:
                best, bstart = run, rstart
        else:
            run = 0
    start = (bstart + best) % 360
    span = 360 - best
    L["arc_rel"] = ((ang - start) % 360).astype(np.float32)
    L["arc_span"] = span


for L in blades:
    _arc_setup(L)
DOT_ANG = ((np.degrees(np.arctan2(YY - C[1], XX - C[0])) + 90 + 360) % 360).astype(np.float32)


def i8_draw(t, cv):
    for k, L in enumerate(blades):
        u = (t - 0.1 - k * 0.3) / 0.8
        if u <= 0:
            continue
        head = out_cubic(u) * (L["arc_span"] + 12)
        m = clamp01((head - L["arc_rel"]) / 10)
        edge = np.exp(-((L["arc_rel"] - head + 6) / 8) ** 2) * (1 - smooth((u - 0.8) / 0.2))
        put(cv, L["rgb"], L["a"] * m, op=1.0, add=(edge * 150)[..., None])
    u = (t - 1.4) / 0.6
    if u > 0:
        head = out_cubic(u) * 372
        m = clamp01((head - DOT_ANG) / 10)
        put(cv, DOT["rgb"], DOT["a"] * m)


OUTLINE = np.zeros((H, W), np.float32)
for L in blades + [DOT]:
    k = np.ones((5, 5), np.uint8)
    OUTLINE = np.maximum(OUTLINE, clamp01(cv2.dilate(L["a"], k) - cv2.erode(L["a"], k)))
R_C = np.hypot(XX - C[0], YY - C[1])
A_C = ((np.degrees(np.arctan2(YY - C[1], XX - C[0])) + 90 + 360) % 360).astype(np.float32)


def i9_fill(t, cv):
    u = (t - 0.1) / 1.0
    fill_u = (t - 1.1) / 1.3
    line_op = clamp01(u * 5) * (1 - smooth((t - 2.2) / 0.5))
    head = out_cubic(u) * 372
    lm = clamp01((head - A_C) / 12)
    put(cv, np.float32([235, 240, 245]), OUTLINE * lm, op=line_op)
    if fill_u > 0:
        R = out_cubic(fill_u) * 260
        wob = 10 * np.sin(np.radians(A_C) * 6 + t * 7) + 6 * np.sin(np.radians(A_C) * 11 - t * 5)
        m = clamp01((R + wob - R_C) / 14)
        for L in blades + [DOT]:
            put(cv, L["rgb"], L["a"] * m)
        put(cv, np.float32([235, 240, 245]), OUTLINE * lm * (1 - m), op=line_op)


def i10_ring(t, cv):
    u = t / 0.6
    s0 = 0.96 + 0.04 * smooth(u)
    op = smooth(u)
    T0 = 0.95
    tl = t - T0
    env = np.exp(-tl / 0.28) if tl > 0 else 0
    for L in blades:
        d = 0.05 * (np.linalg.norm(L["P"] - C) / 100)
        tb = tl - d
        eb = np.exp(-tb / 0.3) if tb > 0 else 0
        wob = 7 * eb * np.sin(tb * 28) if tb > 0 else 0
        push = 14 * eb * np.sin(min(tb * 14, np.pi)) if tb > 0 else 0
        M = tr(*(L["u"] * push)) @ rot(C, wob, s0)
        put(cv, L["rgb"], L["a"], M, op)
    ds = 1 + 0.18 * env * (tl > 0)
    put(cv, DOT["rgb"], DOT["a"], rot(C, 0, s0 * ds), op, add=90 * env)
    if tl > 0:
        glow(cv, DOT["a"], [255, 140, 60], 1.8 * env, sigma=18)
        for delay, strength in ((0.0, 1.0), (0.14, 0.5)):
            tt = tl - delay
            if tt <= 0:
                continue
            rad = 70 + tt * 520
            ring = np.exp(-((R_C - rad) / (6 + tt * 18)) ** 2)
            a = np.exp(-tt / 0.35) * strength
            cv += (ring * a)[..., None] * np.float32([255, 215, 170])


IDEAS = [
    ("01_fly_in", "1 四方から飛来", i1_fly_in),
    ("02_metal", "2 金属で組み上げ→色", i2_metal),
    ("03_vortex", "3 渦で中心へ", i3_vortex),
    ("04_spin", "4 風車スピン", i4_spin),
    ("05_fan", "5 扇のように開く", i5_fan),
    ("06_flip", "6 1枚ずつフリップ", i6_flip),
    ("07_bloom", "7 中心から花開く", i7_bloom),
    ("08_draw", "8 弧をなぞって描く", i8_draw),
    ("09_fill", "9 輪郭→色流し込み", i9_fill),
    ("10_ring", "10 光の輪", i10_ring),
]


def frame(fn, t):
    if t >= EXACT1:
        return REF
    acc = 0
    subs = np.linspace(-0.4, 0.4, 4) / FPS
    for s in subs:
        cv = PLATE.copy()
        fn(max(t + s, 0), cv)
        acc = acc + cv
    out = acc / len(subs)
    if t >= EXACT0:
        k = smooth((t - EXACT0) / (EXACT1 - EXACT0))
        out = out * (1 - k) + REF * k
    return out


def writer(path, w, h):
    return subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
TW, TH, LB = 480, 384, 46
COLS, ROWS = 5, 2
font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 24)

tiles = {}
for key, label, fn in IDEAS:
    frames = [np.clip(frame(fn, f / FPS), 0, 255).astype(np.uint8) for f in range(NF)]
    p = writer(os.path.join(out_dir, key + ".mp4"), W, H)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    tiles[key] = [cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA) for fr in frames]
    print("done", key, flush=True)

GW, GH = TW * COLS, (TH + LB) * ROWS
labels = Image.new("RGB", (GW, GH), (0, 0, 0))
dr = ImageDraw.Draw(labels)
for idx, (key, label, _) in enumerate(IDEAS):
    x, y = (idx % COLS) * TW, (idx // COLS) * (TH + LB) + TH
    dr.rectangle([x, y, x + TW - 1, y + LB - 1], fill=(24, 24, 26))
    dr.text((x + 16, y + 9), label, font=font, fill=(235, 235, 235))
labels = np.array(labels)

p = writer(os.path.join(out_dir, "all_ideas_grid.mp4"), GW, GH)
for loop in range(2):
    for f in range(NF):
        g = labels.copy()
        for idx, (key, _, _) in enumerate(IDEAS):
            x, y = (idx % COLS) * TW, (idx // COLS) * (TH + LB)
            g[y:y + TH, x:x + TW] = tiles[key][f]
        p.stdin.write(g.tobytes())
p.stdin.close()
p.wait()
print("grid done")

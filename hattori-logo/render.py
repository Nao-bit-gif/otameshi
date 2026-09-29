"""HATTORI logo animation: each letter is mechanically assembled from brushed
metal parts, snaps into place, and the last frame matches the reference image.

Usage: python3 render.py reference.png out.mp4
"""
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

SS = 2            # render scale relative to the reference
FPS = 30
DUR = 6.5
N = int(DUR * FPS)
rng = np.random.default_rng(7)


def ease_out_cubic(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3


def ease_out_back(x, k=1.9):
    x = np.clip(x, 0, 1)
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


# ---------------------------------------------------------------- analysis
ref_small = np.array(Image.open(sys.argv[1]).convert("RGB"))
h0, w0, _ = ref_small.shape
ref_f = ref_small.astype(np.float32)

# Text region and per-letter labels (7 components: H A T T O R I).
TX0, TX1, TY0, TY1 = 334, 612, 374, 433
lum = ref_f.mean(2)
bin_ = np.zeros((530, w0), np.uint8)
bin_[TY0:TY1, TX0:TX1] = lum[TY0:TY1, TX0:TX1] > 150
n, lab, st, _ = cv2.connectedComponentsWithStats(bin_, 8)
comps = sorted(range(1, n), key=lambda i: st[i][0])
assert len(comps) == 7, len(comps)

# Clean background plate: each row of the text band gets its row median
# (the backdrop is a vertical gradient) plus noise borrowed from the same row.
plate_small = ref_f.copy()
for y in range(TY0, TY1):
    row = np.concatenate([ref_f[y, 20:TX0 - 10], ref_f[y, TX1 + 10:w0 - 20]])
    med = np.median(row, 0)
    noise = row[rng.integers(0, len(row), TX1 - TX0)] - med
    plate_small[y, TX0:TX1] = med + noise * 0.8
bg_row = np.median(plate_small[:, 20:TX0 - 10], 1).mean(1)  # scalar per row

# Anti-aliased alpha of the white text, and the "white" colour it uses.
alpha_small = np.zeros((h0, w0), np.float32)
band = lum[TY0:TY1, TX0:TX1]
bgb = bg_row[TY0:TY1, None]
text_white = np.percentile(band[band > 150], 90)
alpha_small[TY0:TY1, TX0:TX1] = np.clip((band - bgb) / (text_white - bgb), 0, 1)

# Assign every text pixel to its nearest letter component.
seed = np.zeros((h0, w0), np.int32)
for k, c in enumerate(comps):
    seed[:530][lab == c] = k + 1
dist_idx = np.zeros((h0, w0), np.int32)
best = np.full((h0, w0), 1e9, np.float32)
for k in range(7):
    d = cv2.distanceTransform((seed != k + 1).astype(np.uint8), cv2.DIST_L2, 3)
    m = d < best
    best[m] = d[m]
    dist_idx[m] = k

# ---------------------------------------------------------------- upscale
W, H = w0 * SS, h0 * SS
up = lambda a, interp=cv2.INTER_CUBIC: cv2.resize(a, (W, H), interpolation=interp)
ref = np.clip(up(ref_f, cv2.INTER_LANCZOS4), 0, 255)
plate = np.clip(up(plate_small, cv2.INTER_LANCZOS4), 0, 255)
alpha = np.clip(up(alpha_small), 0, 1)
owner = up(dist_idx.astype(np.float32), cv2.INTER_NEAREST).astype(np.int32)
bx0, bx1, by0, by1 = TX0 * SS, TX1 * SS, TY0 * SS, TY1 * SS
white = np.float32(text_white)

# ---------------------------------------------------------------- metal
def brushed(h, w):
    base = rng.normal(0, 1, (h, w)).astype(np.float32)
    streak = cv2.blur(base, (81, 1))
    fine = cv2.blur(rng.normal(0, 1, (h, w)).astype(np.float32), (15, 1))
    s = streak / (streak.std() + 1e-6) * 0.6 + fine / (fine.std() + 1e-6) * 0.4
    return s


letters = []
for k in range(7):
    a = np.where(owner == k, alpha, 0)[by0:by1, bx0:bx1]
    ys, xs = np.where(a > 0.02)
    y0, y1, x0, x1 = ys.min() - 6, ys.max() + 7, xs.min() - 6, xs.max() + 7
    a = a[y0:y1, x0:x1]
    hh, ww = a.shape
    yy = np.linspace(0, 1, hh, dtype=np.float32)[:, None]
    # Chrome-like vertical reflection: bright top, dark horizon, bright base.
    refl = 0.72 + 0.20 * np.cos(yy * np.pi * 1.6) - 0.10 * np.exp(-((yy - 0.55) / 0.12) ** 2)
    tex = refl + 0.055 * brushed(hh, ww)
    # Bevel lighting from the alpha height field.
    hf = cv2.GaussianBlur(a, (0, 0), 2.2)
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3)
    bevel = np.clip(-(gx * -0.6 + gy * -0.8) * 1.4, -0.35, 0.45)
    lumi = np.clip(tex + bevel, 0, 1.2)
    tint = np.array([0.93, 0.97, 1.03], np.float32)  # faint cool steel
    rgb = lumi[..., None] * tint * 255
    letters.append(dict(a=a, rgb=rgb.astype(np.float32), x=bx0 + x0, y=by0 + y0,
                        h=hh, w=ww))

# ---------------------------------------------------------------- timeline
L_START, L_GAP = 0.45, 0.32
PARTS = 3
PART_GAP = 0.09
SLIDE = 0.42
SWEEP0, SWEEP1 = 3.35, 4.45      # final specular sweep across the word
TO_WHITE0, TO_WHITE1 = 4.55, 5.35
CAM_END = 5.45
EXACT0, EXACT1 = 5.55, 5.85      # dissolve to the untouched reference


def lock_time(i):
    return L_START + i * L_GAP + (PARTS - 1) * PART_GAP + SLIDE


def part_offset(i, p, t):
    """Offset (dx, dy) and opacity of part p of letter i at time t."""
    t0 = L_START + i * L_GAP + p * PART_GAP
    u = (t - t0) / SLIDE
    if u <= 0:
        return None
    side = -1 if (p + i) % 2 == 0 else 1
    e = ease_out_back(u, 1.6)
    dist = 150 * SS / 2
    dx = side * dist * (1 - e)
    dy = (p - 1) * 26 * (1 - ease_out_cubic(u))
    op = smooth(u / 0.35)
    return dx, dy, op


def compose_letter(canvas, shadow, L, i, t, tone):
    a, rgb = L["a"], L["rgb"]
    hh, ww = L["h"], L["w"]
    cuts = np.linspace(0, hh, PARTS + 1).astype(int)
    tl = t - lock_time(i)
    # Snap pulse and flash right after the parts lock together.
    pulse = 1 + 0.07 * np.exp(-max(tl, 0) / 0.06) * (tl > 0)
    flash = 0.55 * np.exp(-max(tl, 0) / 0.12) * (tl > 0)
    for p in range(PARTS):
        sub = []
        for s in np.linspace(-0.5, 0.5, 5) / FPS:  # motion blur
            o = part_offset(i, p, t + s)
            if o is not None:
                sub.append(o)
        if not sub:
            continue
        pa = np.zeros_like(a)
        pa[cuts[p]:cuts[p + 1]] = a[cuts[p]:cuts[p + 1]]
        # Machined seam between parts while they are separate.
        if p > 0 and tl < 0.25:
            seam = 0.55 * (1 - smooth(tl / 0.25 + 1)) if tl < 0 else 0.55 * (1 - smooth(tl / 0.25))
            pa[cuts[p]:cuts[p] + 2] *= 1 - seam
        acc_a = 0
        acc_c = 0
        for dx, dy, op in sub:
            M = np.float32([[pulse, 0, (1 - pulse) * ww / 2 + dx],
                            [0, pulse, (1 - pulse) * hh / 2 + dy]])
            X0 = L["x"] - 190
            X1 = L["x"] + ww + 190
            M[0, 2] += L["x"] - X0
            M[1, 2] += L["y"] - by0 + 40
            size = (X1 - X0, by1 - by0 + 80)
            wa = cv2.warpAffine(pa, M, size, flags=cv2.INTER_LINEAR) * op
            wc = cv2.warpAffine(rgb * pa[..., None], M, size, flags=cv2.INTER_LINEAR) * op
            acc_a = acc_a + wa
            acc_c = acc_c + wc
        acc_a = acc_a / len(sub)
        acc_c = acc_c / len(sub)
        # Clip to canvas (canvas is text band padded by 40 rows, 200 cols).
        cx0 = X0 - (bx0 - 200)
        sl = slice(max(cx0, 0), min(cx0 + size[0], canvas.shape[1]))
        ssl = slice(sl.start - cx0, sl.stop - cx0)
        ca = acc_a[:, ssl]
        cc = acc_c[:, ssl]
        metal = cc + flash * 255 * ca[..., None]
        col = metal * (1 - tone) + white * ca[..., None] * tone
        canvas[:, sl] = canvas[:, sl] * (1 - ca[..., None]) + col
        shadow[:, sl] = np.maximum(shadow[:, sl], ca)


def sweep(canvas_rgb, canvas_a, t):
    """Diagonal specular band across the finished metal word."""
    u = (t - SWEEP0) / (SWEEP1 - SWEEP0)
    if not 0 < u < 1:
        return canvas_rgb
    hh, ww = canvas_a.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    pos = -200 + u * (ww + 400)
    d = (xx + 0.45 * (yy - hh / 2)) - pos
    band = np.exp(-(d / 38) ** 2) * 0.85 + np.exp(-(d / 110) ** 2) * 0.25
    return canvas_rgb + band[..., None] * 255 * canvas_a[..., None] * (1 - smooth((t - TO_WHITE0) / 0.4))


# ---------------------------------------------------------------- camera
def camera(t):
    locks = [lock_time(i) for i in range(7)]
    e = 1 - smooth(t / CAM_END)
    s = 1 + 0.045 * e
    ang = -0.35 * e
    tx = 14 * SS * e * np.cos(t * 0.9)
    ty = -8 * SS * e
    for lt in locks:  # tiny impact nudge on each snap
        d = t - lt
        if 0 < d < 0.25:
            k = np.exp(-d / 0.05) * 1.3 * SS
            ty += k * np.cos(d * 70)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


# ---------------------------------------------------------------- render
ff = imageio_ffmpeg.get_ffmpeg_exe()
proc = subprocess.Popen([
    ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart", sys.argv[2]], stdin=subprocess.PIPE)

pad_y0, pad_y1 = by0 - 40, by1 + 40
pad_x0, pad_x1 = bx0 - 200, bx1 + 200
for f in range(N):
    t = f / FPS
    if t >= EXACT1:
        frame = ref
    else:
        tone = smooth((t - TO_WHITE0) / (TO_WHITE1 - TO_WHITE0))
        canvas = np.zeros((pad_y1 - pad_y0, pad_x1 - pad_x0, 3), np.float32)
        cov = np.zeros(canvas.shape[:2], np.float32)
        for i, L in enumerate(letters):
            compose_letter(canvas, cov, L, i, t, tone)
        canvas = sweep(canvas, cov, t)
        region = plate[pad_y0:pad_y1, pad_x0:pad_x1].copy()
        # Soft contact shadow while the parts are metal.
        sh = cv2.GaussianBlur(cov, (0, 0), 5 * SS)
        sh = np.roll(sh, (4 * SS, 3 * SS), (0, 1)) * 0.45 * (1 - tone)
        region *= 1 - sh[..., None]
        region = region * (1 - cov[..., None]) + np.clip(canvas, 0, 255)
        frame = plate.copy()
        frame[pad_y0:pad_y1, pad_x0:pad_x1] = region
        if t >= EXACT0:
            k = smooth((t - EXACT0) / (EXACT1 - EXACT0))
            frame = frame * (1 - k) + ref * k
    if t < CAM_END:
        frame = cv2.warpAffine(frame, camera(t), (W, H), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT)
    proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
proc.stdin.close()
proc.wait()
print("wrote", sys.argv[2], N, "frames")

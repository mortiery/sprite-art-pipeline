"""Walk animation from a video model: start-frame preparation, clip generation, frame analysis.

A standing view (one per direction) is animated by Wan 2.2 image-to-video. The clip is then keyed,
stabilised (every frame re-centred on the character), searched for one clean gait cycle and sampled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def border_color(img: Image.Image) -> tuple[int, int, int]:
    a = np.asarray(img.convert("RGB")).astype(int)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def start_frame(src: str | Path, size: int = 640) -> Image.Image:
    """One standing view -> start frame (see `start_frames`)."""
    return start_frames({"view": Image.open(src)}, size)[0]["view"]


# ------------------------------------------------------------ clip analysis ----
def load_frames(folder: str | Path) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in sorted(Path(folder).glob("*.png"))]


def key_frames(frames: list[Image.Image], tol: float | None = None, hue_tol: float = 0.30,
               min_chroma: float = 0.40) -> list[Image.Image]:
    """Chroma-key every frame against its own border colour. Video models like to add a
    contact shadow: a darker shade of the key colour. Pixels with the key's hue and at least `min_chroma`
    of its saturation are therefore removed at any lightness."""
    from . import pixelize as P, sprites as S
    out = []
    for f in frames:
        bg = border_color(f)                                   # per frame: the background drifts over a clip
        k = P.rgb_to_oklab(np.array(bg, np.float32) / 255.0)
        k_hue, k_chroma = np.arctan2(k[2], k[1]), float(np.hypot(k[1], k[2]))
        keyed = np.array(S.key_background(f, mode="chroma", tol=tol, bg=bg))
        lab = P.rgb_to_oklab(keyed[..., :3].astype(np.float32) / 255.0)
        dh = np.abs((np.arctan2(lab[..., 2], lab[..., 1]) - k_hue + np.pi) % (2 * np.pi) - np.pi)
        same = (dh < hue_tol) & (np.hypot(lab[..., 1], lab[..., 2]) > min_chroma * k_chroma)
        keyed[..., 3] = np.where(same, 0, keyed[..., 3])
        out.append(S._largest_blob_only(Image.fromarray(keyed, "RGBA")))
    return out


def _anchor(rgba: Image.Image):
    """(head centre x, feet y, top y) of the figure, or None for an empty frame."""
    a = np.asarray(rgba.getchannel("A")) >= 128
    ys, xs = np.where(a)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    head = np.where(a[y0:y0 + max(1, int((y1 - y0) * 0.45))])[1]
    return float(np.median(head)) + 0.5, int(y1), int(y0)


def _features(stab: list[Image.Image], n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Small premultiplied RGBA copies: (whole figure, legs only = below 58 % of the figure height)."""
    an = _anchor(stab[0])
    _, y1, y0 = an
    h = stab[0].size[1]
    cut = int((y0 + 0.58 * (y1 - y0)) / h * n)
    full = []
    for f in stab:
        a = np.asarray(f.resize((n, n), Image.BOX), dtype=np.float32) / 255.0
        full.append(np.dstack([a[..., :3] * a[..., 3:4], a[..., 3]]))
    full = np.stack(full)
    return full, full[:, cut:]


def _dist(feat: np.ndarray) -> np.ndarray:
    v = feat.reshape(len(feat), -1)
    sq = (v ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * v @ v.T, 0) / v.shape[1])
    return d


def find_cycle(stab: list[Image.Image], *, warmup: int = 10, min_step: int = 4, max_step: int = 20) -> dict:
    """Locate one gait cycle (two steps) in a stabilised walking clip.

    step  = number of frames between a pose and the next *similar leg silhouette* (half a gait cycle)
    period= 2 * step (left step + right step)
    start = frame of an extreme pose (legs farthest from the standing pose) chosen so that the loop
            start -> start + period closes with the smallest jump.
    """
    full, legs = _features(stab)
    D, DL = _dist(full), _dist(legs)
    n = len(stab)
    g = DL[:, 0]                                         # leg distance to the standing first frame
    # autocorrelation of the leg pose over the steady part of the clip
    warmup = min(warmup, max(2, n // 6))
    lags = np.arange(2, min(2 * max_step, n - warmup - 8) + 1)   # at least 8 frame pairs per lag
    ac = np.array([np.mean([DL[i, i + p] for i in range(warmup, n - p)]) for p in lags])
    # Local minima of the mismatch are whole numbers of steps. The first clear one is either a single step
    # (in profile both steps look alike) or the full cycle. Poses drift apart over time, so a longer lag
    # normally matches worse; if the lag twice as long matches *better*, the first one was a single step and
    # the double is the gait cycle (left step + right step).
    deep = 0.9 * float(np.median(ac))
    cand = [k for k in range(1, len(lags) - 1) if ac[k] <= ac[k - 1] and ac[k] <= ac[k + 1] and ac[k] <= deep
            and lags[k] >= max(5, min_step)]
    if not cand:
        cand = [int(np.argmin(np.where(lags >= 2 * min_step, ac, np.inf)))]
    first = cand[0]
    period = int(lags[first])
    double = [k for k in cand if abs(lags[k] - 2 * period) <= 2 and ac[k] < 0.97 * ac[first]]
    if double:
        period = int(lags[min(double, key=lambda k: ac[k])])
    # if half of it matches nearly as well, the model draws both steps alike (typical in profile);
    # the period is still two steps, but report it
    half = period // 2
    half_ratio = float(ac[lags == half][0] / ac[lags == period][0]) if half >= 2 else float("inf")
    # extreme pose at both half-cycles, different from each other, and a loop that closes. Colours drift the
    # longer a clip runs, so among nearly equal candidates the earliest one wins.
    scores = {}
    for s in range(warmup, n - period):
        m = s + period // 2
        scores[s] = g[s] + g[m] + DL[s, m] - 3.0 * D[s, s + period]
    top = max(scores.values())
    best = min(s for s, v in scores.items() if v >= top - 0.08 * abs(top))
    return {"start": int(best), "period": period, "step": period // 2, "seam": float(D[best, best + period]),
            "typical_frame_diff": float(np.median([D[i, i + 1] for i in range(warmup, n - 1)])),
            "half_ratio": half_ratio, "legs_vs_standing": [round(float(v), 4) for v in g],
            "autocorr": {int(p): round(float(v), 4) for p, v in zip(lags, ac)}}


def sample_cycle(stab: list[Image.Image], start: int, period: int, count: int) -> tuple[list[Image.Image], list[int]]:
    """`count` frames evenly spaced over one cycle (the frame after the last one is the first again)."""
    idx = [int(round(start + k * period / count)) for k in range(count)]
    return [stab[i] for i in idx], idx


def strip(frames: list[Image.Image], bg=(88, 120, 72), scale: float = 1.0, cols: int | None = None) -> Image.Image:
    """Contact sheet of RGBA frames on a neutral background."""
    w, h = frames[0].size
    w2, h2 = max(1, round(w * scale)), max(1, round(h * scale))
    cols = cols or len(frames)
    rows = -(-len(frames) // cols)
    sheet = Image.new("RGB", (cols * w2, rows * h2), bg)
    for i, f in enumerate(frames):
        f = f.convert("RGBA").resize((w2, h2), Image.LANCZOS if scale < 1 else Image.NEAREST)
        sheet.paste(f, ((i % cols) * w2, (i // cols) * h2), f)
    return sheet


def save_anim(frames: list[Image.Image], path: str | Path, fps: float = 8, scale: int = 1, bg=(88, 120, 72),
              lossless: bool = True) -> None:
    """Looping GIF / WEBP preview (nearest-neighbour upscaled) on a flat background."""
    out = []
    for f in frames:
        c = Image.new("RGB", f.size, bg)
        c.paste(f, (0, 0), f.convert("RGBA"))
        out.append(c.resize((f.width * scale, f.height * scale), Image.NEAREST))
    out[0].save(path, save_all=True, append_images=out[1:], duration=round(1000 / fps), loop=0, lossless=lossless, quality=90)


def _head_width(rgba: Image.Image, frac: float = 0.30) -> int:
    a = np.asarray(rgba.getchannel("A")) >= 128
    ys = np.where(a.any(1))[0]
    if len(ys) == 0:
        return 0
    top = a[ys[0]:ys[0] + max(1, int((ys[-1] + 1 - ys[0]) * frac))]
    xs = np.where(top.any(0))[0]
    return int(xs[-1] - xs[0] + 1) if len(xs) else 0


def normalise_size(keyed: list[Image.Image], window: int = 9, deadband: float = 0.015) -> tuple[list[Image.Image], list[float]]:
    """Undo slow zooming (a character that walks towards or away from the camera): every frame is resized
    so that its head is as wide as in frame 0. The measured width is smoothed with a rolling median, so
    hair sway and single bad frames do not make the size jitter. Returns the frames and the scale found."""
    w0 = _head_width(keyed[0])
    widths = np.array([_head_width(f) or w0 for f in keyed], dtype=np.float32)
    half = window // 2
    smooth = np.array([np.median(widths[max(0, i - half):i + half + 1]) for i in range(len(widths))])
    scales = [float(v / w0) for v in smooth]
    out = []
    for f, sc in zip(keyed, scales):
        if abs(sc - 1) < deadband:
            out.append(f)
            continue
        w, h = f.size
        nw, nh = round(w / sc), round(h / sc)
        big = f.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(big, ((w - nw) // 2, (h - nh) // 2))
        out.append(canvas)
    return out, scales


# ------------------------------------------------- grid-locked sprite frames ----
# Downscaling every frame on its own makes the head "shimmer": the same hair lock is sampled at a slightly
# different sub-pixel position in each frame. So the head of every frame is registered against the first
# (standing) frame and all frames are sampled on one fixed grid; the bob of the walk is kept, but snapped
# to whole sprite pixels.
def _gray(rgba: Image.Image, sub: int = 2) -> np.ndarray:
    a = np.asarray(rgba.resize((rgba.width // sub, rgba.height // sub), Image.BOX), dtype=np.float32) / 255.0
    lum = a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.dstack([lum * a[..., 3], a[..., 3]])


def register_head(keyed: list[Image.Image], head_frac: float = 0.36, reach: int = 10, sub: int = 2) -> list[tuple[int, int, float]]:
    """(dx, dy, error) per frame: displacement in source pixels of the head from the head of frame 0 and
    the remaining mismatch (0 = identical head; it grows when the head turns or changes).
    Sum-of-absolute-differences search, started at the previous frame's result."""
    ref = _gray(keyed[0], sub)
    a = ref[..., 1] > 0.5
    ys, xs = np.where(a)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    hy1 = y0 + int((y1 - y0) * head_frac)
    pad = 4
    bx0, bx1, by0, by1 = max(0, x0 - pad), min(ref.shape[1], x1 + pad), max(0, y0 - pad), hy1
    tmpl = ref[by0:by1, bx0:bx1]
    big = reach * 4
    out, px, py = [(0, 0, 0.0)], 0, 0
    for f in keyed[1:]:
        g = np.pad(_gray(f, sub), ((big, big), (big, big), (0, 0)))
        best = (1e9, px, py)
        for step, r in ((2, reach), (1, 2)):
            cx, cy = best[1], best[2]
            for dy in range(cy - r, cy + r + 1, step):
                for dx in range(cx - r, cx + r + 1, step):
                    if abs(dx) > big - 1 or abs(dy) > big - 1:
                        continue
                    win = g[big + by0 + dy:big + by1 + dy, big + bx0 + dx:big + bx1 + dx]
                    sad = float(np.abs(win - tmpl).mean())
                    if sad < best[0]:
                        best = (sad, dx, dy)
        err, px, py = best
        out.append((px * sub, py * sub, err))
    return out


def lock_frames(keyed: list[Image.Image], cell: float) -> tuple[list[Image.Image], dict]:
    """Pixel-grid aware stabilisation: the head of every frame lands on the same sampling grid as in frame 0
    (horizontal drift removed completely, vertical bob snapped to multiples of `cell` source pixels so that
    the feet stay as close to the ground line of frame 0 as possible). Frame 0 itself is not moved.
    stats: cx / feet = head centre column and ground line of frame 0 (anchor for `locked_sprite_frames`)."""
    w, h = keyed[0].size
    cx0, feet0, _ = _anchor(keyed[0])
    offs = register_head(keyed)
    out, bobs = [], []
    for f, (dx, dy, _) in zip(keyed, offs):
        an = _anchor(f)
        feet = an[1] if an else feet0
        q = int(round((feet0 - feet + dy) / cell))           # whole sprite pixels the body sits lower (+) / higher (-)
        bobs.append(q)
        shifted = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shifted.paste(f, (-dx, int(round(-dy + q * cell))))
        out.append(shifted)
    stats = {"cx": float(cx0), "feet": int(feet0), "drift_x": int(np.ptp([o[0] for o in offs])),
             "drift_y": int(np.ptp([o[1] for o in offs])), "bob_px": [int(v) for v in bobs],
             "head_error": [round(float(o[2]), 4) for o in offs]}
    return out, stats


def locked_sprite_frames(locked: list[Image.Image], cell: float, anchor: tuple[float, float], *, frame: int = 48,
                         method: str = "nearest", bottom_margin: int = 1) -> list[Image.Image]:
    """Sample locked frames on one fixed grid. anchor = (head centre column, ground line) of the standing
    frame: the head sits in the middle column and the feet on the bottom row; if something (a staff, a wide
    stride) would stick out sideways, the window slides just far enough to keep it inside."""
    from . import pixelize as P, sprites as S
    cx, feet = anchor
    side = frame * cell
    x0 = cx - side / 2
    cover = np.max([np.asarray(f.getchannel("A")) for f in locked], axis=0) >= 128
    xs = np.where(cover.any(0))[0]
    if len(xs):
        ux0, ux1 = int(xs.min()), int(xs.max()) + 1
        x0 = (ux0 + ux1 - side) / 2 if ux1 - ux0 > side else max(min(x0, ux0), ux1 - side)
    y1 = feet + bottom_margin * cell
    box = (int(round(x0)), int(round(y1 - side)), int(round(x0 + side)), int(round(y1)))
    return [P.remove_specks(S.downscale(f.crop(box), frame, frame, method)) for f in locked]


# ------------------------------------------------------- side views from a turn ----
def _hshift(a: np.ndarray, b: np.ndarray, reach: int = 6) -> int:
    """Horizontal shift (px) that best maps picture a onto picture b (sum of absolute differences)."""
    best, best_s = None, 0
    w = a.shape[1]
    for sft in range(-reach, reach + 1):
        x0, x1 = max(0, sft), min(w, w + sft)
        d = float(np.abs(b[:, x0:x1] - a[:, x0 - sft:x1 - sft]).mean())
        if best is None or d < best:
            best, best_s = d, sft
    return best_s


def turn_profile(keyed: list[Image.Image]) -> dict:
    """Find the side view in a clip in which the character turns from the front view (first frame) to the
    back view (last frame).

    index  : the frame at which half of the clip's total motion has happened - with both ends pinned, that is
             the middle of the turn. (Body width and visible skin were tried first; a wide hat, a robe or
             pale hair defeat both.)
    facing : while a figure turns, everything on its front slides towards the side it is turning to; the
             summed horizontal shift of the upper body over the first half of the turn gives that side.
    Returns {"index", "facing": "left" | "right" | None, "shift", "problems"}."""
    from . import sprites as S
    small = []
    for f in keyed:
        a = np.asarray(f.resize((160, 160), Image.BOX), dtype=np.float32) / 255.0
        small.append((a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)) * a[..., 3])
    motion = np.array([float(np.abs(small[i + 1] - small[i]).mean()) for i in range(len(small) - 1)])
    prog = np.concatenate([[0.0], np.cumsum(motion)]) / max(1e-9, motion.sum())
    idx = int(np.argmin(np.abs(prog - 0.5)))
    ys = np.where(small[0].sum(1) > 0)[0]
    y0, y1 = ys[0], ys[0] + int((ys[-1] + 1 - ys[0]) * 0.6)          # head and chest
    shift = sum(_hshift(small[i][y0:y1], small[i + 1][y0:y1]) for i in range(idx))
    facing = "right" if shift >= 3 else "left" if shift <= -3 else None
    out = {"index": idx, "facing": facing, "shift": int(shift), "problems": []}
    if facing is None:
        out["problems"].append("could not tell which way the character turns")
    f96 = [f.resize((96, 96), Image.BOX) for f in (keyed[0], keyed[-1])]
    skin = S._skin_color(f96[0])
    if skin is not None and S._skin_stats(f96[1], skin, 0.035)[0] > 0.4 * max(1, S._skin_stats(f96[0], skin, 0.035)[0]):
        out["problems"].append("the last frame still shows a face: no back view")
    return out


# ------------------------------------------------------------ one-call API ----
def start_frames(views: dict[str, Image.Image], size: int = 640, fill: float = 0.74, max_width: float = 0.80,
                 feet: float = 0.90) -> tuple[dict[str, Image.Image], dict]:
    """Standing views (flat key background, any size) -> square start frames for the video model.
    The views are brought to the same body size (measured around the head axis, so props do not count),
    scaled together to fit the frame, centred on the head axis and put on one perfectly flat background
    (the colour of the first view). Returns the frames and meta = {"cell": source pixels per sprite pixel}:
    the largest figure is 44 sprite pixels tall or wide."""
    from . import sprites as S
    info, ref_core, bg = {}, None, None
    for d, im in views.items():
        im = im.convert("RGB")
        bg = bg or border_color(im)
        keyed = S._largest_blob_only(S.key_background(im))
        a = np.asarray(keyed.getchannel("A")) >= 128
        ys, xs = np.where(a)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        core = S.core_height(keyed)
        ref_core = ref_core or core
        rel = min(1.3, max(0.77, ref_core / max(1, core)))      # same body size as the first view
        head = np.where(a[box[1]:box[1] + max(1, int((box[3] - box[1]) * 0.45))])[1]
        info[d] = (keyed, box, rel, float(np.median(head)) + 0.5)
    fh = max((b[3] - b[1]) * r for _, b, r, _ in info.values())
    fw = max((b[2] - b[0]) * r for _, b, r, _ in info.values())
    s = min(fill * size / fh, max_width * size / fw)
    out = {}
    for d, (keyed, (x0, y0, x1, y1), rel, cx) in info.items():
        k = s * rel
        w, h = max(1, round((x1 - x0) * k)), max(1, round((y1 - y0) * k))
        fig = keyed.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS)
        ox = round(size / 2 - (cx - x0) * k)                    # head axis on the centre line ...
        ox = min(max(ox, round(size * 0.03)), size - round(size * 0.03) - w)   # ... unless a prop would leave the frame
        frame = Image.new("RGB", (size, size), bg)
        frame.paste(fig, (ox, round(size * feet) - h), fig)
        out[d] = frame
    return out, {"cell": float(max(fh, fw) * s / 44), "size": size}


def missing_upper(standing: Image.Image, frame: Image.Image, n: int = 160, cut: float = 0.58) -> float:
    """Share of the standing figure's silhouette above the waist that `frame` (position-locked) does not
    cover, with 3 px tolerance. Arm swing and hair sway stay below 2 %; a staff or sword that fades out for a
    few frames shows up as 4 % and more."""
    from PIL import ImageFilter
    m0 = np.asarray(standing.getchannel("A").resize((n, n), Image.BOX)) >= 128
    ys = np.where(m0.any(1))[0]
    m0[ys[0] + int((ys[-1] + 1 - ys[0]) * cut):] = False
    cover = np.asarray(frame.getchannel("A").resize((n, n), Image.BOX).filter(ImageFilter.MaxFilter(7))) >= 128
    return float((m0 & ~cover).sum() / max(1, m0.sum()))


def analyse_clip(folder: str | Path, cell: float) -> dict:
    """Key + lock a walking clip, find one gait cycle and check it.
    Returns {"locked": frames, "anchor": (cx, feet), "cycle": {...}, "lock": {...}, "problems": [...]}"""
    keyed, scales = normalise_size(key_frames(load_frames(folder)))
    locked, stats = lock_frames(keyed, cell)
    cyc = find_cycle(locked)
    s, p = cyc["start"], cyc["period"]
    g = cyc["legs_vs_standing"]
    problems = []
    if max(g[s:s + p]) < 0.07:
        problems.append("the legs barely move")
    if cyc["seam"] > 2.0 * cyc["typical_frame_diff"] + 0.005:
        problems.append(f"the walk cycle does not close cleanly (jump {cyc['seam']:.3f})")
    size = float(np.median(scales[s:s + p]))                # zoom that had to be undone
    if abs(size - 1) > 0.15:
        problems.append(f"the character changes size a lot during the clip (x{size:.2f}): it walks towards or away from the camera")
    cyc["size_ratio"] = round(size, 3)
    lost = max(missing_upper(locked[0], f) for f in locked[s:s + p])
    if lost > 0.035:
        problems.append(f"part of the figure above the waist disappears during the walk ({lost:.0%} of the standing "
                        f"silhouette): a carried item vanishes in some frames")
    cyc["upper_lost"] = round(float(lost), 4)
    head = float(np.median(stats["head_error"][s:s + p]))
    if head > 0.09:
        problems.append(f"the head changes during the walk (error {head:.3f}): turned away or redrawn?")
    cyc["head_error"] = round(head, 4)
    return {"locked": locked, "anchor": (stats["cx"], stats["feet"]), "cycle": cyc, "lock": stats, "problems": problems}

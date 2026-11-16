"""Image helpers for sprites and faces.

    choose_key_color(img)          background colour farthest from the character's own colours
    key_background(img)            flat-colour background -> alpha (chroma key, or border flood fill for white)
    downscale(img, w, h, method)   grid downscale: nearest (crisp), k-centroid (smooth), line-preserving (faces)
    count_figures(img)             how many separate figures a concept image shows
    face_crop_box(img)             crop box for the dialogue face
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import pixelize as P
from .sheets import DIRS


KEY_COLORS = {"green": (0, 255, 0), "magenta": (255, 0, 255), "blue": (0, 0, 255)}
KEY_PROMPT = {"green": "solid flat pure bright green (#00FF00)", "magenta": "solid flat pure magenta (#FF00FF)",
              "blue": "solid flat pure blue (#0000FF)"}


def choose_key_color(img: Image.Image, candidates=("green", "magenta", "blue")) -> str:
    """Pick the chroma-key colour that is farthest (OKLab) from every colour of the character.
    `img` should be the cut-out concept or chibi (RGBA) or a picture on a white background."""
    rgba = img if img.mode == "RGBA" and np.asarray(img.getchannel("A")).min() < 128 else key_background(img, mode="flood")
    small = rgba.resize((96, max(1, round(96 * rgba.height / rgba.width))), Image.NEAREST)
    a = np.asarray(small, dtype=np.float32) / 255.0
    lab = P.rgb_to_oklab(a[a[..., 3] > 0.5][:, :3])
    best, best_d = candidates[0], -1.0
    for name in candidates:
        # the model renders key colours less saturated than asked, so test against a muted version too
        ks = np.array([KEY_COLORS[name], tuple(int(60 + 0.7 * v) for v in KEY_COLORS[name])], np.float32) / 255.0
        d = min(np.percentile(np.sqrt(((lab - k) ** 2).sum(-1)), 2) for k in P.rgb_to_oklab(ks))
        if d > best_d:
            best, best_d = name, d
    return best


def key_background(img: Image.Image, mode: str = "chroma", tol: float | None = None,
                   bg: tuple[int, int, int] | None = None) -> Image.Image:
    """Flat background -> alpha. The background colour is measured from the image border.

    mode="chroma": remove *every* pixel close to the key colour, also in enclosed pockets (between a staff
                   and the body, inside a ring ...). Use with a green/magenta/blue background.
    mode="flood" : remove only background-coloured pixels connected to the border. Use with a white
                   background, where a white shirt enclosed by line art must survive (pockets stay opaque).
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if bg is None:
        edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg_f = np.median(edge, axis=0)
    else:
        bg_f = np.array(bg, np.float32) / 255.0
    dist = np.sqrt(((P.rgb_to_oklab(rgb) - P.rgb_to_oklab(bg_f)) ** 2).sum(-1))
    if mode == "chroma":
        t = 0.10 if tol is None else tol
        removed = (dist < t) | _border_connected(dist < t * 1.6)
    else:
        removed = _border_connected(dist < (0.06 if tol is None else tol))
    out = np.dstack([(rgb * 255 + 0.5).astype(np.uint8), np.where(removed, 0, 255).astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def _spread_rows(reach: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Within every horizontal run of `near` pixels, mark the whole run reached if any pixel is."""
    change = np.ones(near.shape, bool)
    change[:, 1:] = near[:, 1:] != near[:, :-1]
    lab = np.cumsum(change.ravel()) - 1
    hit = np.bincount(lab, weights=reach.ravel(), minlength=lab[-1] + 1) > 0
    return hit[lab].reshape(near.shape) & near


def _border_connected(near: np.ndarray) -> np.ndarray:
    """4-connected flood fill from the image border over `near` (vectorised run propagation)."""
    reach = np.zeros(near.shape, bool)
    reach[0], reach[-1], reach[:, 0], reach[:, -1] = near[0], near[-1], near[:, 0], near[:, -1]
    count = -1
    while reach.sum() != count:
        count = reach.sum()
        reach = _spread_rows(reach, near)
        reach = np.ascontiguousarray(_spread_rows(np.ascontiguousarray(reach.T), np.ascontiguousarray(near.T)).T)
    return reach


def downscale(img: Image.Image, w: int, h: int, method: str = "nearest") -> Image.Image:
    """method: "nearest" (crisp: one real source pixel per output pixel, edge ring eroded first),
    "kcentroid" (smooth: dominant colour per cell), "kcentroid_lines" (dominant colour, but keeps dark line art)."""
    if method == "nearest":
        return P.nearest_downscale(img, w, h, erode=2)
    return P.kcentroid_downscale(img, w, h, prefer_dark=0.3 if method == "kcentroid_lines" else None)


def _bbox(alpha: np.ndarray):
    ys, xs = np.where(alpha)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _largest_blob_only(cell: Image.Image, min_frac: float = 0.02) -> Image.Image:
    """Drop stray fragments (bits of a neighbour poking into the cell): keep columns/rows near the main mass."""
    a = np.asarray(cell.getchannel("A")) >= 128
    if not a.any():
        return cell
    # remove tiny components touching the cell border by eroding from the edges until the mass is solid
    col = a.sum(0)
    row = a.sum(1)
    thr_c, thr_r = max(1, int(a.shape[0] * min_frac)), max(1, int(a.shape[1] * min_frac))
    cols = np.where(col >= thr_c)[0]
    rows = np.where(row >= thr_r)[0]
    if len(cols) == 0 or len(rows) == 0:
        return cell

    def main_run(idx):  # longest contiguous run
        splits = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
        best = max(splits, key=len)
        return best[0], best[-1] + 1

    x0, x1 = main_run(cols)
    y0, y1 = main_run(rows)
    arr = np.asarray(cell).copy()
    keep = np.zeros(a.shape, bool)
    keep[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = True
    arr[~keep] = 0
    return Image.fromarray(arr, "RGBA")


def core_height(cell: Image.Image) -> int:
    """Figure height measured only in a narrow band around the head centre, so that side props
    (a staff, a sword hilt, a scarf tail) do not change it. Used to match scales between images."""
    a = np.asarray(cell.getchannel("A")) >= 128
    b = _bbox(a)
    if b is None:
        return 0
    x0, y0, x1, y1 = b
    xs = np.where(a[y0:y0 + max(1, int((y1 - y0) * 0.45))])[1]
    cx = int(np.median(xs)) if len(xs) else (x0 + x1) // 2
    r = max(2, int((y1 - y0) * 0.12))
    rows = np.where(a[:, max(0, cx - r):cx + r].any(1))[0]
    return int(rows.max() - rows.min() + 1) if len(rows) else 0


def frames_preview(frames: dict[str, list[Image.Image]], scale: int = 4, bg=(88, 120, 72, 255)) -> Image.Image:
    ims = [f for d in DIRS for f in frames[d]]
    return P.contact_sheet(ims, len(frames[DIRS[0]]), bg=bg, pad=2, scale=scale)


# ------------------------------------------------------------------ walk animation helpers ----

def _figure(frame: Image.Image):
    a = np.asarray(frame.getchannel("A")) >= 128
    b = _bbox(a)
    if b is None:
        return a, None, 0.0
    x0, y0, x1, y1 = b
    top = a[y0:y0 + max(1, int((y1 - y0) * 0.45))]
    xs = np.where(top)[1]
    return a, b, (float(np.median(xs)) + 0.5 if len(xs) else (x0 + x1) / 2)


def _skin_color(front_idle: Image.Image):
    """Skin colour of a front-view sprite: the most frequent light, peach-hued colour in the upper 60 %.
    The hue window excludes silver/white hair (too little yellow) and gold trim (too yellow)."""
    a, b, cx = _figure(front_idle)
    if b is None:
        return None
    x0, y0, x1, y1 = b
    arr = np.asarray(front_idle)[y0:y0 + int((y1 - y0) * 0.6)].reshape(-1, 4)
    px = arr[arr[:, 3] > 0][:, :3]
    if len(px) == 0:
        return None
    cols, cnt = np.unique(px // 8 * 8 + 4, axis=0, return_counts=True)
    lab = P.rgb_to_oklab(cols.astype(np.float32) / 255.0)
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    ok = (lab[:, 0] > 0.6) & (lab[:, 1] > 0.006) & (lab[:, 2] > 0.012) & (lab[:, 2] < 0.075) & (hue > 25) & (hue < 76)
    if not ok.any() or cnt[ok].max() < 4:
        return None
    return cols[ok][cnt[ok].argmax()].astype(np.uint8)


def _skin_stats(frame: Image.Image, skin, tol: float = 0.07) -> tuple[int, float]:
    """(number of skin pixels in the head area, their mean x offset from the head centre in px).
    tol = colour distance in OKLab; 0.07 survives palette reduction, 0.035 separates pale skin from silver hair."""
    a, b, cx = _figure(frame)
    if b is None:
        return 0, 0.0
    x0, y0, x1, y1 = b
    arr = np.asarray(frame)[y0:y0 + int((y1 - y0) * 0.55)]
    lab = P.rgb_to_oklab(arr[..., :3].astype(np.float32) / 255.0)
    m = (np.sqrt(((lab - P.rgb_to_oklab(skin.astype(np.float32) / 255.0)) ** 2).sum(-1)) < tol) & (arr[..., 3] > 0)
    n = int(m.sum())
    if n == 0:
        return 0, 0.0
    return n, float(np.where(m)[1].mean() - cx)


def count_figures(img: Image.Image, min_gap: float = 0.03, min_share: float = 0.2) -> int:
    """How many separate figures stand side by side on a flat background (concept QC: we want exactly 1)."""
    a = np.asarray(key_background(img, mode="flood").getchannel("A")) > 0
    col = a.sum(0)
    occupied = col > a.shape[0] * 0.01
    gap = max(2, int(a.shape[1] * min_gap))
    runs, start, empty = [], None, 0
    for x, o in enumerate(occupied):
        if o:
            if start is None:
                start = x
            empty = 0
        elif start is not None:
            empty += 1
            if empty >= gap:
                runs.append((start, x - empty + 1))
                start = None
    if start is not None:
        runs.append((start, len(occupied)))
    areas = [col[x0:x1].sum() for x0, x1 in runs]
    return sum(1 for ar in areas if ar >= min_share * max(areas)) if areas else 0


def face_crop_box(neutral_rgba: Image.Image, zoom: float = 0.8) -> tuple[int, int, int, int]:
    """Square crop that frames the head. Compute once on the neutral face and reuse for every emotion."""
    a = np.asarray(neutral_rgba.getchannel("A")) >= 128
    b = _bbox(a)
    w, h = neutral_rgba.size
    side = int(min(w, h) * zoom)
    if b is None:
        return (0, 0, side, side)
    top = a[b[1]:b[1] + max(1, int((b[3] - b[1]) * 0.35))]
    cx = float(np.median(np.where(top)[1]))
    x0 = int(min(max(0, cx - side / 2), w - side))
    y0 = int(min(max(0, b[1] - side * 0.04), h - side))
    return (x0, y0, x0 + side, y0 + side)

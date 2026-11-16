"""Deterministic pixel-art post-processing (numpy + Pillow only).

The diffusion models give us *pixel-art-looking* pictures at ~1 MP. These functions turn them
into *real* pixel art: one colour per logical pixel, hard alpha, a small shared palette.

Main entry points
    kcentroid_downscale(img, w, h)   detail-preserving downscale (dominant colour per cell)
    build_palette(images, n)         k-means palette in OKLab over one or many images
    apply_palette(img, palette)      snap every pixel to the nearest palette colour
    harden_alpha(img)                binary alpha
    remove_specks(img)               drop isolated opaque pixels / fill pinholes
    outline(img)                     1px dark outline (inner or outer)
    pixelize(img, w, h, ...)         all of the above in the right order
    upscale(img, k)                  nearest-neighbour preview upscale
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# --------------------------------------------------------------- colour ----

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) float 0..1 sRGB -> OKLab."""
    lin = _srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ], axis=-1)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.stack([
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ], axis=-1)
    return _linear_to_srgb(lin)


def _rgba(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


# ------------------------------------------------------------ downscale ----

def erode_alpha(img: Image.Image, n: int = 2) -> Image.Image:
    """Shrink the opaque area by n source pixels. Removes the anti-aliased edge ring, which is a blend of
    character and key colour (the "magenta fringe")."""
    arr = np.asarray(img.convert("RGBA")).copy()
    m = arr[..., 3] >= 128
    for _ in range(n):
        p = np.pad(m, 1, constant_values=False)
        m = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    arr[..., 3] = np.where(m, 255, 0)
    arr[~m, :3] = 0
    return Image.fromarray(arr, "RGBA")


def nearest_downscale(img: Image.Image, w: int, h: int, erode: int = 2) -> Image.Image:
    """Point sampling: every output pixel is one real source pixel, so line art and small highlights stay
    crisp (more detail, more colours, slightly noisier than k-centroid). The edge ring is eroded first so
    that no key-colour fringe can be sampled."""
    src = erode_alpha(img, erode) if erode else img.convert("RGBA")
    return src.resize((w, h), Image.NEAREST)


def kcentroid_downscale(img: Image.Image, w: int, h: int, k: int = 2, iters: int = 6,
                        alpha_threshold: float = 0.5, prefer_dark: float | None = None,
                        strong_contrast: float = 0.38, strong_share: float = 0.10) -> Image.Image:
    """Downscale to (w, h): each output pixel = centroid of the largest colour cluster in its cell.

    Unlike box/bicubic filtering this never invents in-between colours, so outlines stay dark and
    1px features (eyes!) survive. Same idea as Astropulse's k-centroid, vectorised over all cells.
    """
    src = img.convert("RGBA")
    # resample to an exact integer multiple of the target so every cell is fx*fy source pixels
    fx = max(1, round(src.width / w))
    fy = max(1, round(src.height / h))
    if (src.width, src.height) != (w * fx, h * fy):
        src = src.resize((w * fx, h * fy), Image.LANCZOS)
    a = _rgba(src)
    cells = a.reshape(h, fy, w, fx, 4).transpose(0, 2, 1, 3, 4).reshape(h * w, fy * fx, 4)
    alpha = cells[..., 3]
    opaque = alpha >= alpha_threshold                       # (N, P)
    coverage = opaque.mean(axis=1)                          # share of opaque source pixels per cell
    lab = rgb_to_oklab(cells[..., :3])                      # (N, P, 3)
    wgt = opaque.astype(np.float32)
    wgt_safe = np.where(wgt.sum(1, keepdims=True) > 0, wgt, 1.0)

    # init centroids: darkest, lightest, then mean (deterministic)
    L = np.where(wgt_safe > 0, lab[..., 0], np.nan)
    idx_dark = np.nanargmin(L, axis=1)
    idx_light = np.nanargmax(L, axis=1)
    n = np.arange(lab.shape[0])
    cents = [lab[n, idx_dark], lab[n, idx_light]]
    mean = (lab * wgt_safe[..., None]).sum(1) / wgt_safe.sum(1, keepdims=True)
    while len(cents) < k:
        cents.append(mean)
    cent = np.stack(cents[:k], axis=1)                      # (N, k, 3)

    for _ in range(iters):
        d = ((lab[:, :, None, :] - cent[:, None, :, :]) ** 2).sum(-1)   # (N, P, k)
        assign = d.argmin(-1)
        onehot = (assign[..., None] == np.arange(k)) * wgt_safe[..., None]   # (N, P, k)
        cnt = onehot.sum(1)                                              # (N, k)
        new = (onehot[..., None] * lab[:, :, None, :]).sum(1) / np.maximum(cnt[..., None], 1e-6)
        cent = np.where(cnt[..., None] > 0, new, cent)
    best = cnt.argmax(-1)
    if prefer_dark is not None:                             # line-preserving variant: keep the dark cluster
        dark = cent[..., 0].argmin(-1)                      # (outline / line art) when it covers enough of the cell
        share = cnt[n, dark] / np.maximum(cnt.sum(-1), 1e-6)
        contrast = cent[..., 0].max(-1) - cent[..., 0].min(-1)
        keep = (share >= prefer_dark) & (contrast > 0.18)
        # a thin dark line on a much lighter area (mouth, nose, eyelid on skin) survives even when it covers
        # only a sliver of the cell; low-contrast lines (strands inside dark hair) need the full share
        keep |= (share >= strong_share) & (contrast > strong_contrast)
        best = np.where(keep, dark, best)
    out_lab = cent[n, best]
    out = np.zeros((h * w, 4), dtype=np.float32)
    out[:, :3] = oklab_to_rgb(out_lab)
    out[:, 3] = (coverage >= 0.5).astype(np.float32)
    out[out[:, 3] == 0, :3] = 0
    return _to_img(out.reshape(h, w, 4))


# -------------------------------------------------------------- palette ----

def build_palette(images: Image.Image | list[Image.Image], n: int = 24, iters: int = 24, seed: int = 0) -> np.ndarray:
    """k-means (k-means++ init) in OKLab over all opaque pixels of all images. Returns (n, 3) uint8 sRGB,
    sorted dark -> light. Build ONE palette per character and reuse it for every asset of that character."""
    if isinstance(images, Image.Image):
        images = [images]
    px = []
    for im in images:
        a = _rgba(im).reshape(-1, 4)
        px.append(a[a[:, 3] >= 0.5, :3])
    px = np.concatenate(px, 0)
    uniq, counts = np.unique((px * 255 + 0.5).astype(np.uint8), axis=0, return_counts=True)
    if len(uniq) <= n:
        pal = uniq
    else:
        lab = rgb_to_oklab(uniq.astype(np.float32) / 255.0)
        wts = counts.astype(np.float64)
        rng = np.random.default_rng(seed)
        cent = [lab[np.argmax(wts)]]
        d2 = ((lab - cent[0]) ** 2).sum(-1)
        for _ in range(1, n):                                   # weighted farthest-point / k-means++
            p = d2 * np.sqrt(wts)
            cent.append(lab[rng.choice(len(lab), p=p / p.sum())])
            d2 = np.minimum(d2, ((lab - cent[-1]) ** 2).sum(-1))
        cent = np.stack(cent)
        for _ in range(iters):
            assign = ((lab[:, None, :] - cent[None]) ** 2).sum(-1).argmin(-1)
            for j in range(n):
                m = assign == j
                if m.any():
                    cent[j] = (lab[m] * wts[m, None]).sum(0) / wts[m].sum()
        pal = (oklab_to_rgb(cent) * 255 + 0.5).astype(np.uint8)
    order = np.argsort(rgb_to_oklab(pal.astype(np.float32) / 255.0)[:, 0])
    return pal[order]


def apply_palette(img: Image.Image, palette: np.ndarray) -> Image.Image:
    a = _rgba(img)
    lab = rgb_to_oklab(a[..., :3])
    pal_lab = rgb_to_oklab(palette.astype(np.float32) / 255.0)
    idx = ((lab[:, :, None, :] - pal_lab[None, None]) ** 2).sum(-1).argmin(-1)
    out = a.copy()
    out[..., :3] = palette[idx].astype(np.float32) / 255.0
    out[out[..., 3] < 0.5, :3] = 0
    return _to_img(out)


def palette_image(palette: np.ndarray, swatch: int = 16) -> Image.Image:
    im = Image.new("RGB", (len(palette) * swatch, swatch))
    for i, c in enumerate(palette):
        im.paste(tuple(int(v) for v in c), (i * swatch, 0, (i + 1) * swatch, swatch))
    return im


# ---------------------------------------------------------------- alpha ----

def harden_alpha(img: Image.Image, threshold: float = 0.5) -> Image.Image:
    a = _rgba(img)
    a[..., 3] = (a[..., 3] >= threshold).astype(np.float32)
    a[a[..., 3] == 0, :3] = 0
    return _to_img(a)


def _neighbours4(mask: np.ndarray) -> np.ndarray:
    """count of opaque 4-neighbours for each pixel"""
    p = np.pad(mask.astype(np.int32), 1)
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def remove_specks(img: Image.Image) -> Image.Image:
    """Delete opaque pixels with no opaque 4-neighbour; fill transparent pinholes fully enclosed (4 neighbours)."""
    a = _rgba(img)
    m = a[..., 3] >= 0.5
    nb = _neighbours4(m)
    a[m & (nb == 0)] = 0
    holes = (~m) & (nb == 4)
    if holes.any():
        left = np.roll(a, 1, axis=1)
        a[holes] = left[holes]
    return _to_img(a)


def outline(img: Image.Image, color: tuple[int, int, int] | None = None, mode: str = "outer",
            darken: float = 0.45) -> Image.Image:
    """1px outline around the silhouette.

    mode="outer": paint transparent pixels touching the sprite (sprite grows by 1px each side).
    mode="inner": recolour the sprite's own border pixels (size unchanged).
    color=None  -> "selective outline": a darkened version of the adjacent/own colour, the look
                   used by RPG Maker MV/MZ stock sprites, instead of pure black.
    """
    a = _rgba(img)
    m = a[..., 3] >= 0.5
    if mode == "outer":
        p = np.pad(a, ((1, 1), (1, 1), (0, 0)))
        pm = np.pad(m, 1)
        acc = np.zeros_like(a[..., :3])
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            nm = pm[dy:dy + m.shape[0], dx:dx + m.shape[1]]
            acc += p[dy:dy + m.shape[0], dx:dx + m.shape[1], :3] * nm[..., None]
            cnt += nm
        edge = (~m) & (cnt > 0)
        col = acc / np.maximum(cnt[..., None], 1)
    else:
        edge = m & (_neighbours4(m) < 4)
        col = a[..., :3]
    if color is not None:
        a[edge, :3] = np.array(color, np.float32) / 255.0
    else:
        a[edge, :3] = (col * darken)[edge]
    a[edge, 3] = 1.0
    return _to_img(a)


# ------------------------------------------------------------- pipeline ----

def trim(img: Image.Image, pad: int = 0) -> Image.Image:
    """Crop to the opaque bounding box (+pad)."""
    bbox = img.convert("RGBA").getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox()
    if not bbox:
        return img
    l, t, r, b = bbox
    return img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))


def fit_canvas(img: Image.Image, w: int, h: int, anchor: str = "bottom", margin: int = 0) -> Image.Image:
    """Place img on a transparent (w, h) canvas, centred horizontally; anchor 'bottom' (feet) or 'center'."""
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = (w - img.width) // 2
    y = h - img.height - margin if anchor == "bottom" else (h - img.height) // 2
    canvas.alpha_composite(img.convert("RGBA"), (x, y))
    return canvas


def pixelize(img: Image.Image, w: int, h: int, *, palette: np.ndarray | None = None, colors: int | None = 24,
             outline_mode: str | None = None, outline_color: tuple[int, int, int] | None = None,
             k: int = 2, clean: bool = True) -> Image.Image:
    """High-res RGBA picture -> (w, h) pixel art. Pass a shared `palette` for cross-asset consistency."""
    out = kcentroid_downscale(img, w, h, k=k)
    if clean:
        out = remove_specks(out)
    if outline_mode:
        out = outline(out, outline_color, outline_mode)
    if palette is None and colors:
        palette = build_palette(out, colors)
    if palette is not None:
        out = apply_palette(out, palette)
    return out


def upscale(img: Image.Image, k: int = 4) -> Image.Image:
    return img.resize((img.width * k, img.height * k), Image.NEAREST)


def contact_sheet(images: list[Image.Image], cols: int, bg=(64, 64, 72, 255), pad: int = 4, scale: int = 1) -> Image.Image:
    """Preview grid for eyeballing results."""
    images = [upscale(i, scale) if scale > 1 else i for i in images]
    cw = max(i.width for i in images)
    ch = max(i.height for i in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * (cw + pad) + pad, rows * (ch + pad) + pad), bg)
    for n, im in enumerate(images):
        sheet.alpha_composite(im.convert("RGBA"), (pad + (n % cols) * (cw + pad), pad + (n // cols) * (ch + pad)))
    return sheet

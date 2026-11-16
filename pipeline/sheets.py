"""Assemble RPG Maker MV/MZ-layout sheets, a smooth N-frame walk sheet and a Phaser-friendly manifest.

RPG Maker MV/MZ conventions (also the de-facto layout of most asset packs)
    walk sheet : 3 columns x 4 rows of 48x48 frames  -> 144x192 for one character ("$name.png")
                 rows  = down, left, right, up        columns = step-A, idle, step-B
                 walk cycle played as 1,2,1,0 (idle, B, idle, A) or simply 0,1,2,1
    face sheet : 4 columns x 2 rows of 144x144 faces  -> 576x288, 8 faces
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

DIRS = ["down", "left", "right", "up"]            # RPG Maker row order
FRAME = 48
FACE = 144


def walk_sheet(frames: dict[str, list[Image.Image]], frame_w: int = FRAME, frame_h: int = FRAME) -> Image.Image:
    """frames = {"down": [stepA, idle, stepB], "left": [...], "right": [...], "up": [...]}"""
    sheet = Image.new("RGBA", (frame_w * 3, frame_h * 4), (0, 0, 0, 0))
    for r, d in enumerate(DIRS):
        for c, im in enumerate(frames[d]):
            assert im.size == (frame_w, frame_h), f"{d}[{c}] is {im.size}, expected {(frame_w, frame_h)}"
            sheet.paste(im.convert("RGBA"), (c * frame_w, r * frame_h))
    return sheet


def anim_sheet(rows: dict[str, list[Image.Image]], frame_w: int = FRAME, frame_h: int = FRAME) -> Image.Image:
    """Smooth walk sheet: one row per direction (down, left, right, up); column 0 = standing, columns 1..N =
    the N frames of one gait cycle in playing order."""
    n = len(rows[DIRS[0]])
    sheet = Image.new("RGBA", (frame_w * n, frame_h * 4), (0, 0, 0, 0))
    for r, d in enumerate(DIRS):
        for c, im in enumerate(rows[d]):
            sheet.paste(im.convert("RGBA"), (c * frame_w, r * frame_h))
    return sheet


def anim_manifest(image: str, columns: int, frame_rate: dict[str, float], frame_w: int = FRAME, frame_h: int = FRAME) -> dict:
    """frame_rate: frames per second per direction (the gait cycles of the four clips differ in length)."""
    anims = {}
    for r, d in enumerate(DIRS):
        base = r * columns
        anims[f"walk-{d}"] = {"frames": [base + c for c in range(1, columns)], "frameRate": frame_rate[d], "repeat": -1}
        anims[f"idle-{d}"] = {"frames": [base], "frameRate": 1, "repeat": 0}
    return {"image": image, "frameWidth": frame_w, "frameHeight": frame_h, "columns": columns, "rows": 4,
            "rowOrder": DIRS, "animations": anims}


def face_sheet(faces: list[Image.Image], size: int = FACE, cols: int = 4, rows: int = 2) -> Image.Image:
    sheet = Image.new("RGBA", (size * cols, size * rows), (0, 0, 0, 0))
    for i, im in enumerate(faces[: cols * rows]):
        assert im.size == (size, size), f"face {i} is {im.size}"
        sheet.paste(im.convert("RGBA"), ((i % cols) * size, (i // cols) * size))
    return sheet


def manifest(name: str, *, walk_png: str | None = None, faces_png: str | None = None,
             emotions: list[str] | None = None, frame_w: int = FRAME, frame_h: int = FRAME,
             face_size: int = FACE, frame_rate: int = 8, extra: dict | None = None) -> dict:
    """Engine-neutral description that a Phaser loader (see phaser-demo/) turns into
    this.load.spritesheet(...) + this.anims.create(...) calls."""
    m: dict = {"name": name, "format": "rpgmaker-mv"}
    if walk_png:
        anims = {}
        for r, d in enumerate(DIRS):
            base = r * 3
            anims[f"walk-{d}"] = {"frames": [base + 1, base + 2, base + 1, base + 0], "frameRate": frame_rate, "repeat": -1}
            anims[f"idle-{d}"] = {"frames": [base + 1], "frameRate": 1, "repeat": 0}
        m["walk"] = {"image": walk_png, "frameWidth": frame_w, "frameHeight": frame_h, "columns": 3, "rows": 4,
                     "rowOrder": DIRS, "animations": anims}
    if faces_png:
        m["faces"] = {"image": faces_png, "frameWidth": face_size, "frameHeight": face_size, "columns": 4,
                      "frames": {e: i for i, e in enumerate(emotions or [])}}
    if extra:
        m.update(extra)
    return m


def save_manifest(m: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(m, indent=2))

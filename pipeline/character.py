"""End-to-end character asset builder: text description -> concept -> chibi sprite views -> walk sheets + faces.

    python -m pipeline.character --name rin --seed 11 \
        --tags "1girl, solo, short messy red hair, green eyes, ..." \
        --desc "a young female adventurer with short messy red hair and green eyes, wearing ..." \
        --props "a blue scarf around the neck"            # optional: items that must never get lost

    stages   concept -> chibi (= front view) -> back -> [REVIEW "views"] -> turn_left, turn_right (side views)
             -> [REVIEW "sides"] -> walk_down, walk_left, walk_right, walk_up ;  concept -> face_neutral -> face_<emotion>

The build stops twice and prints a review request (JSON with an image to look at, a checklist and the
commands to continue): after the front and back views, and after the side views.

    python -m pipeline.character --name rin --redo back --note back="the scarf tails hang down the back"
    python -m pipeline.character --name rin --approve views       # continues until the side views
    python -m pipeline.character --name rin --pick left=28         # another frame of the turn as left view
    python -m pipeline.character --name rin --approve sides        # continues to the end

Every stage writes its raw model output to <out>/raw/ and is skipped when it already exists, so a run can be
resumed and single stages can be redone (`--redo walk_left`; everything derived from a stage is redone too).
Description, seed, notes and approvals are remembered in <out>/raw/state.json, so later calls only need
`--name`. Fixed seeds make reruns reproduce the same images on the same machine and model versions.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import pixelize as P
from . import prompts as PR
from . import sheets
from . import sprites as S
from . import video as V
from . import workflows as W
from .comfy import Comfy

DEFAULT_EMOTIONS = ["neutral", "happy", "angry", "sad", "surprised", "embarrassed", "determined", "hurt"]
ANIMA_AESTHETIC_NEG = "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration, multiple views, reference sheet"
ANIMA_CONCEPT = ("masterpiece, best quality, score_7, safe, anime, {tags}, full body, standing, straight-on, facing viewer, "
                 "looking at viewer, arms at sides, neutral expression, closed mouth, simple background, white background, "
                 "character design, flat color, clean lineart, anime coloring")
SIDES = ("left", "right")
GATES = {"views": ["chibi", "back"], "sides": ["stand_left", "stand_right"]}      # gate -> raw images it approves


@dataclass
class Spec:
    name: str
    tags: str = ""                 # danbooru-style tags for the Anima concept
    desc: str = ""                 # one prose sentence, pasted into the edit prompts
    seed: int = 1
    props: str = ""                # carried items, e.g. "a greatsword strapped to the back, hilt above the right shoulder"
    notes: dict[str, str] = field(default_factory=dict)       # stage -> correction added to that stage's prompt
    emotions: list[str] = field(default_factory=lambda: list(DEFAULT_EMOTIONS))
    extra_emotions: dict[str, str] = field(default_factory=dict)   # name -> expression description
    key: str | None = None         # chroma key colour: green | magenta | blue | None = auto
    max_retries: int = 2           # extra attempts per stage when the automatic checks fail
    review: tuple[str, ...] = ("views", "sides")   # gates at which the build stops until approved
    face_grid: int = 2             # logical pixel size inside the 144px face (1 smooth, 2 pixel art, 3 chunky)
    face_zoom: float = 0.9
    face_style: str = "lines"       # "lines" (k-centroid that keeps line art) | "smooth" (k-centroid) | "crisp" (nearest)
    pixel_style: str = "crisp"     # "crisp" (nearest, fine detail) | "smooth" (k-centroid) | "classic" (k-centroid + outline, 24 colours)
    sprite_colors: int = 64
    face_colors: int = 64
    walk_frames: int = 8            # frames per gait cycle in the smooth walk sheet (<name>_walk<N>.png)
    video_fast: bool = True         # lightx2v 4-step LoRAs: about 2 min per clip, steadier walk; False = 20 steps, about 13 min
    video_size: int = 640           # clip resolution (square)
    video_length: int = 81          # walk clip: frames at 16 fps; must be 4n+1
    turn_length: int = 49           # half-turn clip (front -> back)
    concept_image: str | None = None  # start from an existing concept image instead of generating one
    edit_model: str = "qwen21"      # "qwen21" (best quality, ~60 s per image) | "flux3" (fast, face artifacts)
    concept_model: str = "turbo"    # Anima "turbo" (v1.1, detailed) | "aesthetic" (v1.1, flatter, 30 steps)

    SOURCE = ("tags", "desc", "seed", "props", "notes", "key")     # remembered in raw/state.json


class Builder:
    def __init__(self, spec: Spec, out: str | Path, comfy: Comfy | None = None):
        self.s, self.out, self.c = spec, Path(out), comfy or Comfy()
        self.raw = self.out / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.redo: set[str] = set()
        self.state = {"source": {}, "variants": {}, "stages": {}, "approved": {}, "pending": []}
        sp = self.raw / "state.json"
        if sp.exists():
            self.state.update(json.loads(sp.read_text()))
        if (self.raw / "face_neutral.png").exists() and not (self.raw / "portrait.png").exists():
            # built before the portrait stage: the old neutral face is the portrait; the neutral tile is made anew
            (self.raw / "face_neutral.png").rename(self.raw / "portrait.png")
            for key in ("stages", "variants"):
                if "face_neutral" in self.state[key]:
                    self.state[key]["portrait"] = self.state[key].pop("face_neutral")
        # stages still to be redone from an earlier call that stopped at a review gate
        self.redo = set(self.state.get("pending", []))

    def _log(self, msg: str) -> None:
        """Progress on stderr (the JSON result goes to stdout)."""
        print(f"[{time.strftime('%H:%M:%S')}] {self.s.name}: {msg}", file=sys.stderr, flush=True)

    # ---- state --------------------------------------------------------------
    def save_state(self) -> None:
        self.state["source"] = {k: getattr(self.s, k) for k in Spec.SOURCE}
        (self.raw / "state.json").write_text(json.dumps(self.state, indent=1))

    def set_redo(self, stages: list[str], variant: int | None = None) -> None:
        """Redoing a stage also redoes everything derived from it. The stages named here get a new seed: `variant`
        if given, otherwise one more than their stored variant. Derived stages keep theirs."""
        redo = set(stages)
        walks, faces = {f"walk_{d}" for d in sheets.DIRS}, {f"face_{e}" for e in self.s.emotions}
        if "concept" in redo:
            redo |= {"chibi", "portrait"}
        if "chibi" in redo:
            redo |= {"back"}
        if "back" in redo:
            redo |= {"turn_left", "turn_right", "walk_down", "walk_up"}
        for side in SIDES:
            if f"turn_{side}" in redo:
                redo |= {f"walk_{side}"}
        if "portrait" in redo:
            redo |= faces
        unknown = redo - ({"concept", "chibi", "back", "turn_left", "turn_right", "portrait"} | walks | faces)
        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")
        self.redo = self.redo | redo
        self.state["pending"] = sorted(self.redo)
        for st in stages:                                   # a new take only for the stages named explicitly;
            old = self.state["variants"].get(st, 0)          # derived stages keep their seeds (their input changed)
            self.state["variants"][st] = old + 1 if variant is None else variant
            if variant is not None and variant == old:
                print(f"[note] {st} already used variant {variant}: same seed as before", flush=True)

    def _seed(self, stage: str, base: int, attempt: int = 0) -> int:
        return self.s.seed + base + attempt * 101 + self.state["variants"].get(stage, 0) * 7919

    def _is_clip(self, stage: str) -> bool:
        return stage.startswith(("walk_", "turn_"))

    def _need(self, stage: str) -> bool:
        if stage in self.redo:
            return True
        if self._is_clip(stage):                                # a clip = a folder of frames
            return not any((self.raw / stage).glob("*.png"))
        return not (self.raw / f"{stage}.png").exists()

    def _done(self, stage: str, **info) -> None:
        self.redo.discard(stage)
        self.state["pending"] = sorted(self.redo)
        self.state["stages"][stage] = info
        self.save_state()

    # ---- image stages ---------------------------------------------------------
    def _gen(self, stage: str, graph, seed: int, attempt: int = 0, **info) -> Image.Image:
        self._log(f"{stage} (seed {seed}) ...")
        t = time.time()
        tmp = self.c.run(graph, self.raw, stem=f"{stage}__tmp")[0]
        dst = self.raw / f"{stage}.png"
        tmp.replace(dst)
        self._done(stage, seconds=round(time.time() - t, 1), attempts=attempt + 1, seed=seed, **info)
        return Image.open(dst)

    def _ref(self, stage: str) -> str:
        return self.c.upload_image(self.raw / f"{stage}.png", f"ccr_{self.s.name}_{stage}.png")

    def _edit(self, kind: str, stage: str, refs: list[str], seed: int, **fmt):
        """Build the edit graph for the configured edit model."""
        s = self.s
        if kind in ("chibi", "back", "side"):
            fmt.update(desc=s.desc, props=PR.props_sentence(s.props), note=PR.note_sentence(s.notes.get(stage, "")),
                       bg=S.KEY_PROMPT[s.key])
        prompt = PR.PROMPTS[s.edit_model][kind].format(**fmt)
        if kind == "side" and s.edit_model != "qwen21":
            raise SystemExit("side views by edit are only implemented for --edit-model qwen21")
        fn = W.qwen21_edit if s.edit_model == "qwen21" else W.flux3_edit
        return fn(prompt, refs, width=1024, height=1024, seed=seed, prefix=f"ccr/{s.name}_{stage}")

    def concept(self) -> None:
        s = self.s
        if s.concept_image:
            if self._need("concept"):
                Image.open(s.concept_image).convert("RGB").save(self.raw / "concept.png")
                self._done("concept", source=s.concept_image)
        elif self._need("concept"):
            for attempt in range(s.max_retries + 1):
                seed = self._seed("concept", 0, attempt) - attempt * 100      # seed, seed+1, seed+2
                if s.concept_model == "aesthetic":       # no score_* tags on Aesthetic
                    g = W.anima_t2i(ANIMA_CONCEPT.format(tags=s.tags).replace("score_7, ", ""), negative=ANIMA_AESTHETIC_NEG,
                                    seed=seed, unet="anima-aesthetic-v1.1.safetensors", steps=30, cfg=4.0,
                                    prefix=f"ccr/{s.name}_concept")
                else:
                    g = W.anima_t2i(ANIMA_CONCEPT.format(tags=s.tags), seed=seed, prefix=f"ccr/{s.name}_concept")
                n = S.count_figures(self._gen("concept", g, seed, attempt, model=s.concept_model))
                self.state["stages"]["concept"]["figures"] = n
                if n == 1:
                    break
        if not s.key:                                          # background colour farthest from the character's colours
            s.key = S.choose_key_color(Image.open(self.raw / "concept.png"))
        self.save_state()

    def chibi(self) -> None:
        """Chibi on the key background; it is also the front standing view of the sprite."""
        if self._need("chibi"):
            seed = self._seed("chibi", 1000)
            self._gen("chibi", self._edit("chibi", "chibi", [self._ref("concept")], seed), seed)

    def back(self) -> None:
        """Back standing view, drawn from the front view. Judged by the reviewer at the `views` gate: a colour test
        for "still shows a face" raised false alarms on 3 of 5 characters (beige headwrap, blanket roll, dark skin,
        red hair) and never caught a real error, so there is none."""
        if self._need("back"):
            seed = self._seed("back", 2000)
            self._gen("back", self._edit("back", "back", [self._ref("chibi")], seed), seed, problems=[])

    # ---- standing views in clip space -----------------------------------------
    def _stands(self) -> dict:
        """Square start frames for the video model from the front and back views (one scale, flat key
        background) -> raw/stand_down.png, raw/stand_up.png. Returns {"cell": source px per sprite px, ...}."""
        views = {"down": Image.open(self.raw / "chibi.png"), "up": Image.open(self.raw / "back.png")}
        frames, meta = V.start_frames(views, self.s.video_size)
        for d, im in frames.items():
            im.save(self.raw / f"stand_{d}.png")
        return meta

    def _clip(self, prompt: str, start: str, end: str, length: int, seed: int, prefix: str, tmp: Path, fast: bool = True):
        self._log(f"{prefix.split('_', 1)[-1]} clip (seed {seed}) ...")
        shutil.rmtree(tmp, ignore_errors=True)
        g = W.wan22_i2v(prompt, start, end=end, width=self.s.video_size, height=self.s.video_size, length=length,
                        seed=seed, fast=fast, negative=W.WAN_NEG + PR.WAN_NEG_EXTRA, prefix=prefix)
        return self.c.run_video(g, lambda lat: W.wan_decode(lat, prefix=prefix), tmp)

    def _keep_clip(self, files: list[Path], tmp: Path, stage: str) -> None:
        final = self.raw / stage
        shutil.rmtree(final, ignore_errors=True)
        for f in files:
            if f.suffix == ".webp":
                f.replace(self.raw / f"{stage}.webp")           # lossy preview of the whole clip
        tmp.rename(final)
        for i, f in enumerate(sorted(final.glob("*.png")), 1):   # ComfyUI numbers on from earlier runs
            f.rename(final / f"frame_{i:03d}.png")

    # -- side views: Wan turns the character from the front view (first frame) to the back view (last frame);
    #    the side view is the narrowest moment of that turn. A clip is filed under the side it really shows.
    def turns(self) -> None:
        s = self.s
        need = [side for side in SIDES if self._need(f"turn_{side}")]
        if not need:
            return
        # a side that already needed the edit fallback goes there directly on a redo (the clips will not turn that way)
        def failed_before(x):
            st = self.state["stages"].get(f"turn_{x}", {})
            return st.get("source") == "edit" or any("no clip turned this way" in p for p in st.get("problems", []))
        for side in [x for x in need if failed_before(x)]:
            self._side_by_edit(side, ["earlier clips did not show this side"])
            need.remove(side)
        if not need:
            return
        self._stands()
        bg = V.border_color(Image.open(self.raw / "stand_down.png"))
        ends = {}
        for mirrored in (False, True):                          # front / back as given and flipped left-right
            for d in ("down", "up"):
                im = Image.open(self.raw / f"stand_{d}.png")
                p = self.raw / f"stand_{d}__m.png"
                (ImageOps.mirror(im) if mirrored else im).save(p)
                ends[d, mirrored] = self.c.upload_image(p, f"ccr_{s.name}_stand_{d}{'_m' if mirrored else ''}.png")
                p.unlink()
        # Which way the model turns cannot be steered by the prompt, and for a given pair of images it tends to
        # turn the same way every time. Flipping both images left-right flips that habit: the flipped clip,
        # flipped back, shows the true other side. Each clip is filed under the side it really shows.
        plan = [("front_back", False), ("front_back", True), ("back_front", False), ("back_front", True)]
        for side in SIDES:                                      # one side exists already: start with its opposite
            st = self.state["stages"].get(f"turn_{side}", {})
            if side not in need and "order" in st:
                o, m = st["order"], st.get("mirrored_input", False)
                o2 = "back_front" if o == "front_back" else "front_back"
                plan = [(o, not m), (o2, not m), (o2, m), (o, m)]
        done, tmp, failures = set(), self.raw / "turn__tmp", []
        for k in range(2 * (s.max_retries + 1)):
            missing = [side for side in need if side not in done]
            if not missing:
                break
            order, mirrored = plan[k % len(plan)]
            seed = self._seed(f"turn_{missing[0]}", 5000, k // 2)
            prompt = PR.TURN_VIDEO.format(order=PR.TURN_ORDER[order], bg=s.key, props=PR.props_sentence(s.props))
            t = time.time()
            first, last = (ends["down", mirrored], ends["up", mirrored])[::1 if order == "front_back" else -1]
            files = self._clip(prompt, first, last, s.turn_length, seed, f"ccr/{s.name}_turn", tmp)
            if mirrored:                                        # store the clip the right way round
                pngs = sorted(tmp.glob("*.png"))
                for f in pngs:
                    ImageOps.mirror(Image.open(f)).save(f)
                for f in files:
                    if f.suffix == ".webp":
                        V.save_anim([Image.open(q).convert("RGBA") for q in pngs], f, fps=16, bg=bg, lossless=False)
            keyed = V.key_frames(V.load_frames(tmp))
            if order == "back_front":
                keyed = keyed[::-1]                             # analyse it as a turn from front to back
            r = V.turn_profile(keyed)
            got = r["facing"] or (missing[0] if not r["problems"] else None)   # no skin to judge by: take it as asked
            how = f"{order}{', mirrored input' if mirrored else ''}, seed {seed}"
            if r["problems"] or got not in missing:
                failures.append(f"{how}: " + ("; ".join(r["problems"]) or f"faces {got} again"))
                keep = self.raw / "turn_tries"
                keep.mkdir(exist_ok=True)
                for f in files:
                    if f.suffix == ".webp":
                        f.replace(keep / f"{order}{'_mirrored' if mirrored else ''}_seed{seed}.webp")
                continue
            flat = Image.new("RGB", keyed[0].size, bg)           # the picked frame on a perfectly flat background
            flat.paste(keyed[r["index"]], (0, 0), keyed[r["index"]])
            flat.save(self.raw / f"stand_{got}.png")
            self._keep_clip(files, tmp, f"turn_{got}")
            done.add(got)
            frame = (r["index"] if order == "front_back" else len(keyed) - 1 - r["index"]) + 1   # as in the file names
            self._done(f"turn_{got}", seconds=round(time.time() - t, 1), seed=seed, order=order, mirrored_input=mirrored,
                       frame=frame, problems=[])
        shutil.rmtree(tmp, ignore_errors=True)
        for side in need:                                       # fallback: the edit model draws the side view
            if side in done:
                continue
            self._side_by_edit(side, failures)

    def _side_by_edit(self, side: str, failures: list[str]) -> None:
        """No turn clip showed this side (for some asymmetric characters the video model always turns the same
        way - experiment 19, Kaela). The image edit model draws it from the approved front and back views."""
        s = self.s
        meta = self._stands()
        seed = self._seed(f"turn_{side}", 5500)
        t = time.time()
        g = self._edit("side", f"turn_{side}", [self._ref("chibi"), self._ref("back")], seed,
                       FACING=side.upper(), facing=side)
        img = Image.open(self.c.run(g, self.raw, stem=f"side_{side}__tmp")[0])
        img.save(self.raw / f"side_{side}.png")
        (self.raw / f"side_{side}__tmp.png").unlink(missing_ok=True)
        frames, _ = V.start_frames({"down": Image.open(self.raw / "chibi.png"), side: img}, s.video_size,
                                   scale=meta["scale"])
        frames[side].save(self.raw / f"stand_{side}.png")
        shutil.rmtree(self.raw / f"turn_{side}", ignore_errors=True)
        self._done(f"turn_{side}", seconds=round(time.time() - t, 1), seed=seed, source="edit", tries=failures,
                   problems=[f"no turn clip showed this side ({len(failures)} tried, previews in raw/turn_tries/); "
                             f"drawn by the image edit model instead - check facing and items carefully"])

    def pick(self, side: str, frame: int) -> None:
        """Reviewer's choice: use frame number `frame` (1-based, as in the file names and the review sheet) of
        the stored turn clip as the side view. The walk clip of that side is redone."""
        files = sorted((self.raw / f"turn_{side}").glob("*.png"))
        if side not in SIDES:
            raise SystemExit(f"--pick: side must be left or right, not '{side}'")
        if len(files) < 2:
            raise SystemExit(f"--pick {side}: this side view was drawn by the edit model, not taken from a turn clip, so "
                             f"there are no frames to pick from. Use --redo turn_{side} (a new drawing), optionally with "
                             f"--note turn_{side}=\"...\"")
        if not 1 <= frame <= len(files):
            raise SystemExit(f"frame must be 1..{len(files)}")
        keyed = V.key_frames([Image.open(files[frame - 1]).convert("RGB")])[0]
        flat = Image.new("RGB", keyed.size, V.border_color(Image.open(self.raw / "stand_down.png")))
        flat.paste(keyed, (0, 0), keyed)
        flat.save(self.raw / f"stand_{side}.png")
        self.state["stages"].setdefault(f"turn_{side}", {}).update(frame=frame, picked_by="reviewer")
        self.redo.add(f"walk_{side}")
        self.state["pending"] = sorted(self.redo)
        self.save_state()

    # -- walk clips: Wan animates each standing view; first frame = last frame = the standing view -----
    def walk(self) -> None:
        s = self.s
        meta = self._stands()
        for d in sheets.DIRS:
            stage = f"walk_{d}"
            if not self._need(stage):
                continue
            start = self.c.upload_image(self.raw / f"stand_{d}.png", f"ccr_{s.name}_stand_{d}.png")
            prompt = PR.WALK_VIDEO.format(view=PR.VIDEO_VIEW[d], bg=s.key, desc=s.desc, props=PR.props_sentence(s.props))
            tmp, best, tries = self.raw / f"{stage}__tmp", None, []
            for attempt in range(s.max_retries + 1):
                seed = self._seed(stage, 4000, attempt)
                t = time.time()
                files = self._clip(prompt, start, start, s.video_length, seed, f"ccr/{s.name}_{stage}", tmp, s.video_fast)
                a = V.analyse_clip(tmp, meta["cell"])
                tries.append({"seed": seed, "problems": a["problems"]})
                if a["problems"]:
                    self._log(f"{stage}: {'; '.join(a['problems'])} - trying again" if attempt < s.max_retries else
                              f"{stage}: {'; '.join(a['problems'])} - keeping the best take")
                score = (len(a["problems"]), a["cycle"]["seam"])
                if best is None or score < best[0]:
                    self._keep_clip(files, tmp, stage)
                    best = (score, dict(seconds=round(time.time() - t, 1), seed=seed, problems=a["problems"],
                                        cycle={k: a["cycle"][k] for k in ("start", "period", "seam", "head_error", "upper_lost", "new_colours")}))
                if not a["problems"]:
                    break
            shutil.rmtree(tmp, ignore_errors=True)
            self._done(stage, attempts=attempt + 1, tries=tries, **best[1])

    def faces(self) -> None:
        """portrait = head-and-shoulders edit of the concept, the shared reference; every face tile, the neutral
        one included, is an expression edit of it. (The neutral tile used to be the portrait itself and looked
        softer than the others, which had been through one more edit.)"""
        s, bg = self.s, S.KEY_PROMPT[self.s.key]
        table = {**PR.EMOTIONS, **s.extra_emotions}
        if self._need("portrait"):
            seed = self._seed("portrait", 3000)
            self._gen("portrait", self._edit("portrait", "portrait", [self._ref("concept")], seed,
                                             expression=table["neutral"], bg=bg), seed)
        ref = None
        for e in s.emotions:
            if not self._need(f"face_{e}"):
                continue
            ref = ref or self._ref("portrait")
            seed = self._seed(f"face_{e}", 3000)
            self._gen(f"face_{e}", self._edit("expression", f"face_{e}", [ref], seed, expression=table[e]), seed)

    # ---- review gates ---------------------------------------------------------
    def _hashes(self, gate: str) -> dict:
        return {f: hashlib.sha1((self.raw / f"{f}.png").read_bytes()).hexdigest() for f in GATES[gate]}

    def approve(self, gate: str) -> None:
        missing = [f for f in GATES[gate] if not (self.raw / f"{f}.png").exists()]
        if missing:
            raise SystemExit(f"cannot approve '{gate}': {', '.join(missing)} not generated yet")
        self.state["approved"][gate] = self._hashes(gate)
        self.save_state()

    def approved(self, gate: str) -> bool:
        return self.state["approved"].get(gate) == self._hashes(gate)

    def review(self, gate: str) -> dict:
        """Write <name>_review_<gate>.png and return the review request for a human or a vision model."""
        s = self.s
        if gate == "views":
            tiles = [("CONCEPT (the design)", "concept"), ("FRONT VIEW (chibi)", "chibi"), ("BACK VIEW", "back")]
            checks = ["FRONT: exactly one character, whole body inside the picture, standing straight, arms hanging down "
                      "(a hand holding an item is fine)",
                      "FRONT: hair, face, outfit and colours match the concept",
                      "BACK: a true back view - no face, back of the hair or headwear, back of the outfit",
                      "BACK: same character, same outfit pieces and colours, same size as the front view",
                      "BACK: nothing invented (no extra bag, weapon, tail or wings the character does not have)",
                      "BOTH: flat single-colour background, nothing cut off at the picture edge"]
            if s.props:
                checks.insert(2, f"FRONT and BACK: every carried item is there and complete: {s.props}")
                checks.insert(3, "BACK: items keep their side - what is on the character's right is on the RIGHT of the back "
                                 "view; an item worn on the back lies on top of cape and hair, fully visible")
            redo = {"chibi": "front view wrong (the back view is redone too)", "back": "only the back view wrong"}
        else:
            tiles = [("FRONT", "stand_down"), ("LEFT (must face left)", "stand_left"),
                     ("RIGHT (must face right)", "stand_right"), ("BACK", "stand_up")]
            checks = ["LEFT faces the left edge and RIGHT faces the right edge, both in clean profile (one eye visible)",
                      "the side views show the same character as front and back: same outfit, colours and size",
                      "carried items are visible in the side views where they should be, with the same shape as in "
                      "front and back" + (f": {s.props}" if s.props else ""),
                      "no smeared or half-transparent parts, nothing cut off"]
            redo = {"turn_left": "left view wrong", "turn_right": "right view wrong"}
        h, pad, top = (640 if gate == "views" else 480), 12, 26
        ims = [Image.open(self.raw / f"{f}.png").convert("RGB") for _, f in tiles]
        ims = [im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
        strips = []
        if gate == "sides":                                     # neighbours of the chosen frame, to pick a better one
            for side in SIDES:
                files = sorted((self.raw / f"turn_{side}").glob("*.png"))
                cur = self.state["stages"].get(f"turn_{side}", {}).get("frame")
                if len(files) < 2 or not cur:
                    continue
                lo = min(max(1, cur - 5), max(1, len(files) - 10))
                order = self.state["stages"].get(f"turn_{side}", {}).get("order", "front_back")
                toward = "the BACK view" if order == "front_back" else "the FRONT view"
                strips.append((side, cur, toward, [(n, Image.open(files[n - 1]).convert("RGB").resize((228, 228), Image.LANCZOS))
                                                   for n in range(lo, min(len(files), lo + 10) + 1)]))
            checks.append("each strip shows the frames around the chosen side view (marked); higher numbers are closer to "
                          "the view named in the strip title. If another frame is a better side view - clean profile with one "
                          "eye visible, items solid and on the right side - take it with --pick <side>=<number>. Open "
                          f"{self.raw}/turn_<side>/frame_NNN.png at full size to compare candidates; all frames of the turn "
                          "are there, not only the ones in the strip")
            checks.append("a flat item (backpack, shield, cape) looks thin from the side - that is correct, not a defect")
        width = max(sum(im.width for im in ims) + pad * (len(ims) + 1), 11 * (228 + 8) + pad * 2)
        sheet = Image.new("RGB", (width, h + top + pad + len(strips) * (228 + top + pad)), (24, 24, 28))
        dr, x = ImageDraw.Draw(sheet), pad
        for (label, _), im in zip(tiles, ims):
            dr.text((x, 7), label, fill=(235, 235, 240))
            sheet.paste(im, (x, top))
            x += im.width + pad
        y = h + top + pad
        for side, cur, toward, frames in strips:
            dr.text((pad, y + 7), f"TURN CLIP FOR THE {side.upper()} VIEW - chosen: frame {cur} - higher numbers turn "
                                  f"towards {toward}", fill=(235, 235, 240))
            for i, (n, im) in enumerate(frames):
                x = pad + i * (228 + 8)
                sheet.paste(im, (x, y + top))
                if n == cur:
                    dr.rectangle((x - 3, y + top - 3, x + 230, y + top + 230), outline=(255, 230, 60), width=3)
                dr.rectangle((x, y + top, x + 70, y + top + 14), fill=(24, 24, 28))
                dr.text((x + 3, y + top + 1), f"frame {n}", fill=(255, 230, 60) if n == cur else (235, 235, 240))
            y += 228 + top + pad
        path = self.out / f"{s.name}_review_{gate}.png"
        sheet.save(path)
        cmd = f"python3 -m pipeline.character --name {s.name}"
        if self.out.resolve() != (Path("characters") / s.name).resolve():
            cmd += f" --out {self.out}"
        warn = [f"{st}: {p}" for st in redo for p in self.state["stages"].get(st, {}).get("problems", [])]
        return {"status": "review_needed", "gate": gate, "image": str(path),
                "instructions": "Open the image at full size and check every point. Approve only if all are true.",
                "checklist": checks, "automatic_warnings": warn,
                "if_ok": f"{cmd} --approve {gate}",
                **({"if_another_frame_is_better": f"{cmd} --pick left=<frame> --pick right=<frame>"} if gate == "sides" else {}),
                "if_wrong": {st: (f'{cmd} --redo {st}' + (f' --note {st}="<what must be different>"'
                                                          if gate == "views" else "") + f"   # {why}")
                             for st, why in redo.items()}}

    # ---- post-processing (pure Python, deterministic) ------------------------
    def post(self) -> dict:
        s, out = self.s, self.out
        # walk sheets. Every clip is keyed, locked to the pixel grid of its standing first frame and searched
        # for one gait cycle; that cycle is sampled for the 3-frame RPG Maker row and the smooth N-frame row.
        method = "nearest" if s.pixel_style == "crisp" else "kcentroid"
        classic = s.pixel_style == "classic"
        n_sprite, n_face = (24, 32) if classic else (s.sprite_colors, s.face_colors)
        cell = self._stands()["cell"]
        nw = s.walk_frames
        frames, anim, walk_qc, rate = {}, {}, {}, {}
        for d in sheets.DIRS:
            a = V.analyse_clip(self.raw / f"walk_{d}", cell)
            cyc, locked = a["cycle"], a["locked"]
            st, per = cyc["start"], cyc["period"]
            sel, idx = V.sample_cycle(locked, st, per, nw)
            px = V.locked_sprite_frames([locked[0], locked[st], locked[st + per // 2]] + sel, cell, a["anchor"], method=method)
            frames[d] = [px[1], px[0], px[2]]                    # step A, standing, step B
            anim[d] = [px[0]] + px[3:]                            # standing, then the cycle
            rate[d] = round(nw * 16 / per, 2)                     # clips are 16 fps: play the cycle at its real speed
            walk_qc[d] = {"problems": a["problems"], "cycle_start": st, "cycle_frames": per, "loop_jump": round(cyc["seam"], 4),
                          "frames_used": idx, "drift_removed_px": [a["lock"]["drift_x"], a["lock"]["drift_y"]]}
        if classic:
            frames = {d: [P.outline(f, None, "inner") for f in frames[d]] for d in sheets.DIRS}
            anim = {d: [P.outline(f, None, "inner") for f in anim[d]] for d in sheets.DIRS}
        problems = [f"walk_{d}: {p}" for d in sheets.DIRS for p in walk_qc[d]["problems"]]
        reviewed = []                                            # warnings of views a reviewer has approved
        for gate, stages in (("views", ["back"]), ("sides", ["turn_left", "turn_right"])):
            for st in stages:
                msgs = [f"{st}: {p}" for p in self.state["stages"].get(st, {}).get("problems", [])]
                (reviewed if self.approved(gate) else problems).extend(msgs)
        generated_qc = {"ok": not problems, "problems": problems, "accepted_at_review": reviewed}
        pal = P.build_palette([f for d in sheets.DIRS for f in frames[d] + anim[d]], n_sprite)
        frames = {d: [P.apply_palette(f, pal) for f in frames[d]] for d in sheets.DIRS}
        anim = {d: [P.apply_palette(f, pal) for f in anim[d]] for d in sheets.DIRS}
        sheets.walk_sheet(frames).save(out / f"{s.name}_walk.png")
        sheets.anim_sheet(anim).save(out / f"{s.name}_walk{nw}.png")
        P.palette_image(pal).save(out / f"{s.name}_palette.png")
        for d in sheets.DIRS:                                    # looping previews, 4x
            V.save_anim(anim[d][1:], out / f"{s.name}_walk_{d}.gif", fps=rate[d], scale=4)

        # faces: one crop box and one palette for all emotions
        keyed = {e: S.key_background(Image.open(self.raw / f"face_{e}.png")) for e in s.emotions}
        box = S.face_crop_box(keyed["neutral"], s.face_zoom)
        crops = {e: im.crop(box) for e, im in keyed.items()}
        n = sheets.FACE // s.face_grid
        fmethod = {"lines": "kcentroid_lines", "smooth": "kcentroid", "crisp": "nearest"}[s.face_style]
        small = {e: P.remove_specks(S.downscale(im, n, n, fmethod)) for e, im in crops.items()}
        fpal = P.build_palette(list(small.values()), n_face)
        tiles = [P.upscale(P.apply_palette(small[e], fpal), s.face_grid) for e in s.emotions]
        for i in range(0, len(tiles), 8):
            suffix = "" if i == 0 else f"_{i // 8 + 1}"
            sheets.face_sheet(tiles[i:i + 8]).save(out / f"{s.name}_faces{suffix}.png")
        busts = out / "busts"
        busts.mkdir(exist_ok=True)
        for e, im in keyed.items():                              # high-res cut-outs for menus / cut-ins
            P.harden_alpha(im).save(busts / f"{s.name}_{e}.png")
        Image.open(self.raw / "concept.png").save(out / f"{s.name}_concept.png")
        for d in sheets.DIRS:                                    # the four standing views, transparent
            P.harden_alpha(S.key_background(Image.open(self.raw / f"stand_{d}.png"))).save(out / f"{s.name}_stand_{d}.png")

        m = sheets.manifest(s.name, walk_png=f"{s.name}_walk.png", faces_png=f"{s.name}_faces.png",
                            emotions=s.emotions[:8])
        m["walkAnim"] = sheets.anim_manifest(f"{s.name}_walk{nw}.png", nw + 1, rate)
        m["busts"] = {e: f"busts/{s.name}_{e}.png" for e in s.emotions}
        m["views"] = {d: f"{s.name}_stand_{d}.png" for d in sheets.DIRS}
        m["source"] = {"tags": s.tags, "desc": s.desc, "props": s.props, "notes": s.notes, "seed": s.seed,
                       "key_color": s.key, "variants": self.state["variants"], "edit_model": s.edit_model,
                       "concept_model": s.concept_model, "pixel_style": s.pixel_style,
                       "walk_model": "wan2.2-i2v-14B" + (" + lightx2v 4-step" if s.video_fast else "")}
        m["qc"] = {"generated_walk": generated_qc, "walk_clips": walk_qc, "stages": self.state["stages"],
                   "approved": sorted(g for g in GATES if self.state["approved"].get(g))}
        sheets.save_manifest(m, out / f"{s.name}.json")

        # register in <out>/../index.json (read by phaser-demo)
        idx = out.parent / "index.json"
        entries = json.loads(idx.read_text()) if idx.exists() else []
        entry = f"{out.name}/{s.name}.json"
        if entry not in entries:
            idx.write_text(json.dumps(sorted(entries + [entry]), indent=2))

        # review sheet: standing sprite + every frame of the smooth cycle, 4x
        V.strip([f for d in sheets.DIRS for f in anim[d]], scale=4, cols=nw + 1).save(out / f"{s.name}_walk_review.png")

        # human/agent preview
        prev = [S.frames_preview(frames, 4), P.contact_sheet(tiles, 4, bg=(40, 40, 48, 255), pad=4, scale=1)]
        P.contact_sheet(prev, 2, bg=(24, 24, 28, 255), pad=8).save(out / f"{s.name}_preview.png")
        return {"status": "done", "out": str(out), "manifest": f"{s.name}.json", "walk": m["walk"]["image"],
                "walk_smooth": m["walkAnim"]["image"], "faces": m["faces"]["image"],
                "look_at": [str(out / f"{s.name}_walk_review.png"), str(out / f"{s.name}_preview.png")],
                "qc_ok": generated_qc["ok"], "problems": generated_qc["problems"]}

    def build(self) -> dict:
        self.concept()
        self.chibi()
        self.back()
        if "views" in self.s.review and not self.approved("views"):
            return self.review("views")
        self.turns()
        if "sides" in self.s.review and not self.approved("sides"):
            return self.review("sides")
        self.walk()
        self.faces()
        return self.post()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", default=None, help="danbooru-style tags for the concept (not needed with --concept-image)")
    ap.add_argument("--desc", default=None, help="one prose sentence describing hair, eyes, outfit and colours "
                                                 "(needed once; remembered afterwards)")
    ap.add_argument("--out", default=None, help="output folder (default characters/<name>)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--props", default=None, help='carried items that must show in every view, e.g. "a greatsword strapped '
                                                  'diagonally across the back, hilt above the right shoulder"')
    ap.add_argument("--note", action="append", default=[], metavar="STAGE=TEXT",
                    help="correction added to the prompt of one stage (chibi, back, or turn_left/turn_right when that "
                         "side view was drawn by the edit model); STAGE= clears it")
    ap.add_argument("--redo", default="", help="comma-separated stages to regenerate: concept chibi back turn_left "
                                               "turn_right walk_down walk_left walk_right walk_up face_<emotion>")
    ap.add_argument("--variant", type=int, default=None, help="with --redo: seed variant for the named stages (default: one "
                                                         "more than last time, so every --redo is a new take)")
    ap.add_argument("--approve", default="", help="comma-separated review gates to approve: views, sides")
    ap.add_argument("--review", default="views,sides", help="gates at which the build stops for approval: views, sides, "
                                                            "views,sides or none (default: views,sides)")
    ap.add_argument("--pick", action="append", default=[], metavar="SIDE=FRAME",
                    help="use another frame of the turn clip as side view, e.g. left=28 (numbers as in the review sheet)")
    ap.add_argument("--post-only", action="store_true", help="re-run only the pixel post-processing")
    ap.add_argument("--emotions", default=",".join(DEFAULT_EMOTIONS))
    ap.add_argument("--key", choices=["green", "magenta", "blue"], default=None)
    ap.add_argument("--face-grid", type=int, default=None, choices=[1, 2, 3])
    ap.add_argument("--pixel-style", default=None, choices=["crisp", "smooth", "classic"],
                    help="crisp: nearest sampling, 64 colours | smooth: k-centroid, 64 colours | classic: outline, 24 colours")
    ap.add_argument("--face-style", default=None, choices=["lines", "smooth", "crisp"])
    ap.add_argument("--face-zoom", type=float, default=None, help="share of the portrait the face tile shows: larger = "
                    "more of the head (ears, hats), smaller = closer (default 0.9, max 1.0); --post-only is enough")
    ap.add_argument("--colors", type=int, default=None, help="palette size for sprites and faces (crisp/smooth, default 64)")
    ap.add_argument("--walk-frames", type=int, default=None, help="frames per gait cycle in the smooth walk sheet (default 8)")
    ap.add_argument("--video-full", action="store_true", help="walk clips with 20 steps and CFG instead of the 4-step "
                    "LoRAs: 6x slower, livelier but less regular motion (experiment 17)")
    ap.add_argument("--concept-image", default=None)
    ap.add_argument("--edit-model", default="qwen21", choices=["qwen21", "flux3"])
    ap.add_argument("--concept-model", default="turbo", choices=["turbo", "aesthetic"])
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    a = ap.parse_args()

    review = tuple(g for g in a.review.split(",") if g and g != "none")
    spec = Spec(a.name, emotions=[e for e in a.emotions.split(",") if e], review=review,
                video_fast=not a.video_full, concept_image=a.concept_image,
                edit_model=a.edit_model, concept_model=a.concept_model)
    b = Builder(spec, a.out or f"characters/{a.name}", Comfy(a.url))
    src = b.state["source"]                                   # what earlier runs stored; arguments override it
    look = b.state.setdefault("look", {})                     # post-processing options are remembered as well
    for key, arg, default in (("face_grid", a.face_grid, 2), ("walk_frames", a.walk_frames, 8),
                              ("pixel_style", a.pixel_style, "crisp"), ("face_style", a.face_style, "lines"),
                              ("colors", a.colors, 64), ("face_zoom", a.face_zoom, 0.9)):
        if arg is not None:
            look[key] = arg
        val = look.get(key, default)
        if key == "colors":
            spec.sprite_colors = spec.face_colors = val
        else:
            setattr(spec, key, min(1.0, val) if key == "face_zoom" else val)
    spec.tags = a.tags if a.tags is not None else src.get("tags", "")
    spec.desc = a.desc if a.desc is not None else src.get("desc", "")
    spec.props = a.props if a.props is not None else src.get("props", "")
    spec.seed = a.seed if a.seed is not None else src.get("seed", 1)
    spec.key = a.key or src.get("key")
    spec.notes = dict(src.get("notes", {}))
    for item in a.note:
        stage, _, text = item.partition("=")
        if text.strip():
            spec.notes[stage.strip()] = text.strip()
        else:
            spec.notes.pop(stage.strip(), None)
    if not spec.desc:
        ap.error("--desc is required for a new character")
    b.set_redo([r for r in a.redo.split(",") if r], a.variant)
    b.save_state()
    for item in a.pick:
        side, _, frame = item.partition("=")
        b.pick(side.strip(), int(frame))
    for gate in (g for g in a.approve.split(",") if g):
        if gate not in GATES:
            ap.error(f"unknown gate '{gate}' (views, sides)")
        b.approve(gate)
    print(json.dumps(b.post() if a.post_only else b.build(), indent=2))


if __name__ == "__main__":
    main()

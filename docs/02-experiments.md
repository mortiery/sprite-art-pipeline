# Experiment log

Machine: ComfyUI 0.45.0, Radeon RX 9070 XT 16 GB (ROCm 7.2), no custom nodes. All images are in
`experiments/<nn>_*/`; files starting with `_` are contact sheets made for comparison. Timings are warm
(model already loaded) wall-clock seconds per image through the HTTP API.

Three test characters are used throughout (`pipeline/characters.py`): **Rin** (simple adventurer),
**Garrick** (armoured knight with beard, scar, cape and greatsword - many details and props),
**Selka** (witch with a wide hat, long silver hair and a staff - awkward silhouette, pale hair).

## Summary of decisions

| Question | Options compared | Winner | Why |
|---|---|---|---|
| Concept model | Anima Turbo v1.1, Anima Aesthetic v1.1, FLUX 3 Klein (text only) | **Anima Turbo v1.1** | strongest anime look, most detail; Aesthetic looks flat/"off"; FLUX 3 is generic |
| Edit model (faces) | FLUX 3 Klein distilled, FLUX 3 Klein base, Qwen-Image 2.1 turbo, Qwen-Image 2.1 base | **Qwen-Image 2.1 base** | FLUX 3 faces have double eyebrows, deformed eyes/mouths at full size in every variant; Qwen is clean |
| Edit model (chibi, back view) | FLUX 3 Klein distilled, Qwen-Image 2.1 | **Qwen-Image 2.1 base** | correct directions, real stride poses, perfectly uniform key colour |
| Walk animation | sprite-sheet LoRA (FLUX.2 klein 4B), whole 3x4 sheet in one edit, one edited strip per direction, strips + compositing onto the standing sprite, **video clip per direction (Wan 2.2 I2V)** | **Wan 2.2 clip per direction, first frame = last frame** | every still-image method dropped or flipped props between frames and repeated the same step; in a clip the props stay and the legs alternate |
| Wan 2.2 settings | 20 steps CFG 3.5 vs 4-step lightx2v LoRA; 81 vs 49 frames; open-ended vs first frame = last frame | **4-step LoRA, 81 frames, first = last** | 20 steps hop irregularly and change the face; an open-ended back view walks away and shrinks |
| Side and back views | 4-view turnaround in one edited image, turntable clip from the front view, **back view as its own edit + half-turn clip from front to back** | **back edit + half-turn clip** | the edited turnaround lost the sword in both side views; a free turntable invents the back (satchel) |
| Quality control of views | automatic checks only, **automatic checks + two review gates** | **review gates** | a clip repeats the mistakes of its first frame; a look at front/back and at the side views is cheap |
| Background removal for sprites | white + flood fill, BiRefNet matting, **chroma key** | **chroma key** | only method that clears enclosed pockets (staff rings, gaps between arm and body) |
| Pixelisation (sprites) | Lanczos, nearest, nearest with edge erosion, k-centroid, line-preserving k-centroid; 24 / 64 / unlimited colours; with and without outline | **nearest + edge erosion, shared 64-colour palette, no outline** ("crisp") | keeps eyes and belt details; erosion removes key-colour fringe; 64 colours look like the unreduced image but stay identical across frames |
| Pixelisation (faces) | same methods at 72 and 144 logical px | **line-preserving k-centroid, 72 px (2 px grid), 64 colours** | nearest breaks line art into dots; plain k-centroid loses nose and mouth |
| Opposite walk steps (still images, superseded) | generic "opposite step" wording, per-frame description, single-frame "swap legs" edit, post-processing | per-frame description + planted-foot check; far-leg shading | the edit model repeats the same stride; solved for good only by video (17) |
| Chibi proportions | wording only, mannequin as proportion reference | **wording only** | mannequin reference leaks grey parts and does not shorten the body |
| Direction QC | local VLM (Qwen 3.5 9B), **skin-side geometric test** | **geometric test** | the VLM answers "LEFT, RIGHT, RIGHT" regardless of the picture |

## 00 - smoke test

API client, BiRefNet cut-out (`LoadBackgroundRemovalModel` + `RemoveBackground` are core nodes) and the
pixeliser. `pixelize_test.png` shows why a chibi redesign is needed: a normally proportioned figure reduced to
48 px is unreadable, and that k-centroid stays crisp where Lanczos smears. Note: `JoinImageWithAlpha` expects
an *inverted* mask.

## 01 - concept models (text to image)

`_sheet_anima.jpg`, `_sheet_flux3.jpg`. 3 characters x 2 seeds.

- **Anima Turbo v1.1** (8 steps, CFG 1, `er_sde`/`simple`, 832x1216): ~3 s. Strong anime style, clean white
  background. 1 of 6 seeds drew a front *and* back view -> the pipeline counts figures and retries.
- **FLUX 3 Klein distilled** (15 steps, CFG 1): ~5 s. Follows the description more literally (kept the scar) but
  the style is generic.

## 02 - FLUX 3 Klein reference edits

`_sheet_rin.jpg`. From one concept: neutral and angry bust, chibi, back view, pixel-art chibi - ~11 s each,
identity kept. The "pixel art" output only *looks* pixelated: irregular fake pixels, ~130 logical px tall. A
real grid needs post-processing regardless of the prompt.

## 03 - whole walk sheet in one edit (FLUX 3)

`_sheet_a.jpg`, `_pixel_preview.png`. Prompt only, chibi + mannequin template, chibi + template + concept.
All give a tidy 3x4 grid of the same character, and the sliced result already reads as an RPG sprite. But side
rows contain frames facing the wrong way and front/back step frames barely differ from the idle frame.

## 04 - published sprite-sheet LoRA (FLUX.2 klein 4B base + svntax `pixel_4walk_small`)

`_sheet.png`. 20 steps, CFG 5, 512x512, ~18 s. The layout is rock solid and the output is close to native
pixel art (4x4 grid of 32x32 cells). But the character is lost at that size: Garrick becomes a grey-haired boy
without beard or plate armour, Selka's hat turns into a cap and purple into navy. Good for generic NPCs, not
for "this exact character".

## 05 - FLUX 3 walk sheets for all three characters

`_cmp_<name>.jpg` (two whole sheets, then the row-by-row composite), `_pixel_rows.png`.

- Whole sheet: 6 of 6 sheets have at least one wrong frame (a side view in the front row, mixed directions in
  the side rows).
- **Row by row** (one 1296x432 strip per direction, template for that direction only): every row came out
  facing the right way, and complex designs survive (beard, scar, armour, sword; hat, staff, robe trim).
- Front/back rows still show almost no leg movement with FLUX 3.

## 06 - background removal for sprite strips

`_cmp.jpg`, `_pixel_cmp.png`. White background + border flood fill and BiRefNet matting both leave the
enclosed pocket inside the staff's ring opaque. A chroma-key background (asked for in the prompt, measured
from the image border, keyed in OKLab) removes pockets too. The key colour is chosen automatically as the one
farthest from the character's colours (`choose_key_color`: green for the purple witch, magenta for the others).
FLUX 3 renders muted, slightly textured key colours; Qwen 2.1 renders a perfectly flat one.

## 07 - emotion portraits with FLUX 3

`_A_rin.jpg`, `_A_garrick.jpg`. Neutral bust from the concept, seven emotions edited from the neutral bust.
At thumbnail size this looks excellent: identical pose and outfit, readable expressions. **At full size the
faces are flawed** (see 08): doubled eyebrows and eyelids, malformed eyes, muddy mouths.

Lesson: judge faces at 100 %, never from a contact sheet.

## 08 - Qwen-Image 2.1 turbo edits

`_rin.jpg`, `_garrick.jpg`, `_face_detail_flux3_vs_qwen.jpg`. Same tasks with Qwen-Image 2.1 turbo (8 steps):
~48 s per image. Faces are clean and on-model. In single-image chibi/back-view edits *without* the character
description it dropped Garrick's sword - the description sentence must be part of every edit prompt.

## 09 - trying to fix FLUX 3 faces

`_detail.jpg`. Editing from the concept instead of the neutral face, 1024 px, 28 steps, two references,
`er_sde`: none removes the artifacts; some variants also miss the expression.

## 10 - Qwen-Image 2.1 on template-guided walk rows

`_cmp.jpg`. ~92 s per strip (turbo). All three frames face the right way, the step frames show a real stride -
also in the front view - and the magenta is flat. Clearly better than the FLUX 3 strip beside it. (That the two
step frames show the *same* stride was only noticed later - see 13.)

## 11 - non-distilled models on faces

`_detail.jpg`. FLUX 3 Klein **base** (50 steps; CFG 1: 24 s, CFG 3.5: 40 s) still deforms eyes and mouths.
Qwen-Image 2.1 **base** (25 steps, ~57 s) is clean and marginally crisper than turbo -> chosen.

## 12 - Anima Aesthetic v1.1 vs Turbo v1.1 for concepts

`_cmp.jpg`, `_faces.jpg`. Aesthetic (30 steps, CFG 4, ~16 s) is flatter and simpler and follows colour words
more literally (blue scarf stays blue), but loses detail (armour plates, sword) and looks off. Turbo v1.1 stays
the default; Aesthetic is available with `--concept-model aesthetic`.

## 13 - getting opposite walk steps

`_explicit_prompt.jpg`, `_swap_edit.jpg`, `_post_fix_preview.png`, `_side_shading.png`.

The first Qwen rows looked right at a glance but **both step frames showed the same stride** (same leg forward).

- *Front/back views*: describing each frame separately in picture coordinates ("the foot on the LEFT side of the
  picture is lifted behind ...") plus a template whose lifted leg is lighter and shorter gives true mirrored steps.
  `steps_are_mirrored()` checks where the planted foot is in each step frame; if both are on the same side,
  step B is rebuilt as the idle upper body + mirrored legs of step A (`mirror_step`).
- *Side views*: no prompt and no single-frame "swap the legs and arms" edit changes the stride. In strict profile
  the two opposite strides have the same silhouette and differ only in which leg overlaps. The post-processor
  therefore shades the far leg and alternates it: trailing leg dark in step A, leading leg dark in step B
  (`shade_leg`, applied after palette quantisation or the shade is snapped away).
- One seed copied the grey mannequin's head and torso into the character. `grey_fraction()` compares the amount
  of neutral grey with the chibi reference and triggers a retry (+62 % on the bad strip, ~0 % on good ones).
- A pixel-difference test for "are the steps mirrored" was unreliable (props and a slight body turn dominate);
  the planted-foot test replaced it.

## 14 - pixelisation methods

`_methods.png`, `_variants.png`, `_rin_walk_styles.png`, `_face_variants.png`.

| Method | Colours (one frame) | Look |
|---|---|---|
| Lanczos | 595 | blurry |
| nearest | 428 | crisp, but magenta key-colour specks on the silhouette |
| Lanczos + median-cut 24 | 23 | muddy |
| k-centroid | 416 | clean, flat; thin lines disappear |
| k-centroid + OKLab 24 | 24 | clean, slightly posterised |
| k-centroid + outline + OKLab 24 | 24 | "classic" heavy-outline sprite |
| **nearest after eroding the alpha edge by 2 px** | 404 | crisp, no fringe |
| any of the above + shared OKLab 64 | 64 | indistinguishable from the unreduced version |

Chosen after review: sprites use `crisp` (nearest + erosion, 64 colours, no outline); `smooth` (k-centroid) and
`classic` remain available via `--pixel-style`. Faces use the line-preserving k-centroid (`prefer_dark`): the
dark cluster of a cell wins when it covers >= 30 % of it and the contrast is high, which keeps outlines, eyes
and mouths continuous. Rendering all three styles from cached raw images takes seconds (`--post-only`).

Closed mouths still vanished at 72 px (`_face_mouth_fix.png`, `_face_mouth_fix2.png`): a thin line covers too
little of a cell. Lowering the share threshold or thickening the line art (min filter) brought them back only as
dots and blackened the hair strands. What worked is a second, contrast-aware rule: a dark cluster also wins with
only >= 10 % coverage when its lightness contrast to the rest of the cell is > 0.38 (a line on skin), while
low-contrast lines (strands inside hair) still need 30 %. Mouths and the jawline return, hair stays clean.

## 15 - chibi proportions

`_cmp.jpg`, `_cmp_template.jpg`. Stock RPG Maker characters are 2 heads tall. Qwen-Image 2.1 produces ~3 heads
with a plain "chibi" request and ~2.6 with "exactly two heads tall like a classic 16-bit JRPG overworld sprite:
the head is as tall as the entire rest of the body". Giving the front mannequin as a second reference for
proportions did not help (still ~3 heads, grey patches, twice the run time).

## 16 - props that switch sides or vanish: turnaround first, canonical compositing

`_reported_failures.jpg` (in 13), `_turnarounds.jpg`, `_rows_from_idle.jpg`, `_canonical_composite.png`.

Building all three characters exposed failures that Rin (no props) had hidden: Selka's skin turned mannequin
grey (only 9 % of her pixels, below the leak threshold), her staff changed hands between frames and was missing
in the standing frame; Garrick's sword changed shoulders and was missing in the standing back view. Cause:
every strip was invented from the *front* chibi, and the phrase "mirror images of each other" invited the model
to mirror the whole figure.

1. **Turnaround first.** One 2048x512 image with the four standing views (chibi + 4-mannequin template). In a
   single image the model draws staff and sword consistently in all views; directions were right on the first
   try for both characters, skin natural, key colour flat.
2. **Strips from the matching standing view.** Each direction's strip uses its own standing view as the
   reference, with "nothing is mirrored or flipped, nothing is added or removed". Better, but still 3 of 12
   frames with a dropped or flipped prop.
3. **Canonical compositing.** The standing sprites are final; a generated step frame only supplies the pixels
   below the waist (front/back: only inside a leg zone of +-26 % of the figure height around the body centre).
   Props above the waist can no longer change. Strips are scaled to the standing view using a prop-independent
   "core height" (measured in a narrow band around the head centre).
4. **Selection instead of thresholds.** `consistency()` measures how much each step frame deviates from the
   standing sprite above the waist. Good frames score 0.22-0.36, a flipped sword 0.5-0.56, but a missing sword
   hilt only 0.35 - not separable by a fixed threshold. So the pipeline retries only on gross deviation (> 0.45),
   keeps the attempt with the lowest total deviation, and relies on compositing for the rest.

Skin detection was rewritten for this (hue window in OKLab over the whole head area): the old centre-window
sampler mistook Selka's gold hat band for skin, which is why her grey face passed the facing check.

## 17 - walk cycles from a video model (Wan 2.2), the approach that replaced 03-16

Experiments 03-16 describe the still-image route to walk frames. It is kept here as a record; its code (pose
templates, strip slicing, compositing onto the standing sprite, mirrored-step repair) was removed from
`pipeline/` when this approach replaced it.

Viewer with every clip and the sprites cut from it: `experiments/17_wan_video/index.html`
(`python3 -m http.server 8765`, then `http://127.0.0.1:8765/experiments/17_wan_video/`). Runner: `run.py`.

Building all three characters with the method of experiment 16 still left defects (Garrick's side views
without the sword, single frames with a prop gone). The edit model has no notion of "the same object a moment
later", so the walk was moved to a video model: Wan 2.2 I2V 14B (high-noise + low-noise expert, fp8),
640x640, the standing view as start image, sampled exactly like the official template.

**Timing on the RX 9070 XT (16 GB).** Both 14 GB experts load only partially (about 8.8 GB in VRAM, the rest
offloaded) and are swapped mid-run.

| Setting | 81 frames | 49 frames |
|---|---|---|
| 4 steps, lightx2v LoRAs, CFG 1 | about 130 s (19 s per step + decode) | about 72 s |
| 20 steps, CFG 3.5 | about 770 s (37 s per step) | about 6 min (17 s per step) |

The VAE decode does not fit next to the diffusion model: ComfyUI's automatic tiled fallback worked sometimes
and failed sometimes. Sampling and decoding are therefore two jobs (`SaveLatent` -> `POST /free` ->
`LoadLatent` + `VAEDecode`). The server also crashed twice with the known ROCm page fault during these model
swaps; the client now waits for it to come back and resubmits.

**What the clips showed** (clip names as in the viewer)

| Clip | Result |
|---|---|
| `rin_left_fast` | a real walk in profile from the first try: arms swing, scarf moves, 16-frame gait cycle |
| `rin_left_q` (20 steps) | similar walk; colours drift late in the clip (background brighter, boots reddish) |
| `rin_down_fast` | front view: left and right step differ clearly - the defect of experiment 13 is gone |
| `rin_down_q49` (20 steps, 49 frames) | **worse**: hops irregularly, mouth opens; no clean cycle |
| `rin_up_fast`, `rin_up_fast2`, `rin_up_fast_follow` | back view walks away and shrinks to 87 % / 90 % / 79 %, whatever the wording ("on the spot", "camera follows") |
| `rin_up_fast_flf` | **first frame = last frame**: size constant within 0.4 %, head unchanged, clean 16-frame cycle |
| `selka_down_fast` | staff stays in the same hand and the hat keeps its shape for all 81 frames |
| `garrick_left_fast` | armour and cape stable; no sword - because the start frame had none |
| `garrick_spin_fast`, `*_spin_flf` | turntable from the front view: the sword stays attached through 360 degrees, but the back is invented (odd scabbard, a satchel) |
| `garrick_half_left` / `_right` | first frame = front, last frame = back: clean half turn, side view in the middle; asked to turn left and right, it turned left both times |

**Decisions**

1. *4-step LoRA by default.* For this job the distilled model is not only 6x faster, it is steadier: small,
   regular steps that loop. The research had warned that the LoRA damps motion; here that is an advantage.
2. *Every clip is pinned at both ends* (`WanFirstLastFrameToVideo` with the same I2V weights). Walk: the
   standing view is first and last frame. Side views: front view first, back view last.
3. *The video only transports; it does not design.* Whatever the first frame lacks, the clip lacks. So front and
   back view are made by the edit model (the chibi itself on the key background, and a dedicated back-view
   edit with the carried items named), checked by a reviewer, and only then animated.
4. *Turn direction.* Wording has no effect, and for a given pair of pictures the model turns the same way
   every time (Garrick: three clips, three times to the right). Feeding both pictures flipped left-right and
   flipping the clip back gave the other side for Rin on the first try. Clips are filed under the side they
   actually show.

**Getting sprites out of a clip** (`pipeline/video.py`)

- *Keying*: per frame against its own border colour (the background drifts a little); pixels with the key's
  hue are removed at any lightness, which deletes the contact shadow the model adds under the feet.
- *Size lock*: every frame is resized until the head is as wide as in the first frame (rolling median). Not
  needed any more for pinned clips, but it repaired the receding back views completely.
- *Position lock*: the head region of every frame is registered against the standing frame; horizontal drift
  is removed, the vertical bob is snapped to whole sprite pixels, and all frames are sampled on one grid.
  Downscaling each frame separately made the hair shimmer.
- *Gait cycle*: mean leg-region difference between frames i and i+p for every lag p; the first clear minimum
  is one step or the full cycle, and it is doubled when the double lag matches clearly better. The cycle is
  taken as early as possible (colours are best there) at a pose far from standing, and where the loop closes.
  Found: 16-17 frames at 16 fps for all side and front clips, 22-24 for back views.
- *Sampling*: 8 frames evenly spaced over the cycle for the smooth sheet; the two half-cycle extremes plus the
  standing frame for the RPG Maker 3-frame row.
- *Side view from a turn*: first tried "narrowest body + most visible skin", which failed on Selka (hat, robe,
  silver hair: clips were filed under swapped sides and the picked frames were nearly back views). Now: the
  frame at which half of the clip's motion has happened, and the facing from the direction in which the
  texture of the upper body slides during the first half of the turn. Right on all eight test clips for the
  facing, within two frames for the frame - the review sheet shows the neighbours so a reviewer can pick.

**Review gates.** Selka's back view was first rejected three times by an automatic "shows a face" check that
counted her silver hair as skin (fixed with a tighter colour tolerance), while Garrick's hidden sword passed
every automatic check. Neither kind of error is acceptable before 20 minutes of video generation, so the
builder stops after front + back and after the side views and asks for a look.

## Negative results worth knowing

- **Wan 2.2 with 20 steps and CFG** for walk cycles: slower and worse than the 4-step LoRA (17).
- **Prompting a camera or a direction in Wan 2.2 I2V**: "on the spot", "camera follows at a constant
  distance", "turns to the left" were all ignored; only pinned frames control the clip (17).
- **Unpinned turntable** for side and back views: invents the back (17).
- **Skin colour as a universal detector**: fine for telling a front from a back view with a tight tolerance,
  useless for pale hair under a hat when used to find a profile (17).
- **VLM as judge**: the bundled Qwen 3.5 9B (FLUX 3 prompt upsampler) can be called through `TextGenerate` with
  an image. It tells front from back reliably but answers "LEFT, RIGHT, RIGHT" for most side strips.
- **Whole-sheet generation** is tempting (1 call instead of 4) but unreliable.
- **Flood fill from the border in Pillow** (`ImageDraw.floodfill`) silently did nothing on Pillow 13 for this
  use; `pipeline/sprites.py` has its own vectorised fill.
- **GPU out-of-memory** in `VAEDecode` happened once when two jobs shared the GPU; `Comfy.run()` now calls
  `POST /free` and retries.
- **Civitai downloads** need a login token (HTTP 401), so the `rpgchara` Anima sprite LoRA was not tested.

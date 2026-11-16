# Anime character -> RPG Maker-style pixel assets with ComfyUI

A reproducible, scriptable workflow that turns a text description into

- **anime concept art** of a character,
- a **face set with emotions** (4x2 sheet of 144x144, RPG Maker MV/MZ layout) for dialogue boxes and menus,
- **chibi walking sprites** for the overworld map: the classic 3x4 sheet of 48x48 frames (down/left/right/up)
  and a smooth 8-frames-per-direction walk cycle,
- high-resolution transparent busts and standing views, a palette and a JSON manifest that a Phaser game can
  load directly.

Everything is driven through the ComfyUI HTTP API with **core nodes only** (no custom nodes), so scripts and
AI agents can call it. Researched and tested on 2026-11-15/16 with ComfyUI 0.45.0 on a Radeon RX 9070 XT (16 GB).

| Document | Content |
|---|---|
| this file | the recommended workflow, models, how to run it |
| [docs/01-research-notes.md](docs/01-research-notes.md) | what the web research found (models, approaches, specs, licences) |
| [docs/02-experiments.md](docs/02-experiments.md) | every experiment, what won and why, negative results |
| [docs/03-agent-guide.md](docs/03-agent-guide.md) | how an agent generates characters, reviews them and uses them in Phaser |

## Quick start

```bash
# 1. describe the character; the build stops at the first review gate
python3 -m pipeline.character --name garrick --seed 11 \
  --tags "1boy, solo, mature male, veteran knight, short grey hair, grey beard, steel plate armor, red cape, large greatsword on back" \
  --desc "a burly veteran male knight with short grey hair and a grey beard, wearing steel plate armor, a long red cape and a large greatsword strapped to his back" \
  --props "a large greatsword in a brown scabbard strapped diagonally across his back, its hilt sticking up above his right shoulder"

# 2. look at characters/garrick/garrick_review_views.png (front + back view), then
python3 -m pipeline.character --name garrick --approve views

# 3. look at characters/garrick/garrick_review_sides.png (side views), then
python3 -m pipeline.character --name garrick --approve sides

# 4. play
python3 -m http.server 8765      # then open http://127.0.0.1:8765/phaser-demo/
```

About 25 minutes of GPU time per character. Output lands in `characters/<name>/` (file list in the agent
guide). `--review none` skips the gates for unattended runs.

## The workflow

```
 text -> 1 CONCEPT -+-> 2 CHIBI = FRONT VIEW -> 3 BACK VIEW -> [review] -> 4 SIDE VIEWS -> [review] -> 5 WALK CLIPS (x4)
         Anima      |   Qwen-Image 2.1          Qwen 2.1                   Wan 2.2: half turn         Wan 2.2: each standing
         Turbo v1.1 |   1024x1024               from the front             from front to back,        view walks on the spot,
         832x1216   |                           view                       side view = the middle     first frame = last frame
                    |                                                                                        |
                    |                                              key, stabilise, find one gait cycle, pixelise
                    |                                                       -> name_walk.png (3x4)  name_walk8.png (9x4)
                    |
                    +-> 6 PORTRAIT -> 7 FACES (x8) -> key, crop, pixelise -> name_faces.png
                        Qwen 2.1      Qwen 2.1, every expression (neutral too)      576x288
                                      edited from the portrait
```

| # | Stage (`--redo` name) | Model | Input | Key settings |
|---|---|---|---|---|
| 1 | `concept` | Anima Turbo v1.1 | tags | 8 steps, CFG 1, `er_sde`/`simple`, 832x1216; retried until exactly one figure is in the picture |
| 2 | `chibi` | Qwen-Image 2.1 base | concept | 25 steps, CFG 1, `euler`/`simple`, 1024x1024, chroma-key background; this is the front standing view |
| 3 | `back` | Qwen-Image 2.1 base | chibi | same; retried while it still shows a face |
| - | **review `views`** | human or vision model | front + back | checklist: items present, on the correct side, nothing invented |
| 4 | `turn_left`, `turn_right` | Wan 2.2 I2V 14B + 4-step LoRA | front view = first frame, back view = last frame | 640x640, 49 frames, 4 steps, CFG 1, shift 5; the side view is the frame in the middle of the turn. Fallback when no clip turns that way: Qwen 2.1 draws the side view from front + back |
| - | **review `sides`** | human or vision model | side views + neighbouring frames | may pick another frame (`--pick left=28`) |
| 5 | `walk_down/left/right/up` | Wan 2.2 I2V 14B + 4-step LoRA | standing view = first **and** last frame | 640x640, 81 frames at 16 fps; a gait cycle is found and cut out automatically |
| 6 | `portrait` | Qwen-Image 2.1 base | concept | 1024x1024 tight head-and-shoulders, chroma-key background; the reference for all faces |
| 7 | `face_<emotion>` | Qwen-Image 2.1 base | portrait | "Change the facial expression: ... keep everything else unchanged"; neutral included, so all tiles have the same line quality |
| 8 | post-processing (`--post-only`) | Python (Pillow + numpy) | raw images and clips | chroma key -> size and position lock -> gait cycle -> grid downscale -> shared 64-colour palette -> sheets + manifest |

### Why it is built this way

- **Three models, each for what it is good at.** A tag-trained anime model (Anima) designs the character. An
  instruction-based edit model (Qwen-Image 2.1) redraws that design as chibi, back view and faces. A video
  model (Wan 2.2) produces everything that has to stay *the same object over time*: the turn to the side
  views and the walk cycles. No LoRA training, IP-Adapter or ControlNet is needed.
- **Walk cycles come from video, not from single edits.** Asking an edit model for "the next walk frame"
  failed in every variant tried (experiments 03-16): props vanished or changed sides between frames and both
  steps showed the same leg. In a video clip the sword, staff and hat simply stay where they are, and the legs
  alternate because that is what walking looks like.
- **The video model only fills in between given pictures.** Left alone it invents: a turntable from the front
  view grew a satchel on the back, a back view "walking away" shrank into the distance. So every clip is pinned
  at both ends. Walk: first frame = last frame = the standing view (no drifting, zooming or turning away, and
  the loop closes). Side views: first frame = front, last frame = back; the model only has to rotate.
- **Two review gates.** A video keeps whatever its first frame shows - including mistakes. The front and back
  views are therefore inspected before any video is made, and the side views before the walk clips. The build
  stops, writes a review sheet and a checklist, and continues after `--approve`. Corrections are cheap at this
  point: `--redo back --note back="the sword lies on top of the cape"` or `--pick left=28`.
- **Chroma key instead of matting.** The background colour is requested in the prompt (the colour farthest
  from the character's own colours is picked automatically) and removed by colour, which also clears enclosed
  gaps that matting models and flood fills leave behind. Video models like to add a contact shadow in a darker
  shade of the key colour; pixels with the key's hue are removed at any lightness.
- **Real pixel art is made in code, not by the model.** Every frame of a clip is registered against the
  standing frame (head position, snapped to whole sprite pixels) and sampled on the same grid, so the head does
  not shimmer; all frames of a character share one palette. Three looks are built in (`--pixel-style`):
  **crisp** (default: nearest-neighbour after eroding the blended edge ring, 64 colours), **smooth** (dominant
  colour per cell) and **classic** (dominant colour + dark outline, 24 colours). Faces use a line-preserving
  variant at 72 logical pixels. Switching looks needs no regeneration: `--post-only`.
- **Deterministic and resumable.** Fixed seeds per stage (checked: regenerating a stage with the same seed gives
  a bit-identical image or clip on this machine); every raw output is cached in `raw/`; description,
  notes, seeds and approvals are remembered in `raw/state.json`; single stages can be redone and everything
  derived from them follows.

### Models

| File | Folder under `ComfyUI/models/` | Source | Licence |
|---|---|---|---|
| `anima-turbo-v1.1.safetensors` | `diffusion_models` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | CircleStone non-commercial weights; outputs usable |
| `qwen_3_06b_base.safetensors` | `text_encoders` | same repo | same |
| `qwen_image_vae.safetensors` | `vae` | same repo | same |
| `qwen_image_2.1_int8_convrot.safetensors` | `diffusion_models` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | Qwen Research licence (non-commercial) |
| `qwen3vl_8b_int8_convrot.safetensors` | `text_encoders` | same repo | same |
| `qwen_image_2.1_vae_bf16.safetensors` | `vae` | same repo | same |
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`, `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models` | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) | Apache-2.0 |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`, `..._low_noise.safetensors` | `loras` | same repo | Apache-2.0 |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) | Apache-2.0 |
| `wan_2.1_vae.safetensors` | `vae` | Wan 2.2 repo above | Apache-2.0 |

Also on disk from the comparisons (not needed): FLUX 3 Klein distilled and base, Qwen-Image 2.1 turbo, Anima
Aesthetic v1.1, FLUX.2 klein 4B base + the svntax sprite-sheet LoRA, BiRefNet. `--edit-model flux3` switches
the image edits to FLUX 3 Klein (about 6x faster, poor faces; its back-view prompt is untested).

### Prompt patterns that worked

All templates live in `pipeline/prompts.py`.

- Concept (Anima): `masterpiece, best quality, score_7, safe, anime, <character tags>, full body, standing,
  straight-on, facing viewer, arms at sides, neutral expression, simple background, white background,
  character design, flat color, clean lineart, anime coloring`
- Edit prompts (Qwen 2.1) start with the operation, name what changes, and end with a keep-clause.
- Paste the one-sentence character description into every edit prompt, and name carried items separately
  (`--props`): "a greatsword in a brown scabbard strapped diagonally across his back, its hilt above his right
  shoulder". Without it the model draws a hilt but hides the blade under the cape.
- Describe expressions by observable features ("eyebrows sharply lowered, mouth open shouting with clenched
  teeth") rather than by a single emotion word.
- Video (Wan 2.2): describe the motion as a loop on the spot ("walks on the spot like on a treadmill: a steady,
  even, looping walk cycle ... stays exactly in the centre of the frame at the same size"). Never write
  "walking away from the viewer" for the back view. Direction words for a turn ("turns to the left") are
  ignored.

## Repository layout

```
pipeline/
  character.py         end-to-end builder + CLI            (python3 -m pipeline.character --help)
  comfy.py             ComfyUI HTTP client and API-graph builder (stdlib only)
  workflows.py         graph builders: Anima, Qwen-Image 2.1, Wan 2.2, FLUX 3 Klein, FLUX.2 klein, BiRefNet
  prompts.py           prompt templates per model, emotion table
  video.py             clip analysis: keying, size/position lock, gait-cycle search, side-view pick
  sprites.py           chroma key, grid downscale, walk QC, face crop
  pixelize.py          k-centroid downscale, OKLab palette, outlines, previews
  sheets.py            RPG Maker sheet assembly + manifest
  export_workflows.py  writes workflows/*.json
workflows/             API-format JSON for every stage (for POST /prompt or an MCP server)
characters/            generated characters (one folder each) + index.json for the demo
phaser-demo/           Phaser 4 demo and the reusable loader rpgchar.js
experiments/           raw results of all comparisons (17_wan_video/index.html is a viewer for the video tests)
docs/                  research notes, experiment log, agent guide
```

The raw video frames (`characters/*/raw/walk_*/`, `turn_*/`, about 26 MB per clip) are not in git; a lossy
`.webp` of every clip is. After a fresh clone, `--post-only` therefore needs the clips regenerated first
(`--redo walk_down,...`, same seeds).

## Known limits and next steps

See the end of [docs/02-experiments.md](docs/02-experiments.md) for negative results. Open points:

- The automatic side-view pick takes the middle of the turn and was 2-3 frames off for three of six side views
  (a reviewer picked a neighbour); the `sides` review sheet shows the neighbours for that reason.
- The video model invents what the standing view hides: Garrick's legs are behind his cape in the left view,
  and the first left walk gave him teal trousers. The clip check now flags colours the standing view does not
  have, and the walk prompt carries the character description.
- Which way the model turns cannot be prompted. The other side is obtained by feeding left-right flipped
  pictures and flipping the result back; this worked on the first try for Rin, but it is a habit of the model,
  not a guarantee. For Kaela (shield on one arm) no clip in twelve showed her right side; the image edit model
  then draws that side view from front and back, and the reviewer judges it. Far-side items are its weak
  point (the shield tends to stay visible).
- The 4-step LoRA gives a steadier, more regular walk than 20 full steps (which hop and change the face), but
  the motion is modest; long robes hide most of it.
- Stock RPG Maker characters are about two heads tall; generated chibis come out at about 2.3-2.6 heads.
- ComfyUI on this machine sometimes dies with a ROCm page fault while models are moved out of VRAM. The client
  waits for the server to come back and resubmits; finished stages are cached.
- Walk only. Attack, hurt or idle animations would use the same mechanism (pinned clip, cycle or segment
  extraction) but are untested. Tilesets and battlers are out of scope.

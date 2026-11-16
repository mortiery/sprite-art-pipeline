# Research notes (web + local primary sources, 2026-11-15)

Collected before the experiments. **[P]** = read in a primary source (model card, official docs, official
workflow template, licence text). **[S]** = secondary source or inference. Reddit and Civitai HTML were not
reachable from the research tooling, so "community consensus" rests on model cards, blogs and write-ups;
Civitai facts come from its public API.

## 1. RPG Maker MV/MZ asset specification [P]

Sources: [MV help](https://rpgmakerofficial.com/product/MV_Help/page/01_11_01.html),
[MZ help](https://rpgmakerofficial.com/product/MZ_help-en/01_11_01.html),
[charset structure blog](https://rpgmakerweb.com/blog/charsets-structure-and-avoiding-traps).

| Asset | Spec |
|---|---|
| Character cell | 48x48 (default; cell = 1/12 of file width, 1/8 of height for an 8-character file) |
| One character | 3 columns x 4 rows. Rows top to bottom: **down, left, right, up** |
| Idle frame | **centre column**. Step cycle 2-1-2-3 (1-based), i.e. idle, step A, idle, step B |
| Single-character file | `$name.png`, 144x192. `!` prefix disables the 6 px upward shift |
| 8-character file | 4 x 2 characters = 576x384 |
| Faces | 144x144 each, 4 columns x 2 rows = 576x288 |
| SV battlers | 576x384, 9x6 cells of 64x64 |
| Busts / pictures | no engine standard, any size |
| VX Ace (for comparison) | 32x32 cells, faces 96x96 [S] |

Stock look: characters are 2 heads tall (head : body = 1 : 1) [P]. Sprites are native 48 px art with dark
*tinted* outlines; stock faces are painted anime illustrations at 144 px rather than pixel art [S].
Useful integer grids for faces: 144 = 48x3 = 72x2.

**Licence**: the RPG Maker EULA limits stock ("RTP") assets to games made with RPG Maker
([EULA](https://rpgmakerweb.com/eula)) [P]. They must not be used as templates/ControlNet inputs for a Phaser
game. Several free itch.io bases now forbid AI use. This project therefore draws its own mannequin template
(`pipeline/template.py`).

## 2. Base models for anime character concepts

| Model | What it is | Licence for a game | Notes |
|---|---|---|---|
| **Anima** (CircleStone Labs) [P] [card](https://huggingface.co/circlestone-labs/Anima) | 2B DiT derived from Cosmos-Predict2; Qwen3-0.6B text encoder; Qwen-Image VAE. Variants: base v1.0, aesthetic v1.1, **turbo v1.1** (on disk, 8-12 steps, CFG 1) | weights non-commercial, **outputs may be used commercially** (README lists "concept art for paid products") | danbooru tags + prose; `er_sde` = "flat colours, sharp lines"; prefix `masterpiece, best quality, score_7, safe,` |
| **FLUX 3 Image Klein** (BFL, 2026-10-15) [P] [blog](https://bfl.ai/blog/flux-3-image-klein), [ComfyUI docs](https://docs.comfy.org/tutorials/flux/flux-3-klein) | 5.6B flow transformer; t2i **and** editing with up to 10 reference images; Qwen3-VL-4B encoder; FLUX.2 VAE. On disk: distilled int8 build (16 steps, CFG 1) | FLUX Kommunity licence: commercial outputs allowed under US$5M revenue | prose captions only, no tag knowledge; native anime look reported as generic [S] |
| WAI-illustrious-SDXL v17 [P via Civitai API] | SDXL/Illustrious fine-tune, tag-only | commercial image use allowed | biggest LoRA/ControlNet ecosystem; weak on long outfit descriptions [S] |
| NoobAI-XL | SDXL | **prohibits any commercial use of outputs** [P] | excluded |
| Z-Anime, NetaYume Lumina | Apache-2.0 alternatives | free | small ecosystems, not tested here |

Local primary source: the official templates shipped with ComfyUI 0.45
(`image_anima_base_v1`, `image_flux3_klein_distilled_t2i`, `image_flux3_klein_distilled_image_edit`) give the
exact node graphs used in `pipeline/workflows.py`.

Known AMD/ROCm issues with int8 "convrot" models on RX 9070 XT [P]: black/NaN output with
`--enable-triton-backend` ([ComfyUI #15084](https://github.com/Comfy-Org/ComfyUI/issues/15084)); slow eager
fallback before comfy-kitchen 0.2.26 ([comfy-kitchen #78](https://github.com/Comfy-Org/comfy-kitchen/issues/78)).
Neither occurred on this machine (comfy-kitchen 0.2.41).

## 3. Walking sprite sheets - approaches found

| Approach | Evidence | Verdict before testing |
|---|---|---|
| **Sprite-sheet LoRA on an edit model** - svntax-dev: [Qwen-Edit-2511 6x4 32x48](https://huggingface.co/svntax-dev/pixel_spritesheet_4walk_combat_32x48_v1), [Boogu-Edit v2](https://huggingface.co/svntax-dev/pixel_spritesheet_4walk_combat_32x48_v2), [FLUX.2 klein 4B base, 4x4 32x32](https://huggingface.co/svntax-dev/pixel_spritesheet_4walk_small_lora_v1) (all Apache-2.0) [P] | layout is learned; rows already down/left/right/up; author reports cut-off hair, inconsistent back row, identity drift | strongest published option; needs extra base models |
| **Text-to-image sprite LoRA** - ["RPG Character Sprite" `rpgchara`](https://civitai.com/models/1397312) for Illustrious and Anima (3x4 grid), ["Walking Sprite"](https://civitai.com/models/1437914) [P via API] | no reference image input, so the character is re-described in tags | cheap to test with Anima |
| **Template "dressing" with an edit model + LoRA** - [SpriteForge](https://github.com/tantk/spriteforge-pipeline) (Qwen-Edit-2511 LoRA, ~40 GB VRAM), LPC walk LoRA for klein 9B (non-commercial) [P] | proven design, heavy | not usable as-is on 16 GB |
| **Zero-shot template dressing** (edit model + pose-guide sheet, no LoRA) | no write-up found | *our experiment* |
| ControlNet/inpaint over a template with Illustrious ([Civitai workflow 1790750](https://civitai.com/models/1790750)) | author: output "generally needs to be tweaked"; 5 custom node packs | skipped |
| Video models (Wan 2.2 i2v, pixel-walk LoRAs) | "frame-to-frame coherence is the exact thing they are weakest at" ([nuilab docs](https://aigaming.nuilab.org/workflows/tilesets-and-animation/)); tuned for 24 GB NVIDIA | skipped |
| Image -> 3D -> rig -> render | one tutorial, poorly rated | skipped |
| Hosted services (Retro Diffusion `four_angle_walking` 48x48, PixelLab) | closed | design reference only |

## 4. Turning generated images into real pixel art

Agreed practice [P]: never generate at native sprite size. Generate large, cut out, reduce by a known factor,
quantise to one shared palette, upscale with nearest neighbour for display. Matting must come **before**
quantisation ([nuilab pixel-perfect guide](https://aigaming.nuilab.org/workflows/pixel-perfect-output/)).

- Reduction: **k-centroid** (dominant colour per cell; Astropulse) - implemented in
  [PixelOE](https://github.com/KohakuBlueleaf/PixelOE) (Apache-2.0) and in this repo's `pipeline/pixelize.py`.
- Unknown/drifting grid: [pixel-art-fixer](https://github.com/Retro-Diffusion/pixel-art-fixer) (MIT),
  [unfake.py](https://github.com/painebenjamin/unfake.py) (MIT),
  [proper-pixel-art](https://pypi.org/project/proper-pixel-art/) (MIT).
- Palette: Pillow quantise, k-means, fixed Lospec palettes; in-graph via
  [ComfyUI-PixelArt-Detector](https://github.com/dimtoneff/ComfyUI-PixelArt-Detector) (MIT).
- Pixel LoRAs exist per base model (SDXL `nerijs/pixel-art-xl`, Illustrious, FLUX.2 klein, Qwen-Image
  portraits `svntax-dev/pixel_portrait_lora_v1-lora`); none for FLUX 3 klein was found.
- No source compared methods at 48x48 / 144x144 specifically - tested here.

## 5. Transparent backgrounds

- Sprite-specific projects key out a flat colour (magenta/green/white) and binarise alpha.
- Matting models: BiRefNet (MIT; **ships as a core ComfyUI node** in 0.45: `LoadBackgroundRemovalModel` +
  `RemoveBackground`), BEN2 (MIT), RMBG-2.0 (non-commercial).
- LayerDiffuse is SD1.5/SDXL only.

## 6. Existing end-to-end projects worth borrowing from

- [agent-sprite-forge](https://github.com/0x0funky/agent-sprite-forge) (MIT): approve one master still, QC gates
  with retries, palette lock, colour keyer, atlas JSON. Uses paid APIs.
- [comfyui-mcp](https://github.com/artokun/comfyui-mcp) (MIT): MCP server over the ComfyUI REST/WebSocket API
  (`enqueue_workflow`, `upload_image`, `get_image`).
- [pixel-art-spritesheet-sandbox](https://github.com/svntax/pixel-art-spritesheet-sandbox) (MIT): Phaser preview
  of LoRA sheets.
- [pxl-cleanup](https://github.com/alvincrespo/pxl-cleanup) (MIT): bg-remove -> grid -> palette -> bottom-centre.

## 7. Phaser

Phaser 4.2.x is current (v4.0 released April 2026). Uniform-cell sheets load with
`this.load.spritesheet(key, url, {frameWidth, frameHeight})`; frames number row-major from 0, so an RPG Maker
`$` sheet maps to down 0-2, left 3-5, right 6-8, up 9-11. Set `pixelArt: true` in the game config.

## 8. Keeping one character consistent across derived images

No published anime-specific comparison of edit models for expressions / chibi / four views was found, so the
head-to-head had to be run here.

| Approach | Evidence | Licence | Fit for 16 GB |
|---|---|---|---|
| **FLUX 3 Klein edit** (on disk) [P] | up to 10 references via `ReferenceLatent`/`ReferenceMultiLatent`, cfg 1, 16 steps; good at one-image multi-view sheets; reports of weaker face consistency than Qwen 2.1 on photos [S] | commercial outputs OK (< US$5M revenue) | yes (5.6 GB int8) |
| **Qwen-Image 2.1** (2026-09-20) [P existence, S quality] | best reported identity retention incl. one anime report; expression/angle sheet LoRAs exist; can under-apply expressions | Qwen Research licence - **non-commercial** | yes (7.3 GB int8 + 9.4 GB encoder, sequential) |
| Qwen-Image-Edit 2511 + fal Multiple-Angles LoRA [P] | newest Apache-2.0 Qwen edit model; left/right sometimes swapped [S] | Apache-2.0 | 20.5 GB, needs RAM offload |
| FLUX.2 klein 4B [P] | light, Apache-2.0; 4B drifts in side views [S] | Apache-2.0 | yes |
| Anima + SnapRef v2a reference adapter [P card] | regenerates rather than edits; chibi/back view untested | non-commercial weights | yes |
| SDXL IP-Adapter / sheet LoRAs (Illustrious, NoobAI) [S] | copies "feel", not identity; sheet LoRAs are text-to-image only | mixed | yes |
| Per-character LoRA | needs a synthetic dataset made with an edit model anyway; no AMD timings found | - | only if edit models drift |

No chibi-conversion LoRA and no sheet/turnaround LoRA exists for FLUX 3 Klein (as of 2026-11-15).

Recurring failure modes reported by people building similar pipelines
([VNCCS](https://github.com/AHEKOT/ComfyUI_VNCCS), [SpriteKiln](https://github.com/brunocpf/spritekiln),
[modl character-sheet guide](https://modl.pedroalonso.net/guides/character-sheet/),
[PixelArtistry multiview](https://github.com/pixelartistry/PixelArtistry-Pixal3D-MultiView)):

- Small faces in full-body outputs get redrawn -> make one bust portrait first and derive every emotion from it.
- Chained edits degrade -> always edit from the approved canonical image, never from an edit of an edit.
- Left/right confusion -> check programmatically, or generate one side and mirror (symmetric designs only).
- The back view is invented -> approve one back view, then reuse it as a reference.
- Head-to-body ratio drifts.
- The FLUX 3 prompt upsampler "has a profound impact on model quality" (BFL staff) but multiplies run time [S].

## 9. Walk animation with a video model (second research round, 2026-11-15)

Done after the frame-by-frame edit approach (experiments 10-16) kept producing defects. Two research agents
(Claude Haiku) searched the web; the settings were cross-checked against the workflow templates that ship
with ComfyUI 0.45 (`comfyui_workflow_templates_json/templates/video_wan2_2_*.json`), which count as primary.
[P] = read on a primary page or in a local template, [S] = secondary / anecdotal.

**Model.** Wan 2.2 (Apache-2.0, [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)) is what people use
for sprite animation. Variants: I2V 14B (two experts: high-noise for layout and motion, low-noise for
detail), T2V 14B, TI2V 5B, Fun InP, Fun Control, S2V, Animate [P]. One sprite write-up reports that the 5B
model "changed the character design and froze the pose" [S], so the 14B I2V pair is the candidate.
Core nodes exist for all of them (`WanImageToVideo`, `WanFirstLastFrameToVideo`, `WanFunInpaintToVideo`,
`Wan22FunControlToVideo`, `WanVaceToVideo`, `WanAnimate2ToVideo`); no custom nodes are needed [P, local].

**Settings (official templates) [P].**

| Workflow | Node | Steps | CFG | Shift | Notes |
|---|---|---|---|---|---|
| I2V 14B | `WanImageToVideo` 640x640, 81 frames | 20 (high 0-10, low 10-20) | 3.5 | 5 | `euler`/`simple`, 16 fps |
| first + last frame | `WanFirstLastFrameToVideo` 640x640, 81 | 20 (split at 10) | 4 | 8 | same I2V weights |
| 4-step LoRA (lightx2v) | either | 4 (split at 2) | 1 | 5 | "acceleration is significant but the video dynamic is lost" (docs.comfy.org) |

Only official timing: RTX 4090D, 640x640x81: about 520 s without and 80-140 s with the 4-step LoRA [P].
Nothing was found for 16 GB cards or ROCm, so this had to be measured here.

**How people make sprites with it [S].**

- [cueqzapper/pixel-video-sprites](https://github.com/cueqzapper/pixel-video-sprites): Wan 2.2 I2V 14B,
  384x384, 33 frames, prompt "... walking on the spot like on a treadmill", "side view facing left", other
  directions by mirroring. Notes: "shrinking the video gives mush" (generate large, redraw/downscale later),
  the 4-step LoRA on the high-noise stage damps motion, not every seed works.
- Spriterrific (commercial tool): "simple west / left-facing in-place walk cycle for a 2D game sprite",
  "flat chroma background only", 8 frames.
- LoRAs for Wan 2.2 I2V on Hugging Face, not gated: [theamusing/wan2.2_pixel_walk_lora](https://huggingface.co/theamusing/wan2.2_pixel_walk_lora)
  (trigger `pix3lwalk`, side-view walk, weight 0.7-0.9, 24 frames, high + low file) and
  [styly-agents/Wan2-2-pixel-animate](https://huggingface.co/styly-agents/Wan2-2-pixel-animate)
  (226 sprite clips incl. walk cycles, rank 256, 45 frames, trained at 600x370).

**Not found.** No verified method for seamless loops (beyond first frame = last frame), for keeping the
character centred, for back views, or for picking frames of exactly one gait cycle; no head-to-head of newer
open video models (LTX-2.x, MiniMax H3) on anime sprites. These parts are solved in code here
(`pipeline/video.py`) and tested in experiment 17.

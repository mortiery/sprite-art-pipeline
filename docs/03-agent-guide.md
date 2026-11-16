# Agent guide: generating RPG characters for a Phaser game

Written for an AI coding agent (or a script) that needs characters for a 2D RPG. Everything runs against the
local ComfyUI at `http://127.0.0.1:8188`; the only dependencies are Python 3 with Pillow and numpy. The agent
must be able to **look at images**: two steps of the process are visual reviews.

## 1. Build a character: three calls and two looks

```bash
cd /home/tashi/Projects/claude-comfyui-research

# call 1 - describe the character
python3 -m pipeline.character --name selka --seed 11 \
  --tags "1girl, solo, young witch, long straight silver hair, violet eyes, large purple witch hat with gold band, purple robe with gold trim, white collar, black tights, short brown boots, holding wooden staff with blue crystal" \
  --desc "a young witch with long straight silver hair and violet eyes, wearing a large purple witch hat with a gold band, a purple robe with gold trim and a white collar, black tights and short brown boots, holding a wooden staff topped with a blue crystal" \
  --props "a wooden staff as tall as herself, topped with a blue crystal, held upright in her right hand"
```

- `--tags`: danbooru-style tags for the concept image (Anima). Start with `1girl` / `1boy`, then hair, eyes,
  outfit from top to bottom, carried items. Lowercase, comma separated.
- `--desc`: the same character as **one prose sentence** starting with "a ...". It is pasted into the edit
  prompts.
- `--props`: everything the character carries, with **where** it is: which hand, which shoulder, "across the
  back". Leave it out for characters without items. This is what keeps a sword from vanishing.
- `--seed`: fixes every stage. Same seed + same models = same character.

Every call prints one JSON object. Its `status` says what to do next.

### `"status": "review_needed", "gate": "views"`  (after about 3 minutes)

Open `image` (`characters/selka/selka_review_views.png`: concept, front view, back view) at full size and go
through `checklist`. Typical failures: an item is missing or only half drawn in the back view (a sword hilt
without blade), the item is on the wrong side (what is on the character's right is on the **right** of the back
view), something was invented (a bag), the back view still shows a face.

```bash
python3 -m pipeline.character --name selka --approve views                 # all good -> continue
python3 -m pipeline.character --name selka --redo back --variant 1 \
        --note back="the staff is in her right hand, on the right side of the picture"   # fix the back view, look again
python3 -m pipeline.character --name selka --redo chibi --variant 1        # front view wrong (back is redone too)
```

`--variant n` changes the seed of the redone stages; `--note <stage>="..."` adds a sentence to that stage's
prompt and is remembered. Only `--name` is needed on later calls - description, notes, seeds and approvals are
stored in `characters/<name>/raw/state.json`.

### `"status": "review_needed", "gate": "sides"`  (about 3 more minutes)

The side views are frames from two short videos in which the character turns from the approved front view to
the approved back view. Open `selka_review_sides.png`: the top row shows front, left, right, back; below are
strips with the frames around each chosen side view, numbered.

- LEFT must face the left edge, RIGHT the right edge, in profile (one eye visible, not the back of the head).
- If a neighbouring frame is the better side view, take it: `--pick left=21 --pick right=28`.
- An item held on the far side is correctly hidden behind the body. A half-transparent or smeared item is a
  defect - pick a frame where it is solid.
- If a side is wrong altogether (`automatic_warnings` says it was mirrored): `--redo turn_left --variant 1`.

```bash
python3 -m pipeline.character --name selka --pick left=21 --approve sides
```

### `"status": "done"`  (about 20 more minutes: 4 walk clips, 8 faces)

```
characters/selka/
  selka.json             manifest (read this, not the file names)
  selka_walk.png         144x192 = 3x4 frames of 48x48, rows down/left/right/up, RPG Maker MV/MZ "$" layout
  selka_walk8.png        432x192 = 9x4 frames: column 0 standing, columns 1-8 one walk cycle per direction
  selka_faces.png        576x288 = 4x2 faces of 144x144, order = manifest.faces.frames
  selka_walk_<dir>.gif   the four walk cycles as they will look in the game (4x)
  selka_walk_review.png  all 36 frames of the smooth sheet, 4x
  selka_preview.png      3-frame sheet + faces
  selka_stand_<dir>.png  the four standing views, 640 px, transparent
  selka_concept.png      full-size anime concept art
  selka_palette.png      the sprite palette
  busts/selka_<emotion>.png   1024 px transparent bust for menus / cut-ins
  raw/                   unprocessed model outputs per stage (cache), state.json; clips as folders of
                         frame_001.png ... (numbers = the ones on the review sheets) plus a .webp preview
```

Look at `look_at` (the walk review sheet and the preview) and read `problems`. The character is registered
in `characters/index.json`, which the Phaser demo reads.

## 2. Repair only what is wrong

```bash
python3 -m pipeline.character --name selka --redo walk_left --variant 1     # another take of one walk clip
python3 -m pipeline.character --name selka --redo face_angry --variant 2    # one face
python3 -m pipeline.character --name selka --post-only --pixel-style smooth # only slicing / pixelising (seconds, no GPU)
```

Stages: `concept chibi back turn_left turn_right walk_down walk_left walk_right walk_up face_<emotion>`.
Redoing a stage redoes what is derived from it (`chibi` -> `back` -> turns and walks; `turn_left` -> `walk_left`;
`face_neutral` -> all faces) and invalidates the approval of a gate whose pictures changed.

`problems` comes from automatic checks and can be wrong in both directions; the eyes decide.

| Message | Meaning | What to do |
|---|---|---|
| `the walk cycle does not close cleanly` | last and first frame of the cycle differ | watch the GIF; if it jumps, `--redo walk_<dir> --variant n` |
| `the legs barely move` | long robe, or the model stood still | usually acceptable for robes, else redo |
| `the head changes during the walk` | face redrawn or head turned | redo |
| `part of the figure above the waist disappears` | a carried item fades out in some frames | redo (the builder already retried twice) |
| `colours appear that the standing view does not have` | the model invented clothing, typically for body parts hidden in the standing view (legs behind a cape) | redo; if it repeats, add the missing part to `--desc` ("armoured legs") |
| `the character changes size a lot` | walked towards / away from the camera | redo |
| `left[0] does not face left ...` | colour heuristic on the 3-frame sheet | check by eye; pale hair, hats and helmets cause false alarms |
| `turn_left: ... mirrored the other side view` | no clip turned that way | `--redo turn_left --variant n`, or accept for symmetric characters |

Useful switches: `--review none` (no gates, for batches), `--pixel-style crisp|smooth|classic`,
`--face-style lines|smooth|crisp`, `--colors 64`, `--face-grid 1|2|3`, `--walk-frames 8`, `--video-full`
(20-step walk clips: 6x slower, livelier but less regular), `--key green|magenta|blue`,
`--concept-image path.png` (start from existing art), `--emotions neutral,happy,...`.

## 3. Use the assets in Phaser (3.x or 4.x)

Copy `phaser-demo/rpgchar.js` into the game. It reads the manifest and prefers the smooth sheet:

```js
import { fetchManifests, preloadCharacter, createAnimations, addCharacter, faceFrame } from './rpgchar.js';

const manifests = await fetchManifests(['characters/selka/selka.json']);   // before new Phaser.Game(...)

// in the scene
preload() { manifests.forEach((m) => preloadCharacter(this, m)); }
create() {
  manifests.forEach((m) => createAnimations(this, m));
  const hero = addCharacter(this, manifests[0], 100, 100);   // origin at the feet
  hero.walk('left');                                         // plays selka-walk-left (8 frames, loops)
  hero.stop();                                               // standing frame of the current direction
  this.add.image(12, 200, 'selka-faces', faceFrame(manifests[0], 'angry')).setOrigin(0).setDisplaySize(72, 72);
}
```

By hand, without the loader:

```js
// smooth sheet: 9 columns x 4 rows of 48x48; row = down, left, right, up; column 0 = standing
this.load.spritesheet('selka-anim', 'characters/selka/selka_walk8.png', { frameWidth: 48, frameHeight: 48 });
const rows = ['down', 'left', 'right', 'up'];
rows.forEach((dir, r) => {
  this.anims.create({ key: `selka-walk-${dir}`, repeat: -1, frameRate: manifest.walkAnim.animations[`walk-${dir}`].frameRate,
    frames: this.anims.generateFrameNumbers('selka-anim', { start: r * 9 + 1, end: r * 9 + 8 }) });
  this.anims.create({ key: `selka-idle-${dir}`, frames: [{ key: 'selka-anim', frame: r * 9 }] });
});

// RPG Maker sheet: 3 columns x 4 rows; frames down 0-2, left 3-5, right 6-8, up 9-11; middle = standing
this.load.spritesheet('selka-walk', 'characters/selka/selka_walk.png', { frameWidth: 48, frameHeight: 48 });
// play as [b + 1, b + 2, b + 1, b] at 8 fps
```

Game config: `pixelArt: true, roundPixels: true`. Scale sprites by whole numbers only (2x, 3x). Faces are
144 px with a 2 px grid: show them at 72 px or 144 px.

Run the demo: `python3 -m http.server 8765` in the project root, then open
`http://127.0.0.1:8765/phaser-demo/` (arrows walk, Tab switches the hero, Space talks / next emotion;
`?rm` uses the 3-frame sheets).

## 4. Calling ComfyUI directly

`pipeline/comfy.py` is a small stdlib client; the HTTP API is all that is needed:

| Step | Request |
|---|---|
| upload a reference | `POST /upload/image` (multipart field `image`, `overwrite=true`) -> `{name}` |
| run a workflow | `POST /prompt` with `{"prompt": <API-format graph>, "client_id": "..."}` -> `{prompt_id}` |
| wait | `GET /history/<prompt_id>` until the id appears; `status.status_str == "error"` means failure |
| fetch images | `GET /view?filename=..&subfolder=..&type=output` for each entry in `outputs.<node>.images` |
| unload models | `POST /free` with `{"unload_models": true, "free_memory": true}` |

Ready-made API graphs for every stage are in `workflows/*.json` (regenerate with
`python3 -m pipeline.export_workflows`); replace the ALL-CAPS tokens. In Python:

```python
from pipeline.comfy import Comfy
from pipeline import workflows as W, prompts as PR
c = Comfy()

# an extra expression from an existing neutral face
ref = c.upload_image("characters/rin/raw/face_neutral.png", "rin_neutral.png")
graph = W.qwen21_edit(PR.PROMPTS["qwen21"]["expression"].format(expression="smug, half-closed eyes, one corner of the mouth raised"), [ref])
files = c.run(graph, "out/")          # -> [PosixPath('out/....png')]

# a video clip: sampling and decoding are two jobs with a model unload in between (16 GB VRAM)
start = c.upload_image("characters/rin/raw/stand_left.png", "rin_left.png")
g = W.wan22_i2v("...prompt...", start, end=start, width=640, height=640, length=81, seed=1, fast=True, prefix="ccr/clip")
frames = c.run_video(g, lambda latent: W.wan_decode(latent, prefix="ccr/clip"), "out/clip/")   # 81 PNGs + .webp
```

An MCP server such as [comfyui-mcp](https://github.com/artokun/comfyui-mcp) exposes the same calls
(`upload_image`, `enqueue_workflow`, `get_history`, `get_image`) and can be fed the JSON files unchanged.

## 5. Rules that keep characters consistent

1. **Pictures for design, video for motion.** The edit model makes the front view, the back view and the
   faces. Everything that must stay the same object over time - the turn to the side views, the walk - comes
   from the video model.
2. **Pin every clip at both ends.** Walk: first frame = last frame. Turn: front view -> back view. An unpinned
   clip drifts, zooms or invents.
3. **Review before video.** A clip repeats the mistakes of its first frame for 81 frames.
4. **Name carried items and their place** (`--props`), and correct single views with `--note`.
5. **Chroma-key background** for anything that will be cut out; never white.
6. **Post-process in code**: key -> lock size and position -> one gait cycle -> grid downscale -> one palette
   per character -> nearest-neighbour only.
7. **Look at faces and side views at 100 %** before accepting them.
8. If ComfyUI stops answering, it probably crashed while swapping models (known ROCm issue on this machine,
   see `/home/tashi/Projects/comfyui/crash-2026-11-15-gpu-memory-fault.md`). The client waits up to 15 minutes
   for the server to return and resubmits the job; restart it with
   `/home/tashi/Projects/comfyui/start-comfyui.sh` (or set `COMFY_RESTART_CMD` to that script and the client
   starts it itself). Finished stages are cached, so rerunning the same command is always safe.

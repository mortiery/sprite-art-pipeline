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
  outfit from top to bottom, carried items. Lowercase, comma separated, short (2-4 words each, e.g.
  `holding spear`, `round shield`, `dark skin`); details like "hangs over the left shoulder" go into `--desc`
  and `--props`, which steer every later picture. The concept may miss such details - check them at the
  `views` gate.
- `--desc`: the same character as **one prose sentence** starting with "a ...". It is pasted into the edit
  prompts.
- `--props`: everything the character carries, with **where** it is: which hand, which shoulder, "across the
  back", "hanging from the left side of the backpack". Always mean the **character's** left and right. Leave it
  out for characters without items. This is what keeps a sword from vanishing.
- `--seed`: fixes every stage. Same seed + same models = same character (verified bit-identical).

Every call prints one JSON object. Its `status` says what to do next; `if_ok` / `if_wrong` contain the
commands to continue. Only `--name` is needed on later calls - description, props, notes, seeds, approvals and
look options are stored in `characters/<name>/raw/state.json`.

**Left and right.** The character's right side is on the **left** of the picture in the front view and on the
**right** of the picture in the back view. Facing left, you see the character's left side; facing right, the
right side. When a view is wrong, write a note that names both, e.g. "the lantern hangs from the right side of
his backpack, on the left side of the picture".

### `"status": "review_needed", "gate": "views"`  (after about 3-5 minutes)

Open `image` (`characters/selka/selka_review_views.png`: concept, front view, back view) and go through
`checklist`. Front and back are judged as a pair. Typical failures: an item is missing or only half drawn in
the back view (a sword hilt without blade), an item is on the wrong side, front and back disagree, something
was invented (a bag), an item that should hang from something floats next to it, the back view shows a face.

```bash
python3 -m pipeline.character --name selka --approve views       # all good -> continue
python3 -m pipeline.character --name selka --redo back \
        --note back="the staff is in her right hand, on the right side of the picture"   # new back view, look again
python3 -m pipeline.character --name selka --redo chibi \
        --note chibi="..."                                        # new front view; the back view is redone too
```

- `--redo` gives the named stages a **new seed** every time (variant 1, 2, 3 ... counted per stage, stored in
  `state.json`). Stages derived from them are regenerated with their own seed, because their input changed.
  `--variant n` sets a specific variant instead, e.g. to go back to an earlier take.
- `--note <stage>="..."` adds a sentence to that stage's prompt and is remembered; `--note <stage>=` removes it.
  Notes work for `chibi` (front view) and `back`. Everything later is steered by the pictures, not by text.
- `automatic_warnings` come from colour heuristics. "back: may show a face" has so far only been a false alarm
  (beige or skin-coloured items on the back); trust your eyes.
- Give up after about three takes of the same stage and approve the best one, writing down what is wrong; the
  video stages repeat whatever the pictures show.

### `"status": "review_needed", "gate": "sides"`  (about 3 more minutes; up to 15 when the model will not turn one way)

The side views are frames from two short videos in which the character turns from the approved front view to
the approved back view. Open `selka_review_sides.png`: the top row shows front, left, right, back; below are
strips with the frames around each chosen side view, numbered, with the direction of the turn in the title.

- LEFT must face the left edge, RIGHT the right edge, in clean profile: **one eye visible**. The back of the
  head and no eye means the frame is too close to the back view.
- The automatic choice is often 2-4 frames off. The strips are small: open candidates at full size from
  `characters/<name>/raw/turn_left/frame_NNN.png` (all frames of the turn are there) and take the best with
  `--pick left=25 --pick right=27`.
- An item on the far side is correctly hidden by the body; a flat item (backpack, shield) looks thin from the
  side. A half-transparent, smeared or "held" item that should hang is a defect - pick another frame.
- If a side is wrong altogether: `--redo turn_left` (a new turn clip).
- For some one-sided characters (a shield on one arm) the video model only ever turns one way. The builder
  then tries up to six clips and lets the image edit model **draw** the missing side view from front and back
  (`automatic_warnings`: "drawn by the image edit model"). Such a view has no frames, so `--pick` does not
  apply; retake it with `--redo turn_right`, steered with `--note turn_right="..."`. Items on the far side are
  hard for the model ("hidden behind the body" still often shows them) - accept a near miss that reads
  correctly at 48 px rather than a mirrored view.
- `--pick` and `--redo` can be given in one call; the pick is applied first.

```bash
python3 -m pipeline.character --name selka --pick left=21 --approve sides
```

### `"status": "done"`  (about 15-20 more minutes: 4 walk clips of about 2-4 minutes, 8 faces of about a minute)

```
characters/selka/
  selka.json             manifest (read this, not the file names)
  selka_walk.png         144x192 = 3x4 frames of 48x48, rows down/left/right/up, RPG Maker MV/MZ "$" layout
  selka_walk8.png        432x192 = 9x4 frames: column 0 standing, columns 1-8 one walk cycle per direction
  selka_faces.png        576x288 = 4x2 faces of 144x144, order = manifest.faces.frames
  selka_walk_<dir>.gif   the four walk cycles as they will look in the game (4x)
  selka_walk_review.png  all 36 frames of the smooth sheet, 4x (one row per direction)
  selka_preview.png      3-frame sheet + faces
  selka_stand_<dir>.png  the four standing views, 640 px, transparent
  selka_concept.png      full-size anime concept art
  selka_palette.png      the sprite palette
  busts/selka_<emotion>.png   1024 px transparent bust for menus / cut-ins
  raw/                   unprocessed model outputs per stage (cache), state.json; clips as folders of
                         frame_001.png ... (numbers = the ones on the review sheets) plus a .webp preview
```

Check, in this order:

1. `problems` in the printed JSON: open defects only. Warnings about views you approved at a gate are listed
   separately under `qc.generated_walk.accepted_at_review` in the manifest; the reasons for every retry are in
   `raw/state.json` (`stages.<stage>.tries`), previews of rejected turn clips in `raw/turn_tries/`.
2. `<name>_walk_review.png`: in every row the legs alternate, items stay where they are, the colours stay the
   same. If you cannot watch the GIFs, judge the loop from the stills: the last frame of a row should lead
   back into the first one. The full-size frames are in `raw/walk_<dir>/`; the manifest's
   `qc.walk_clips.<dir>` says which ones were used (`frames_used`) and how long one cycle is (`cycle_frames`).
3. The walk speed: `walkAnim.animations.walk-<dir>.frameRate` should be similar for all four directions
   (6-9 fps for normal clips). A direction at about half the others means two cycles were taken as one.
4. `<name>_faces.png`: every expression distinct; nothing important cut off at the top (ears, hat, horns).
   If it is: `--post-only --face-zoom 1.0` (more of the portrait), or `--redo portrait` (new portraits, all
   emotions follow).

The character is registered in `characters/index.json`, which the Phaser demo reads.

## 2. Repair only what is wrong

```bash
python3 -m pipeline.character --name selka --redo walk_left          # another take of one walk clip
python3 -m pipeline.character --name selka --redo face_angry         # one face
python3 -m pipeline.character --name selka --post-only --pixel-style smooth   # only slicing / pixelising (seconds, no GPU)
```

Stages: `concept chibi back turn_left turn_right walk_down walk_left walk_right walk_up portrait face_<emotion>`.
Progress is printed on stderr (one line per generation); the JSON result on stdout.
Redoing a stage redoes what is derived from it (`chibi` -> `back` -> turns and walks; `turn_left` -> `walk_left`;
`portrait` -> all faces) and invalidates the approval of a gate whose pictures changed. `portrait` is the
head-and-shoulders picture all eight faces are edited from (the neutral face too, so all tiles get the same
line quality); it is not shown itself.

`problems` comes from automatic checks and can be wrong in both directions; the eyes decide. The builder
already retried twice before reporting a walk problem.

| Message | Meaning | What to do |
|---|---|---|
| `the walk cycle does not close cleanly` | last and first frame of the cycle differ | look at the row; if it jumps, `--redo walk_<dir>` |
| `no clear step rhythm` | legs hidden by long clothing, or the model shuffled; the cycle length is a guess | look at the row and the speed; acceptable for robes |
| `the legs barely move` | long robe, or the model stood still | usually acceptable for robes, else redo |
| `the head changes during the walk` | face redrawn or head turned | redo |
| `part of the figure above the waist disappears` | a carried item fades out in some frames | redo |
| `colours appear that the standing view does not have` | the model invented clothing, typically for body parts hidden in the standing view (legs behind a cape) | redo; if it repeats, add the missing part to `--desc` ("armoured legs") |
| `the character changes size a lot` | walked towards / away from the camera | redo |
| `turn_right: no turn clip showed this side ... drawn by the image edit model` | see the `sides` gate above | check it there; retake with `--redo turn_right --note turn_right="..."` |

Useful switches: `--review none` (no gates, for batches), `--pixel-style crisp|smooth|classic`,
`--face-style lines|smooth|crisp`, `--face-zoom 0.9`, `--colors 64`, `--face-grid 1|2|3`, `--walk-frames 8`
(all of these are post-processing: `--post-only` applies them in seconds, and they are remembered),
`--video-full` (20-step walk clips: 6x slower, livelier but less regular), `--key green|magenta|blue`,
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
ref = c.upload_image("characters/rin/raw/portrait.png", "rin_portrait.png")
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

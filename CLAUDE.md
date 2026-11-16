# Project notes for Claude sessions

Local pipeline that turns a character description into RPG Maker-style pixel assets (walk sheets in four
directions, emotion faces) through ComfyUI, plus a Phaser 4 demo that plays them. Goal: AI agents call it to
make characters for Phaser games. Hobby project; the user enjoys iterating on quality.

Read first: `README.md` (workflow, models), `docs/03-agent-guide.md` (how to drive the pipeline - written for
agents and tested on fresh subagents), `docs/02-experiments.md` (what was tried, what failed, why; experiments
00-19, decision table at the top).

## Environment

- ComfyUI 0.45.0: `/home/tashi/Projects/comfyui/ComfyUI`, start with `/home/tashi/Projects/comfyui/start-comfyui.sh`,
  API at `http://127.0.0.1:8188`, models under `ComfyUI/models/`. Core nodes only - no custom nodes.
- GPU: Radeon RX 9070 XT 16 GB, ROCm. ComfyUI sometimes dies with a GPU page fault while swapping models
  (`/home/tashi/Projects/comfyui/crash-2026-11-15-gpu-memory-fault.md`). The user usually restarts it; the client
  (`pipeline/comfy.py`) waits up to 15 min and resubmits. Finished stages are cached, so rerunning is safe.
- Python 3 with Pillow and numpy (system python, no venv). No scipy.
- Headless browser for demo screenshots: `chromium --headless=new --no-sandbox --disable-gpu --window-size=1000,840
  --virtual-time-budget=6000 --screenshot=out.png "http://127.0.0.1:8765/phaser-demo/?shot=1&emotion=2"`.

## Everyday commands

```bash
python3 -m pipeline.character --name <n> --seed <s> --tags "..." --desc "..." --props "..."  # new character
python3 -m pipeline.character --name <n> --approve views|sides     # after looking at the review sheets
python3 -m pipeline.character --name <n> --redo <stage> [--note <stage>="..."] [--pick left=25]
python3 -m pipeline.character --name <n> --post-only               # re-slice / re-pixelise, no GPU
python3 -m pipeline.export_workflows                               # regenerate workflows/*.json
python3 tools/build_demo.py                                        # self-contained demo in dist/ (static hosting)
python3 -m http.server 8765    # then /phaser-demo/ (demo) and /experiments/17_wan_video/ (clip viewer)
```

Characters: rin, garrick, selka (test characters from the start), tobi (round-1 test), kaela (round-2 test).
Per-character state (description, notes, seeds, approvals, pending redos, look options): `characters/<n>/raw/state.json`.

## How to work here

- **Judge images at full size and show the user what you look at.** Name every output (path + one line on
  what it shows) as it lands; the user reviews alongside and has caught defects the checks missed (teal
  trousers, stale walk, soft neutral face). Their eye is the final quality check.
- Quality over speed: slower settings are fine when they look better. Exception found in experiment 17: the
  4-step Wan LoRA walks better than 20 steps.
- Pixel look the user prefers: `crisp` (nearest after edge erosion), ~64 colours, no outline.
- Every pipeline change: update `docs/03-agent-guide.md` if behaviour or output changes, log findings in
  `docs/02-experiments.md` (numbered experiments; negative results too), keep README's tables in sync, rerun
  `--post-only` for all characters if post-processing changed, then commit.
- Never run two builders on the same character at once; ComfyUI runs one job at a time (others queue).
- Raw video frames (`characters/*/raw/walk_*/`, `turn_*/`) are gitignored (26 MB per clip). A fresh clone
  needs the clips regenerated (same seeds give identical results on this machine) before `--post-only` works.

## Git

- Public repository: https://github.com/mortiery/sprite-art-pipeline (remote `origin`, SSH). Push `master`.
- Commits are authored as **Yannick Mortier** with the GitHub no-reply address
  `34096941+mortiery@users.noreply.github.com`, set in the repository's own `.git/config`. Do not override it
  with the machine's global identity, and do not put any other e-mail address into commits or files - GitHub
  rejects pushes that expose the account's private address.
- `push.autoSetupRemote` is on, so the first `git push` of a new branch sets its upstream.

## Iterating: fresh-agent test rounds

The pipeline is tested by giving a subagent with no context the brief in `docs/test-round-brief.md` (a new
character each time) and watching it work. Done so far: round 1 (Tobi), round 2 (Kaela) - findings and fixes
in experiment 18. Protocol that worked:

1. Spawn a general-purpose subagent in the background with the brief; pick a character that stresses something
   new (round ideas below).
2. Watch `characters/<n>/raw/state.json` with a Monitor (stages, approvals, notes, picks) and judge every review
   sheet yourself before the subagent decides; compare.
3. Do not edit pipeline code while the subagent runs a command (its Python process already loaded the old
   code). If a bug blocks it, fix it and tell the subagent with SendMessage.
4. After the report: fix code first, then docs, regenerate affected assets, look at them, commit.

Ideas for the next rounds: a helmet that hides the face, a character with wings or a long tail, a very wide
weapon (halberd, bow on the back), a non-human creature, a character in a long dress (walk rhythm), a run
without review gates (`--review none`) to see what slips through.

## Open issues (backlog)

- Long robes/dresses: no visible step rhythm, cycle length is a guess (Selka down/left are flagged).
- A thin item (spear shaft) blinking in a walk is not caught by the vanishing-item check (`missing_upper`).
- Side views drawn by the edit fallback keep far-side items visible (Kaela's shield; experiment 19).
- Automatic side-view frame is often 2-4 frames off; the reviewer fixes it with `--pick`.
- Faces do not get `--desc`/`--props` (portrait prompt uses the concept only); details like a headwrap tail
  can go missing in faces.
- Small colour fringes on metal (cyan spear tip on Kaela) - keying/model artefact, not checked.
- Chibis are 2.3-2.6 heads tall instead of RPG Maker's ~2 (experiment 15).
- Only walk cycles. Idle, run, attack would reuse the pinned-clip mechanism; untested.
- Unattended mode (`--review none`) has no automatic reviewer; a local VLM (Qwen 3.5 9B) was unreliable for left/right
  ("Negative results" in docs/02); a stronger VLM as gate reviewer is untested.

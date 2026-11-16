"""(Re)analyse every clip of experiment 17 and write index.html, a side-by-side viewer.

    python3 experiments/17_wan_video/gallery.py            # then open
    http://127.0.0.1:8765/experiments/17_wan_video/        # (python3 -m http.server 8765 in the project root)
"""
import html, json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import run  # noqa: E402

NOTES = json.loads((HERE / "notes.json").read_text()) if (HERE / "notes.json").exists() else {}
rows = []
for d in sorted(p for p in HERE.iterdir() if (p / "frames").is_dir() and list((p / "frames").glob("*.png"))):
    info = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
    spin = info.get("prompt") in ("spin", "halfturn")
    if "--reuse" not in sys.argv or not (d / "all_frames.jpg").exists():
        run.analyse(d, spin)
    cyc = json.loads((d / "cycle.json").read_text()) if (d / "cycle.json").exists() else {}
    mode = "4-step LoRA (fast)" if info.get("fast") else "full quality, 20 steps"
    if info.get("loop"):
        mode += ", first frame = last frame"
    if info.get("lora"):
        mode += ", LoRA " + ", ".join(info["lora"])
    meta = (f"{info.get('char')} / {info.get('dir')} / {mode} / {info.get('size')} px, {info.get('length')} frames / "
            f"seed {info.get('seed')} / {info.get('seconds')} s")
    if cyc:
        meta += f" / gait cycle {cyc['period']} frames from frame {cyc['start']}"
    cells = [f'<figure><img src="{d.name}/clip.webp" width="320"><figcaption>raw clip from the model</figcaption></figure>']
    if not spin:
        cells.append(f'<figure><img src="{d.name}/loop_hires.webp" width="320"><figcaption>detected cycle, keyed + centred</figcaption></figure>')
        for st in ("crisp", "smooth"):
            for n in (4, 8):
                cells.append(f'<figure><img src="{d.name}/sprite_{st}_{n}.gif" width="192" class="px"><figcaption>{st}, {n} frames</figcaption></figure>')
        extra = f'<img src="{d.name}/sprite_strip.png" class="px strip">'
    else:
        extra = f'<img src="{d.name}/all_frames.jpg" class="strip">'
    note = f"<p class=note>{html.escape(NOTES[d.name])}</p>" if d.name in NOTES else ""
    rows.append(f'<section id="{d.name}"><h2>{d.name}</h2><p>{html.escape(meta)}</p>{note}<div class=row>{"".join(cells)}</div>{extra}</section>')

(HERE / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Experiment 17 - Wan 2.2 walk clips</title>
<style>body{{background:#16181d;color:#d5d8df;font:14px/1.45 system-ui,sans-serif;margin:16px}}h2{{margin:28px 0 2px;color:#fff}}
p{{margin:2px 0 8px;color:#9aa1ad}}.note{{color:#e6c27a}}.row{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}}
figure{{margin:0}}figcaption{{font-size:12px;color:#9aa1ad}}.px{{image-rendering:pixelated}}.strip{{display:block;margin-top:8px;max-width:100%}}
nav a{{color:#8ab4f8;margin-right:12px}}</style>
<h1>Experiment 17 - walk cycles from Wan 2.2 image-to-video</h1>
<nav>{"".join(f'<a href="#{html.escape(r.split(chr(34))[1])}">{html.escape(r.split(chr(34))[1])}</a>' for r in rows)}</nav>
{"".join(rows)}""")
print("wrote", HERE / "index.html", len(rows), "clips")

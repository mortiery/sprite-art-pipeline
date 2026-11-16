"""Build a self-contained copy of the Phaser demo for a static web host.

    python3 tools/build_demo.py                 # -> dist/  (index.html at the top level)
    python3 tools/build_demo.py --out site --busts
    python3 tools/build_demo.py --characters rin,kaela

Copies the demo page, the loader, Phaser and - per character - only what the game loads: the manifest
(without the pipeline's QC log and prompts) and the walk and face sheets. Upload the whole output folder; it
needs no server-side code. Check it locally with
    python3 -m http.server 8000 --directory dist      # then open http://127.0.0.1:8000/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO, CHARS = ROOT / "phaser-demo", ROOT / "characters"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist", help="output folder (replaced on every build)")
    ap.add_argument("--characters", default="", help="comma-separated names (default: all in characters/index.json)")
    ap.add_argument("--busts", action="store_true", help="also copy the 1024 px busts (about 1 MB each)")
    a = ap.parse_args()
    out = (ROOT / a.out).resolve()
    if out == ROOT or ROOT not in out.parents:
        raise SystemExit(f"refusing to replace {out}: the output folder must be inside {ROOT}")

    entries = json.loads((CHARS / "index.json").read_text())
    if a.characters:
        wanted = {n.strip() for n in a.characters.split(",") if n.strip()}
        entries = [e for e in entries if e.split("/")[0] in wanted]
        missing = wanted - {e.split("/")[0] for e in entries}
        if missing:
            raise SystemExit(f"not in characters/index.json: {', '.join(sorted(missing))}")
    if not entries:
        raise SystemExit("no characters to build")

    shutil.rmtree(out, ignore_errors=True)
    (out / "vendor").mkdir(parents=True)
    page = (DEMO / "index.html").read_text()
    if "const base = '../characters/';" not in page:
        raise SystemExit("phaser-demo/index.html: character path line not found - update tools/build_demo.py")
    (out / "index.html").write_text(page.replace("const base = '../characters/';", "const base = 'characters/';"))
    shutil.copy(DEMO / "rpgchar.js", out / "rpgchar.js")
    shutil.copy(DEMO / "vendor" / "phaser.min.js", out / "vendor" / "phaser.min.js")

    size = 0
    for entry in entries:
        name = entry.split("/")[0]
        src, dst = CHARS / name, out / "characters" / name
        dst.mkdir(parents=True)
        m = json.loads((CHARS / entry).read_text())
        files = [m["walk"]["image"], m["faces"]["image"]] + ([m["walkAnim"]["image"]] if "walkAnim" in m else [])
        if a.busts:
            (dst / "busts").mkdir()
            files += list(m.get("busts", {}).values())
        else:
            m.pop("busts", None)
        for f in files:
            shutil.copy(src / f, dst / f)
            size += (dst / f).stat().st_size
        for key in ("qc", "views"):                       # pipeline bookkeeping / files not shipped
            m.pop(key, None)
        m["source"] = {k: m.get("source", {}).get(k) for k in ("desc", "seed")}
        (dst / Path(entry).name).write_text(json.dumps(m, indent=1))
    (out / "characters" / "index.json").write_text(json.dumps(entries, indent=2))
    size += sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and "characters" not in f.parts)
    print(f"built {out} with {len(entries)} characters ({', '.join(e.split('/')[0] for e in entries)}), "
          f"{size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

"""Experiment 17: walk cycles from Wan 2.2 image-to-video.

    python3 experiments/17_wan_video/run.py <tag> --char rin --dir left [--fast] [--loop] [--size 640] [--length 81]

Writes experiments/17_wan_video/<tag>/: frames/ (raw PNG frames), clip.webp, stab_sheet.jpg (every 2nd
stabilised frame), cycle.json (gait-cycle analysis), cycle_<n>.png + .webp (sampled loop).
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline import video as V, workflows as W          # noqa: E402
from pipeline.comfy import Comfy                          # noqa: E402

HERE = Path(__file__).parent
from pipeline import prompts as PR                        # noqa: E402
VIEW = PR.VIDEO_VIEW
PROMPTS = {"treadmill": PR.WALK_VIDEO}
PROMPTS["spin"] = ("2D anime game character turnaround animation. The small chibi character stands upright in a relaxed "
                   "neutral pose with the arms at the sides and slowly rotates on the spot like a figure on a turntable: "
                   "one full smooth 360 degree turn, showing the front, the side, the back, the other side and the "
                   "front again. The pose does not change while turning. Everything the character wears and carries "
                   "stays attached and is seen from every angle. The character stays exactly in the centre of the "
                   "frame at the same size. Fixed static camera, flat uniform {bg} background, no shadow, no floor.")
PROMPTS["halfturn"] = ("2D anime game character turnaround animation. The small chibi character stands upright in a relaxed "
                       "neutral pose with the arms at the sides and slowly turns around on the spot like a figure on a "
                       "turntable: half a turn, from the front view to the back view. {view} The pose does not change while "
                       "turning. Everything the character wears and carries stays attached, keeps its shape and is seen "
                       "from every angle. The character stays exactly in the centre of the frame at the same size. Fixed "
                       "static camera, flat uniform {bg} background, no shadow, no floor.")
TURN = {"left": "It turns towards the left side of the picture: halfway through the turn it is seen in profile, facing the left edge of the picture.",
        "right": "It turns towards the right side of the picture: halfway through the turn it is seen in profile, facing the right edge of the picture."}
NEG_EXTRA = PR.WAN_NEG_EXTRA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--char", default="rin"); ap.add_argument("--dir", default="left")
    ap.add_argument("--prompt", default="treadmill"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--loop", action="store_true", help="first-last-frame: end frame = start frame")
    ap.add_argument("--size", type=int, default=640); ap.add_argument("--length", type=int, default=81)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float); ap.add_argument("--shift", type=float)
    ap.add_argument("--bg", default="magenta"); ap.add_argument("--frames", default="3,4,6,8")
    ap.add_argument("--end-dir", help="use the standing view of this direction as the last frame")
    ap.add_argument("--turn", choices=["left", "right"], help="halfturn prompt: side the character faces mid-turn")
    ap.add_argument("--view", help="override the view description of the prompt")
    ap.add_argument("--lora", action="append", default=[], help="name:strength (applied to both experts)")
    a = ap.parse_args()
    out = HERE / a.tag
    fdir = out / "frames"
    if not list(fdir.glob("*.png")):
        c = Comfy()
        src = V.start_frame(ROOT / f"characters/{a.char}/raw/idle_{a.dir}.png", a.size)
        out.mkdir(parents=True, exist_ok=True)
        src.save(out / "start.png")
        name = c.upload_image(out / "start.png", f"wan_{a.char}_{a.dir}_{a.size}.png")
        end = name if a.loop else None
        if a.end_dir:
            V.start_frame(ROOT / f"characters/{a.char}/raw/idle_{a.end_dir}.png", a.size).save(out / "end.png")
            end = c.upload_image(out / "end.png", f"wan_{a.char}_{a.end_dir}_{a.size}.png")
        prompt = PROMPTS[a.prompt].format(view=a.view or (TURN[a.turn] if a.turn else VIEW[a.dir]), bg=a.bg, props="", desc="the character in the first frame")
        loras = [(n.rsplit(":", 1)[0], float(n.rsplit(":", 1)[1])) for n in a.lora]
        g = W.wan22_i2v(prompt, name, end=end, width=a.size, height=a.size, length=a.length,
                        seed=a.seed, fast=a.fast, steps=a.steps, cfg=a.cfg, shift=a.shift, loras=loras,
                        negative=W.WAN_NEG + NEG_EXTRA, prefix=f"ccr/wan_{a.tag}")
        t0 = time.time()
        files = c.run_video(g, lambda lat: W.wan_decode(lat, prefix=f"ccr/wan_{a.tag}"), fdir)
        secs = round(time.time() - t0, 1)
        for f in files:
            if f.suffix == ".webp":
                f.rename(out / "clip.webp")
        (out / "run.json").write_text(json.dumps({**vars(a), "prompt_text": prompt, "seconds": secs}, indent=1))
        print(f"generated {len(files)} files in {secs}s")
    analyse(out, a.prompt in ("spin", "halfturn"))


def analyse(out: Path, spin: bool = False, counts=(4, 8)) -> None:
    """Key, lock and sample a clip. Outputs (all in the clip folder):
    all_frames.jpg      every 2nd frame of the keyed, stabilised clip
    loop_hires.webp     the detected gait cycle at source resolution (plays as a loop)
    sprite_<style>_<n>.gif   48 px sprite loop, n frames per cycle, style crisp / smooth, shown at 4x
    sprite_strip.png    standing sprite + the 8 walk frames; top row crisp, bottom row smooth
    """
    from PIL import Image
    from pipeline import pixelize as P, sprites as S
    keyed = V.key_frames(V.load_frames(out / "frames"))
    if spin:                                               # turn clips: no stabilising, only the side-view pick
        V.strip(keyed, scale=0.25, cols=14).convert("RGB").save(out / "all_frames.jpg", quality=88)
        half = keyed[:len(keyed) // 2 + 1] if len(keyed) > 60 else keyed
        r = V.turn_profile(half)
        V.strip([keyed[0], keyed[r["index"]], keyed[len(half) - 1]], scale=0.5).save(out / "views.png")
        (out / "turn.json").write_text(json.dumps(r))
        print(out.name, r); return
    keyed, scales = V.normalise_size(keyed)
    cell = S.core_height(keyed[0]) / 44
    locked, stats = V.lock_frames(keyed, cell)
    V.strip(locked[::2], scale=0.25, cols=14).convert("RGB").save(out / "all_frames.jpg", quality=88)
    cyc = V.find_cycle(locked)
    (out / "cycle.json").write_text(json.dumps({"lock": stats, **cyc}, indent=1))
    print(out.name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cyc.items()
                     if k not in ("legs_vs_standing", "autocorr")}, "drift", stats["drift_x"], stats["drift_y"],
          "head_error", round(float(sorted(stats["head_error"][cyc["start"]:cyc["start"] + cyc["period"]])[cyc["period"] // 2]), 4),
          "amp", max(cyc["legs_vs_standing"][cyc["start"]:cyc["start"] + cyc["period"]]),
          "zoom", round(min(scales), 3), round(max(scales), 3))
    loop = locked[cyc["start"]:cyc["start"] + cyc["period"]]
    V.save_anim([f.resize((320, 320), Image.LANCZOS) for f in loop], out / "loop_hires.webp", fps=16)
    rows = []
    for style, method in (("crisp", "nearest"), ("smooth", "kcentroid")):
        for n in counts:
            sel, idx = V.sample_cycle(locked, cyc["start"], cyc["period"], n)
            px = V.locked_sprite_frames([locked[0]] + sel, cell, (stats['cx'], stats['feet']), method=method)
            pal = P.build_palette(px, 64)
            px = [P.apply_palette(f, pal) for f in px]
            V.save_anim(px[1:], out / f"sprite_{style}_{n}.gif", fps=n * 16 / cyc["period"], scale=4)
        rows.append(V.strip(px, scale=4))
    sheet = Image.new("RGB", (rows[0].width, rows[0].height * 2))
    sheet.paste(rows[0], (0, 0)); sheet.paste(rows[1], (0, rows[0].height))
    sheet.save(out / "sprite_strip.png")


if __name__ == "__main__":
    main()

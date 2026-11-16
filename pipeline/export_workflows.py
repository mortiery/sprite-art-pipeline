"""Write one API-format JSON per pipeline stage into workflows/ (for `POST /prompt` or MCP `enqueue_workflow`).

    python -m pipeline.export_workflows

The JSON files are generated from pipeline/workflows.py + pipeline/prompts.py; edit those, not the JSON.
Tokens in ALL CAPS (CHARACTER TAGS, CHARACTER DESCRIPTION, *.png names) are the values to replace.
"""
from pathlib import Path

from . import prompts as PR
from . import workflows as W
from .character import ANIMA_CONCEPT

DESC = "CHARACTER DESCRIPTION"
BG = "solid flat pure magenta (#FF00FF)"
Q, F = PR.PROMPTS["qwen21"], PR.PROMPTS["flux3"]


def main(out: str = "workflows") -> None:
    o = Path(out)
    walk = PR.WALK_VIDEO.format(view=PR.VIDEO_VIEW["left"], bg="magenta", desc=DESC, props="")
    turn = PR.TURN_VIDEO.format(order=PR.TURN_ORDER["front_back"], bg="magenta", props="")
    neg = W.WAN_NEG + PR.WAN_NEG_EXTRA
    g = {
        "01_concept_anima_turbo": W.anima_t2i(ANIMA_CONCEPT.format(tags="CHARACTER TAGS"), seed=1, prefix="concept"),
        "02_chibi_front_qwen21": W.qwen21_edit(Q["chibi"].format(desc=DESC, props="", note="", bg=BG), ["CONCEPT.png"],
                                               seed=1001, prefix="chibi"),
        "03_back_qwen21": W.qwen21_edit(Q["back"].format(desc=DESC, props="", note="", bg=BG), ["CHIBI.png"],
                                        seed=2001, prefix="back"),
        # side views: half turn from the front view (first frame) to the back view (last frame); swap the two
        # images and say "from the back view to the front view" for the other half of the turn
        "04_turn_clip_wan22": W.wan22_i2v(turn, "STAND_DOWN.png", end="STAND_UP.png", width=640, height=640, length=49,
                                          seed=5001, fast=True, negative=neg, prefix="turn"),
        # fallback for a side no turn clip shows: the edit model draws it from front and back view
        "04b_side_view_qwen21": W.qwen21_edit(Q["side"].format(FACING="RIGHT", facing="right", desc=DESC, props="", note="",
                                                                bg=BG), ["CHIBI.png", "BACK.png"], seed=5501, prefix="side_right"),
        # walk: the standing view is first and last frame
        "05_walk_clip_wan22": W.wan22_i2v(walk, "STAND_LEFT.png", end="STAND_LEFT.png", width=640, height=640, length=81,
                                          seed=4001, fast=True, negative=neg, prefix="walk_left"),
        "05_walk_clip_wan22_20steps": W.wan22_i2v(walk, "STAND_LEFT.png", end="STAND_LEFT.png", width=640, height=640,
                                                  length=81, seed=4001, negative=neg, prefix="walk_left"),
        # second job of every clip (after POST /free): decode the latent that the clip workflow saved
        "06_clip_decode_wan": W.wan_decode("ccr/walk_left_latent_00001_.latent [output]", prefix="walk_left"),
        "07_face_neutral_qwen21": W.qwen21_edit(Q["portrait"].format(expression=PR.EMOTIONS["neutral"], bg=BG),
                                                ["CONCEPT.png"], seed=3001, prefix="face_neutral"),
        "08_face_emotion_qwen21": W.qwen21_edit(Q["expression"].format(expression=PR.EMOTIONS["angry"]),
                                                ["FACE_NEUTRAL.png"], seed=3001, prefix="face_angry"),
        "alt_chibi_flux3": W.flux3_edit(F["chibi"].format(desc=DESC, props="", note="", bg=BG), ["CONCEPT.png"], width=1024,
                                        height=1024, seed=1001, prefix="chibi"),
        "alt_spritesheet_lora_flux2_klein": W.flux2_klein_base_edit(
            W.SVNTAX_SHEET_PROMPT + " White background.", ["CONCEPT.png"], seed=1,
            loras=[("pixel_4walk_small_flux2_klein_base_4b_v1.safetensors", 1.0)], prefix="sheet"),
        "util_cutout_birefnet": W.cutout("IMAGE.png", prefix="cutout"),
    }
    for name, graph in g.items():
        graph.save(o / f"{name}.json")
        print("wrote", o / f"{name}.json")


if __name__ == "__main__":
    main()

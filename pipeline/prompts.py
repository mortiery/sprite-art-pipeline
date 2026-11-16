"""Prompt templates for the edit stages, per edit model.

Placeholders: {desc} = one prose sentence describing the character (hair, eyes, outfit, colours),
{bg} = chroma-key colour phrase, {view} = facing description, {expression} = expression description.

Model conventions
  qwen21 : start with the operation ("Redraw ...", "Change ..."); one reference = "the image",
           several = <image1>, <image2>; name what changes plus one "keep everything else" clause.
  flux3  : write a full caption of the *result*; references are <ref_image_0>, <ref_image_1>.
"""

STYLE = "Clean anime line art with flat cel shading."
KEYED = "The background is a {bg} colour filling the whole image edge to edge, perfectly uniform, no gradient, no ground shadow, no text."
CHIBI_BODY = ("exactly two heads tall like a classic 16-bit JRPG overworld sprite: the head is as tall as the entire rest of the "
              "body, with a very large round head, a tiny compact torso, very short stubby legs and arms and small simple feet")


EMOTIONS = {
    "neutral": "calm neutral expression, relaxed eyebrows, closed mouth",
    "happy": "happy, a wide cheerful open-mouth smile, eyes bright, eyebrows raised",
    "angry": "furious, eyebrows sharply lowered and drawn together, eyes glaring, mouth open shouting with clenched teeth",
    "sad": "sad, eyebrows slanted upward in the middle, downcast teary eyes, mouth turned down",
    "surprised": "surprised, eyes wide open, eyebrows raised high, small round open mouth",
    "embarrassed": "embarrassed, strong pink blush across the cheeks, eyes glancing away to the side, wavy awkward closed mouth",
    "determined": "determined, confident smirk, eyebrows lowered, eyes focused straight ahead",
    "hurt": "hurt and in pain, one eye squeezed shut, teeth gritted, a sweat drop on the temple",
}

def note_sentence(note: str) -> str:
    """Optional correction for one stage, e.g. after a reviewer rejected a view."""
    return f" Important: {note.strip().rstrip('.')}." if note and note.strip() else ""


def props_sentence(props: str) -> str:
    """Optional explicit list of carried items; without it the model sometimes omits them in some views."""
    return f" Carried items that must be visible in every picture: {props.strip().rstrip('.')}." if props.strip() else ""


PROMPTS = {
    "qwen21": {
        # refs: [concept]  -> the chibi is also the front standing view of the sprite
        "chibi": ("Redraw the character in the image as a super-deformed chibi, " + CHIBI_BODY + ". Full body from head to feet, "
                  "standing upright and still, facing the viewer, arms hanging down at the sides, feet side by side. The character "
                  "is {desc}.{props} Keep the same hairstyle, hair colour, eye colour, outfit pieces, carried items and colours, "
                  "simplified to the essential shapes.{note} " + STYLE + " " + KEYED),
        # refs: [chibi]  -> the back standing view
        "back": ("Redraw the chibi character in the image seen from directly behind: a back view of the same character in the "
                 "same standing pose, at the same size and position in the picture, facing away from the viewer. The face is not "
                 "visible; the picture shows the back of the hair or headwear, the back of the outfit and the heels. The "
                 "character is {desc}.{props} Everything worn or carried is still there and is drawn as it looks from behind; "
                 "an item worn on the back is fully visible. Nothing is mirrored: whatever is on the character's right-hand "
                 "side appears on the right side of this picture. Nothing is added that the character does not have.{note} "
                 "Same chibi proportions, colours and art style. " + STYLE + " " + KEYED),
        # refs: [concept]
        "portrait": ("Redraw the character in the image as a face portrait for an RPG dialogue box: a tight head-and-shoulders "
                     "close-up, facing the viewer, head upright and centred. The head is large and fills most of the frame; the "
                     "top of the hair or headwear is just inside the upper edge and only the top of the shoulders is visible at "
                     "the bottom edge. Expression: {expression}. Keep the same face, hairstyle, outfit, colours and anime art "
                     "style. " + KEYED),
        # refs: [neutral portrait]
        "expression": ("Change the character's facial expression: {expression}. Keep the hairstyle, outfit, colours, art style, "
                       "framing, pose and background unchanged."),
    },
    "flux3": {
        "chibi": ("A chibi super-deformed version of the character in <ref_image_0>, full body, standing upright and facing the "
                  "viewer, arms at the sides. Chibi proportions: the character is " + CHIBI_BODY + ". The character is {desc}.{props} "
                  "Same colours and outfit as the reference, simplified to the essential shapes.{note} " + STYLE + " " + KEYED),
        "back": ("A back view of the chibi character in <ref_image_0>: the same character in the same standing pose and at the "
                 "same size, seen from directly behind, facing away from the viewer, face not visible. The character is "
                 "{desc}.{props} Everything worn or carried is drawn as it looks from behind; nothing is mirrored and nothing "
                 "is added.{note} " + STYLE + " " + KEYED),
        "portrait": ("Face portrait of the character from <ref_image_0> for an RPG dialogue box: a tight head-and-shoulders "
                     "close-up, facing the viewer, head upright and centred. The head is large and fills most of the frame; the "
                     "top of the hair or headwear is just inside the upper edge and only the top of the shoulders is visible at "
                     "the bottom edge. Expression: {expression}. The character is {desc}. Same face, hairstyle, colours and "
                     "outfit as the reference. " + STYLE + " " + KEYED),
        "expression": ("The same bust portrait as <ref_image_0>: same character, same framing, same pose, same hairstyle, same "
                       "outfit, same colours, same art style and same plain background. Only the facial expression changes: "
                       "{expression}."),
    },
}


# ---------------------------------------------------------------- Wan 2.2 walk clips ----
# Image-to-video prompt: the standing view of one direction is the first frame; the model animates it.
VIDEO_VIEW = {
    "down": "seen from the front, facing the camera, walking towards the viewer",
    # no "walking away from the viewer": the model then really walks off and shrinks
    "up": "seen from behind in back view, facing away from the camera, marching on the spot without moving away",
    "left": "in strict side view, in profile, facing left, walking to the left",
    "right": "in strict side view, in profile, facing right, walking to the right",
}
WALK_VIDEO = ("2D anime game sprite animation. The small chibi character {view}, walks on the spot like on a "
              "treadmill: a steady, even, looping walk cycle with clear steps, the legs alternate, the feet lift off "
              "the ground, the arms swing slightly, the body bobs a little with every step. The character stays "
              "exactly in the centre of the frame at the same size and keeps the same direction the whole time. "
              "The character is {desc}; its outfit and colours stay exactly as in the first frame, also on parts "
              "that become visible while walking. Everything the character carries stays where it is.{props} Fixed static camera, flat uniform {bg} "
              "background, no shadow, no floor.")
# Side views: the clip is pinned to the front view and the back view (first and last frame); the model only
# has to turn the character, and the side view is the middle of that turn. Wan ignores "turn to the left /
# right" and always spins the same way, so the left view comes from the turn front -> back and the right
# view from the turn back -> front (experiment 17).
TURN_VIDEO = ("2D anime game character turnaround animation. The small chibi character stands upright in a relaxed neutral "
              "pose with the arms at the sides and slowly turns around on the spot like a figure on a turntable: half a "
              "turn, {order}. Halfway through the turn it is seen exactly from the side, in profile. The pose does not "
              "change while turning. Everything the character wears and carries stays attached, keeps its shape and is "
              "seen from every angle.{props} The character stays exactly in the centre of the frame at the same size. "
              "Fixed static camera, flat uniform {bg} background, no shadow, no floor.")
TURN_ORDER = {"front_back": "from the front view to the back view", "back_front": "from the back view to the front view"}
# appended to the official Wan negative prompt: camera move, zoom, turning around, turning the head, shadow,
# ground, blur, motion blur
WAN_NEG_EXTRA = "，镜头移动，镜头推拉，转身，转头，影子，地面，模糊，运动模糊"

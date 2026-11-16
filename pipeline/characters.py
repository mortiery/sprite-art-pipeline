"""Test characters used throughout the experiments. Each has a tag-style and a prose description so
both tag-trained (Anima) and natural-language (FLUX) models can be prompted fairly."""

CHARS = {
    "rin": {
        "tags": "1girl, solo, young adventurer, short messy red hair, green eyes, brown leather vest over white long-sleeved shirt, blue scarf, dark navy shorts, brown knee-high boots, brown leather gloves, belt with pouches",
        "prose": "a young female adventurer with short messy red hair and green eyes, wearing a brown leather vest over a white long-sleeved shirt, a blue scarf around her neck, dark navy shorts, a belt with small pouches, brown leather gloves and brown knee-high boots",
    },
    "garrick": {
        "tags": "1boy, solo, mature male, veteran knight, short grey hair, grey beard, scar across left eyebrow, steel plate armor, red cape, large greatsword on back, brown belt, armored boots, gauntlets",
        "prose": "a burly veteran male knight with short grey hair, a grey beard and a scar across his left eyebrow, wearing steel plate armor with gauntlets and armored boots, a long red cape, a brown belt, and a large greatsword strapped to his back",
    },
    "selka": {
        "tags": "1girl, solo, young witch, long straight silver hair, violet eyes, large purple witch hat with gold band, purple robe with gold trim, white collar, black tights, short brown boots, holding wooden staff with blue crystal",
        "prose": "a young witch with long straight silver hair and violet eyes, wearing a large purple witch hat with a gold band, a purple robe with gold trim and a white collar, black tights and short brown boots, holding a wooden staff topped with a blue crystal",
    },
}

ANIMA_CONCEPT = ("masterpiece, best quality, score_7, safe, anime, {tags}, full body, standing, straight-on, facing viewer, "
                 "looking at viewer, arms at sides, neutral expression, closed mouth, simple background, white background, "
                 "character design, reference sheet style, flat color, clean lineart, anime coloring")

FLUX_CONCEPT = ("Anime character design concept art, full body from head to feet, of {prose}. The character stands upright in a "
                "neutral relaxed pose facing the viewer directly, arms at the sides, calm neutral expression with closed mouth. "
                "Clean crisp line art, flat cel-shaded anime colouring with simple two-tone shading, no gradients. Plain pure white "
                "background, no floor, no cast shadow, no text.")

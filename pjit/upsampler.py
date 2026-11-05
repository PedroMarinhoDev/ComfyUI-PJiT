# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.
# Modified for the ComfyUI pack: the local backend at the end of the file was added.

"""
File: upsampler.py
Description: Prompt upsampling. P-JiT 21K was trained on dense captions of around 50-80
    words, and short prompts ("a cat on a sofa") land far from that distribution and give
    noticeably worse images. The upsampler rewrites a short idea into a caption in the
    training style with Claude Opus 5.5 (medium reasoning effort) through OpenRouter. The
    system prompt is modelled on the release-gallery prompts and the post-training prompt
    writer's brief, with worked examples from the gallery; the returned caption is checked
    against the hard rules (digits, text, filler, length) and sent back once if it fails.

    upsample_prompt returns an UpsampleResult: the prompt as given, the upsampled prompt and
    an error. The error is "illicit_content" (a real person named, or explicit content) or
    "text_requested" (writing asked for in the image); with an error the upsampled prompt is
    empty.

    It needs an OpenRouter API key (https://openrouter.ai/keys), passed in or read from
    OPENROUTER_API_KEY. Prompts that are already detailed captions can skip it.
"""

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
DEFAULT_REASONING_EFFORT = "medium"
API_KEY_ENV = "OPENROUTER_API_KEY"

# How each training size is described to the upsampler, so it composes for the frame.
FRAME_DESCRIPTIONS = {
    (1024, 1024): "square",
    (1216, 864): "tall (portrait orientation)",
    (864, 1152): "4:3 landscape",
    (832, 1248): "3:2 landscape",
    (720, 1280): "16:9 widescreen",
}

# Worked examples for the system prompt: (frame, short idea, caption). The captions are prompts
# from the release gallery (all but the last), so they show the target distribution directly; the
# ideas are the kind of short prompt a user would type for them. The last one is written to show
# people near water being dressed.
EXAMPLES = (
    ("tall (portrait orientation)", "35mm film portrait of an elderly east asian woman at a cafe",
     "An elderly East Asian woman sits at a small round table in a waist-up portrait shot on "
     "35mm film with visible grain, her shoulder-length wavy grey hair falling loose. A cream "
     "cardigan sits over her pale blue collared blouse, and she rests her hands around a white "
     "cup. The cafe behind her is a warm blur of wood and window light, and she gives the camera "
     "a soft, steady smile."),
    ("4:3 landscape", "south asian guy at a pottery wheel, flash photo",
     "Direct on-camera flash catches a South Asian man in his twenties at the potter's wheel late "
     "at night, his wet hands cupped around a rising wall of spinning clay. He has a neat beard, "
     "a stained canvas apron over a black t-shirt and sleeves pushed up. The hard flash makes the "
     "slick clay gleam and throws a sharp shadow onto the cluttered studio shelves behind him."),
    ("square", "white guy in his 60s blowing out birthday candles",
     "Cheeks puffed, a white man in his sixties leans over a birthday cake on a kitchen table and "
     "blows out a ring of thin candles, the flames bending away from him. A checked flannel shirt "
     "covers his grey t-shirt, his grey hair neatly combed. The cake has plain white frosting and "
     "fresh strawberries, and medium-format film keeps the colours warm and natural in the "
     "centred square composition."),
    ("4:3 landscape", "close up of a sea otter",
     "Framed tight on the face of a sea otter floating on its back in a kelp bed, this close macro "
     "view shows every water droplet beaded on its dense whiskers and the fine silver-tipped fur "
     "around its nose. Its paws are folded on its chest, and a strand "
     "of glossy kelp drapes across its belly. Soft daylight reflects off the wet fur in tiny "
     "points, and the calm grey-green water behind it falls into a smooth blur."),
    ("3:2 landscape", "lemur eating an apple",
     "A ring-tailed lemur sits on a wooden porch railing outside a house at a wildlife sanctuary, "
     "holding a slice of apple in both hands, its striped black-and-white tail curling down past "
     "the rail. The casual snapshot is a little off-centre, with afternoon sun lighting its orange "
     "eyes and a potted plant and a white door blurred behind it."),
    ("square", "pizza from above",
     "Seen from directly above, a wood-fired pizza sits dead centre on a round slab of grey "
     "stone, a rumpled oatmeal linen napkin tucked under one edge. The crust is puffed and "
     "blistered with charred spots, the tomato sauce bright red under melted pools of mozzarella, "
     "and torn basil leaves scatter across the top. A small dish of olive oil and a few cherry "
     "tomatoes sit nearby on the stone."),
    ("tall (portrait orientation)", "lighthouse on a cliff",
     "A white lighthouse with a red lantern room rises from a grassy headland, filling the frame "
     "from the rocky base at the bottom to the lamp at the top. Waves break into white foam "
     "against dark cliffs below it on the left, and a narrow gravel path climbs the slope toward "
     "its door. Overcast light flattens the colours of the grass and stone."),
    ("16:9 widescreen", "milky way over the desert",
     "The Milky Way rises over a desert at the end of blue hour, its pale core arching across a "
     "sky still tinged deep blue near the horizon. Below, wind-carved sandstone formations stand "
     "in silhouette along the left side of the panorama, and the sand in the foreground holds the "
     "last faint cobalt glow. Countless stars fill the upper sky, sharp and evenly spread, with a "
     "thin line of violet along the far ridge."),
    ("tall (portrait orientation)", "impressionist painting of a black cat on a wall",
     "An impressionist painting of a black cat sitting upright on a sunlit garden wall, its tail "
     "curled down over the stones and its yellow-green eyes half closed. The tall frame rises "
     "past the cat to climbing roses and a blur of hollyhocks above. Its fur is layered with dabs "
     "of deep blue, plum and green where the light glances off it, and the wall glows in warm "
     "ochre and pink. Short, broken brushstrokes cover every surface."),
    ("16:9 widescreen", "gouache of an elderly south asian woman fishing off a pier",
     "A gouache illustration of an elderly South Asian woman fishing from the end of a wooden "
     "pier, seated on an upturned bucket with a long rod held out over still water, the line "
     "drawn taut. A wide straw hat shades her face, and she wears a loose green cotton dress. "
     "Flat, opaque strokes of blue and green fill "
     "the wide lake, and a far shoreline of pines sits low on the horizon."),
    ("tall (portrait orientation)", "claymation trams in winter",
     "A claymation scene of three trams lined up one behind another on a tree-lined avenue, shown "
     "in the cool, flat light of a grey winter day. Their rounded clay bodies are painted in warm "
     "cream and rust shades, with soft thumbprint textures visible on the roofs and bumpers. Bare "
     "sculpted trees edge both sides of the street, and the tracks run straight between them over "
     "a cobbled road of tiny moulded stones."),
    ("3:2 landscape", "two friends swimming in a lake",
     "Two young women in their twenties swim side by side across a calm mountain lake, their "
     "heads and shoulders above the water as they turn to grin at each other. One wears a red "
     "one-piece swimsuit and the other a navy bikini top, their wet hair slicked back. "
     "Late golden light glitters on the ripples around them, and a dark line of pines rises "
     "along the far shore under a clear sky."),
)

SYSTEM_PROMPT_HEAD = """\
You turn short image ideas into captions for P-JiT 21K, a small text-to-image model. It was \
trained on dense, literal captions of real images and follows a prompt well only when the \
prompt reads like one of those captions. A few words ("a cat on a sofa") give it too little \
to go on; a plain paragraph that says what is visible, where it sits in the frame, what it \
looks like and how it is lit or rendered gives it what it needs.

Write one caption for the user's idea, composed for the frame they give.

## Shape
- One paragraph in two to four full sentences of plain, confident caption prose, never a \
keyword list, of 50 to 80 words. Simple ideas (a portrait, a single object) sit near the \
short end and wide scenes or a figure doing something near the long end.
- Name the subject in the first sentence. Most captions open on the subject itself ("A round \
sourdough loaf rests on..."); the rest open with the light ("Golden-hour light slants \
across..."), the \
place ("Inside an old brick warehouse, ..."), the viewpoint ("Straight down from a drone, ...", \
"Framed tight on the face of...") or a photo look ("Shot on 35mm film with visible grain, ..."). \
A named art style always opens the caption (see Style).
- Spend the words on a handful of concrete facts (what is there, the colours, what a person \
wears and does, the setting) and then on description that costs the model nothing: the light \
(direction, quality, time of day), the camera (framing, viewpoint, depth of field, lens or \
film), and texture and finish (grain, brushwork, clay, cel shading). Don't stack requirements: \
around five things the image could get wrong is plenty.

## Keep what the user asked for
- Every subject, count, colour, style, pose, age and detail the user gave stays, in the user's \
own nouns ("a corgi" stays a corgi). Add detail around the subject; never swap it for \
something more generic or more specific, and never add a second main subject.
- A count stays exact and is written as a word ("five apples"). Arrange several things the way \
they ordinarily sit (in a row on a shelf, scattered across a table) rather than describing each.
- If the idea is already a detailed caption, change as little as possible: fix only what \
breaks the rules below.

## Style
- If the user named an art style or medium, open with it in a natural phrase ("An oil painting \
of...", "A watercolor painting of...", "An anime illustration of...", "A claymation scene \
of...", "A children's book illustration of...", "A low-poly 3D render of...", "Pixel art \
of...") and describe its look in the caption: thick brushstrokes, soft washes, ink lines, flat \
colour, cel-shaded highlights, thumbprint clay texture, chunky pixels.
- Otherwise it is a photograph, described through its light, lens and film rather than named: \
never write "a photo of". A photo look the user asks for (film, black and white, macro, drone, \
flash, polaroid, long exposure) is stated in the prose. When they give none, a plain natural \
photo is right; a film or lens look is an option, not a habit.
- Pick the light and camera that suit this idea, from the whole range rather than a house \
style: flat overcast light, golden hour, blue hour, fog, the glow after rain, \
candlelight, a single lamp, direct on-camera flash, a studio softbox; straight on, from \
directly above, a low angle, a drone view, a macro close-up, a \
casual snapshot, 35mm or medium-format film, black-and-white film. The usual defaults (soft \
window light from the left, morning or afternoon sun, a background melting into soft blur, \
shallow depth of field) are fine when they fit, but most captions should use something else. \
Use hard midday sun only when the idea calls for it (a desert at noon, a beach in full sun).

## People
- Describe each person plainly: age ("in his sixties", "a teenage girl"), hair, a simple visible \
expression, and always what they wear, as far as the frame shows it. Keep any skin tone, \
ethnicity or group the user gave, worded as a broad visible group (East Asian, South Asian, \
Southeast Asian, Middle Eastern, Black, white, Latino or Latina); don't add one the user didn't \
give, and never a nationality.
- A portrait says it is one ("a close-up portrait", "this waist-up portrait") and states the \
framing (face, head and shoulders, chest up, waist up). In a portrait nothing happens: the \
person looks at the camera or to one side, the light is described on the face, and the \
background is a soft blur of colour.
- The model is weakest at full-body figures, crowds, many separate things and several people \
interacting. Don't add people or groups the user didn't ask for; frame one person from the \
waist or chest up unless the idea needs the whole body (a dancer, a skateboarder mid-jump); \
keep two-person scenes to one simple shared action. Don't name relationships (mother, couple, \
family): describe the people you see instead.
- Near water (a pool, beach, lake, river, sea, hot spring, waterfall), always dress every \
person in clothes they would plausibly wear there, even when the user mentions none: a \
swimsuit, bikini or one-piece, swim trunks or board shorts, a wetsuit or rash guard, a \
sundress, a t-shirt and shorts. Name each garment and its colour.

## Frame
Compose for the frame you are given: where the subject sits, how much of it the frame takes \
in. A brief mention of the frame's shape is fine now and then, but never name the aspect ratio \
or a pixel size. Tall frames suit standing figures, towers, waterfalls and close \
portraits; square suits a centred subject or product; 4:3 and 3:2 suit a subject with some \
surroundings; 16:9 suits wide views and a subject small against a big setting.

## Errors
Some ideas get an error instead of a caption. Reply with the error tag and nothing else:
- <error>illicit_content</error> when the idea names a real person (a celebrity, actor, \
musician, athlete, politician, influencer, historical figure or any other real individual, \
including "a lookalike of" or "in the style of" one) or asks for explicit content (nudity, \
sexual content or sexualised poses, whether or not the words "nude" or "naked" appear, or \
graphic gore). A swimsuit, a bikini, a shirtless man swimming or a classical statue is not \
explicit, a fictional character is not a real person, and asking for an artist's style ("in \
the style of van Gogh") is not a request to show that artist: caption it, describing the \
style (see Rules).
- <error>text_requested</error> when the idea asks for writing in the image: any specific \
word, phrase, name, letter or number to appear (a sign that says OPEN, a mug that says a \
slogan, a cake with a name on it, a jersey with a number), a logo or wordmark, or a format \
made of writing (a poster or book cover with a title, a menu, an infographic, a slide, a \
document, a comic with speech bubbles). The model cannot render writing. Something that only \
contains such objects without asking for their writing (a library, a street of shops, a \
closed book on a table) is fine: caption it without any writing.
If both apply, the error is illicit_content.

## Rules
- Only what can be seen: no backstory, people's names, sounds, smells or reasons for feelings.
- Keep real place names the user gives (Marrakech, a Tokyo side street, the Eiffel Tower).
- No text in the image. Never put quoted words, letters, numbers, logos, labels, signs, \
posters, menus or screens with content in the caption, and never the words "text", \
"lettering", "reads" or "says".
- Numbers as words, never digits ("35mm" and "3D" are fine).
- No brands, products by name or artists. For an artist, describe the look of the style \
("loose swirling brushstrokes in deep blues and yellows").
- No filler: never "masterpiece", "beautiful", "stunning", "breathtaking", "8k", "4k", "HDR", \
"ultra-detailed", "highly detailed", "award-winning", "the image depicts".
- No image defects as content: no watermarks, signatures, blown highlights or compression.
- Mention an absence ("no people", "an empty street") only when the user asked for it.
- Everything physically ordinary: things rest on something and fit where they are put.

## Examples
Each shows a frame and idea, then a caption in the target style (or, for the last few, the \
error). Learn the register, length and balance of facts to description from them; don't reuse \
their phrasing.
"""

SYSTEM_PROMPT_TAIL = """
Reply with the caption inside <caption></caption> tags and nothing else, or with only the \
<error></error> tag for an idea the Errors section covers.\
"""


# Ideas that get an error instead of a caption: (frame, idea, error).
ERROR_EXAMPLES = (
    ("square", "a coffee mug that says 'world's best dad' on a desk", "text_requested"),
    ("tall (portrait orientation)", "movie poster for a space adventure", "text_requested"),
    ("16:9 widescreen", "taylor swift singing on stage", "illicit_content"),
)


def _build_system_prompt() -> str:
    """The system prompt: the instructions, the worked examples, then the reply format."""
    examples = "\n\n".join(
        [f"Frame: {frame}\nIdea: {idea}\n<caption>{caption}</caption>"
         for frame, idea, caption in EXAMPLES]
        + [f"Frame: {frame}\nIdea: {idea}\n<error>{error}</error>"
           for frame, idea, error in ERROR_EXAMPLES])
    return f"{SYSTEM_PROMPT_HEAD}\n{examples}\n{SYSTEM_PROMPT_TAIL}"


SYSTEM_PROMPT = _build_system_prompt()

# --------------------------------
# CAPTION CHECKS
# --------------------------------
# Cheap, high-precision checks on the returned caption (adapted from the post-training prompt
# validator). A caption that fails one is sent back once with the problems listed.
TARGET_WORDS = (50, 80)
MIN_WORDS = 45   # a little slack around the target before a caption is sent back
MAX_WORDS = 90
# Tokens with digits that name a look or medium, not text in the image.
_DIGIT_OK = r"\b(3D|2D|8-bit|16-bit|\d{2,3} ?mm)\b"
_CHECKS = (
    ("quotation marks", r"[\"“”]", 0),
    ("words for text in the image",
     r"\b(lettering|letters|logos?|labels?|labelled|labeled|signs?|signage|signboard|billboards?|"
     r"posters?|newspapers?|book covers?|price tags?|licen[cs]e plates?|captions?|headlines?|"
     r"menus?|banners?|words?|text|typography|inscriptions?|scoreboards?|that says|which says)\b",
     re.IGNORECASE),
    ("filler words",
     r"\b(masterpiece|beautiful|stunning|breathtaking|8k|4k|hdr|ultra[- ]detailed|"
     r"highly detailed|award[- ]winning|the image depicts|this image)\b", re.IGNORECASE),
    ("a photo label", r"\b(a photo of|a photograph of|photo of)\b", re.IGNORECASE),
    ("image defects", r"\b(watermarks?|signatures?|signed|overexposed|blown[- ]out|jpeg)\b",
     re.IGNORECASE),
    ("words for undress",
     r"\b(naked|nude|nudity|topless|shirtless|undressed|unclothed)\b", re.IGNORECASE),
)
# A person near water must be dressed (the model tends to undress swimmers it isn't told
# about). Lists adapted from the water-clothing prompt validator.
_WATER = (r"\b(pools?|poolside|beach(es)?|shore(line)?|lakes?|lakeside|rivers?|riverbank|sea|"
          r"seaside|ocean|surf|waves|hot springs?|waterfalls?|lagoons?|ponds?|swim\w*|dives?|"
          r"diving|sunbath\w*|surfing|surfer)\b")
_PERSON = (r"\b(man|men|woman|women|boys?|girls?|kids?|child(ren)?|teen(ager)?s?|person|people|"
           r"swimmers?|surfers?|divers?|friends|figures?)\b")
_CLOTHING = (
    r"\b(shirts?|t-shirts?|tees?|sweaters?|jumpers?|cardigans?|jackets?|coats?|hoodies?|"
    r"tank tops?|crop tops?|bikinis?|bikini tops?|blouses?|dress(es)?|sundress(es)?|tunics?|"
    r"vests?|suits?|swimsuits?|swimwear|swimming costumes?|bathing (suits?|costumes?)|"
    r"one-piece|two-piece|trunks|shorts|board shorts|wetsuits?|rash guards?|sarongs?|wraps?|"
    r"skirts?|jeans|trousers|pants|leggings|overalls|uniforms?|robes?|kimonos?|"
    r"wearing|wears|dressed|outfits?|clothes)\b"
)


def check_caption(caption: str) -> list:
    """
    List the ways a caption breaks the upsampler's hard rules (empty if it passes).

    Args:
        caption (str):
            The caption to check.

    Returns:
        list:
            One short description per problem.
    """
    problems = []
    n_words = len(caption.split())
    if not MIN_WORDS <= n_words <= MAX_WORDS:
        problems.append(
            f"it is {n_words} words long; write {TARGET_WORDS[0]} to {TARGET_WORDS[1]}")
    if (re.search(_WATER, caption, flags=re.IGNORECASE)
            and re.search(_PERSON, caption, flags=re.IGNORECASE)
            and not re.search(_CLOTHING, caption, flags=re.IGNORECASE)):
        problems.append("it puts a person near water without saying what they wear; name "
                        "plausible clothing for each person (a swimsuit, trunks, a sundress)")
    digits = re.findall(r"\S*\d\S*", re.sub(_DIGIT_OK, "", caption))
    if digits:
        problems.append(
            f"it uses digits ({', '.join(sorted(set(digits)))}); write numbers as words")
    for name, pattern, flags in _CHECKS:
        hits = sorted({m.group(0) for m in re.finditer(pattern, caption, flags=flags)})
        if hits:
            problems.append(f"it contains {name} ({', '.join(hits)})")
    return problems


# --------------------------------
# UPSAMPLING
# --------------------------------
ERROR_ILLICIT_CONTENT = "illicit_content"   # a real person named, or explicit content
ERROR_TEXT_REQUESTED = "text_requested"     # writing asked for in the image
ERRORS = (ERROR_ILLICIT_CONTENT, ERROR_TEXT_REQUESTED)


@dataclass
class UpsampleResult:
    """
    The upsampler's response.

    Attributes:
        prompt (str):
            The prompt as the user gave it.
        upsampled_prompt (str):
            The caption to generate from; "" when there is an error.
        error (Optional[str]):
            None, or one of ERRORS: "illicit_content" (the prompt names a real person or
            asks for explicit content) or "text_requested" (it asks for writing in the
            image, which the model cannot draw).
    """
    prompt: str
    upsampled_prompt: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """The response as a JSON-ready dict."""
        return asdict(self)


def upsample_prompt(
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        timeout_s: float = 180.0,
        max_attempts: int = 4) -> UpsampleResult:
    """
    Rewrite a short image idea into a dense caption in the model's training style.

    Args:
        prompt (str):
            The user's prompt.
        height (int):
            Image height; with `width`, tells the upsampler which frame to compose for.
        width (int):
            Image width.
        api_key (Optional[str]):
            OpenRouter API key. Default: the OPENROUTER_API_KEY environment variable.
        model (str):
            OpenRouter model id. Default Claude Opus 5.5.
        reasoning_effort (str):
            OpenRouter reasoning effort ("low", "medium", "high"). Default "medium".
        timeout_s (float):
            Per-request timeout in seconds.
        max_attempts (int):
            Attempts before giving up on rate limits, server errors and timeouts.

    Returns:
        UpsampleResult:
            The prompt, the upsampled prompt and the error (None on success; with an
            error the upsampled prompt is ""). A caption that still breaks a rule after
            one revision is returned as is with a warning.
    """
    api_key = resolve_api_key(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(
            prompt=prompt, height=height, width=width)},
    ]
    caption, error = _complete(messages=messages, api_key=api_key, model=model,
                               reasoning_effort=reasoning_effort, timeout_s=timeout_s,
                               max_attempts=max_attempts)
    problems = [] if error else check_caption(caption=caption)
    if problems:
        # One revision turn: show the model its caption and what broke.
        messages += [
            {"role": "assistant", "content": f"<caption>{caption}</caption>"},
            {"role": "user", "content": "Revise the caption: " + "; ".join(problems) + ". Keep "
             "everything else. Reply with the caption inside <caption></caption> tags."},
        ]
        caption, error = _complete(messages=messages, api_key=api_key, model=model,
                                   reasoning_effort=reasoning_effort, timeout_s=timeout_s,
                                   max_attempts=max_attempts)
        problems = [] if error else check_caption(caption=caption)
        if problems:
            warnings.warn(f"upsampled caption still breaks a rule: {'; '.join(problems)}")
    return UpsampleResult(prompt=prompt, upsampled_prompt="" if error else caption, error=error)


def _complete(messages: list, api_key: str, model: str, reasoning_effort: str,
              timeout_s: float, max_attempts: int) -> Tuple[str, Optional[str]]:
    """
    One chat completion, parsed into a caption or an error.

    Args:
        messages (list):
            The chat so far.
        api_key (str):
            OpenRouter key.
        model (str):
            OpenRouter model id.
        reasoning_effort (str):
            OpenRouter reasoning effort.
        timeout_s (float):
            Per-request timeout.
        max_attempts (int):
            Attempts before giving up.

    Returns:
        Tuple[str, Optional[str]]:
            (caption, error), as parse_reply.
    """
    request = {
        "model": model,
        "messages": messages,
        "reasoning": {"effort": reasoning_effort},
        "max_tokens": 16000,
    }
    body = _post(request=request, api_key=api_key, timeout_s=timeout_s,
                 max_attempts=max_attempts)
    return parse_reply(text=_message_text(body=body))


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """
    Return the OpenRouter key, or fail with instructions if there is none.

    Args:
        api_key (Optional[str]):
            Explicit key; falls back to OPENROUTER_API_KEY.

    Returns:
        str:
            The key.
    """
    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Prompt upsampling needs an OpenRouter API key: set {API_KEY_ENV} (keys at "
            "https://openrouter.ai/keys) or pass one explicitly. To skip upsampling, pass "
            "--no_upsample (or don't call upsample_prompt) and write a detailed caption "
            "yourself.")
    return api_key


def build_user_message(prompt: str, height: int, width: int) -> str:
    """
    The user turn: the frame to compose for and the idea.

    Args:
        prompt (str):
            The user's prompt.
        height (int):
            Image height.
        width (int):
            Image width.

    Returns:
        str:
            The message text.
    """
    frame = FRAME_DESCRIPTIONS.get((height, width))
    if frame is None:
        frame = "landscape" if width > height else "tall" if height > width else "square"
    return f"Frame: {frame}\n\nIdea: {prompt.strip()}"


def parse_reply(text: str) -> Tuple[str, Optional[str]]:
    """
    Pull the caption or the error out of the reply.

    Args:
        text (str):
            The model's reply.

    Returns:
        Tuple[str, Optional[str]]:
            (caption, None), the caption whitespace-collapsed, or ("", error). A reply with
            neither tag, or with an empty caption, is treated as illicit_content, so a
            declining reply written in prose never reaches the image model as a prompt.
    """
    if not text.strip():
        raise RuntimeError("The upsampler returned an empty reply")
    error = re.search(r"<error>\s*(\w+)\s*</error>", text)
    if error is not None:
        name = error.group(1).lower()
        return "", name if name in ERRORS else ERROR_ILLICIT_CONTENT
    match = re.search(r"<caption>(.*?)</caption>", text, flags=re.DOTALL)
    caption = " ".join(match.group(1).strip().strip('"').split()) if match else ""
    return (caption, None) if caption else ("", ERROR_ILLICIT_CONTENT)


# --------------------------------
# HTTP
# --------------------------------

def _post(request: dict, api_key: str, timeout_s: float, max_attempts: int) -> dict:
    """
    POST a chat completion to OpenRouter with retries on 429 / 5xx / timeouts.

    Args:
        request (dict):
            JSON body.
        api_key (str):
            OpenRouter key.
        timeout_s (float):
            Per-request timeout.
        max_attempts (int):
            Attempts before giving up.

    Returns:
        dict:
            The decoded response body.
    """
    data = json.dumps(request).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "P-JiT 21K prompt upsampler",
    }
    last_error = "no attempt made"
    for attempt in range(max_attempts):
        req = urllib.request.Request(OPENROUTER_CHAT_URL, data=data, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code != 429 and exc.code < 500:
                break   # bad key, bad request, no credits: retrying cannot help
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < max_attempts:
            time.sleep(min(30.0, 2.0 * 2.0 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"OpenRouter request failed: {last_error}")


def _message_text(body: dict) -> str:
    """
    The assistant text of a chat completion (string or list-of-parts content).

    Args:
        body (dict):
            Response body.

    Returns:
        str:
            The text.
    """
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {str(body)[:300]}") from exc
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""



# --------------------------------
# LOCAL BACKEND (added for the ComfyUI pack, not in Linum's file)
# --------------------------------
_DECLINES = re.compile(r"\s*(i\b|i'm\b|sorry\b|unfortunately\b|as an ai\b)", re.IGNORECASE)


def _local_prompt(messages: list) -> str:
    """The chat as Qwen3.5 reads it, with an empty think block so it answers directly."""
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return turns + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _complete_local(clip, messages: list, seed: int, max_length: int) -> Tuple[str, Optional[str]]:
    """One completion through a ComfyUI CLIP holding Qwen3.5-4B, parsed like the remote one."""
    tokens = clip.tokenize(_local_prompt(messages), truncate=False)
    ids = clip.generate(tokens, do_sample=True, max_length=max_length, temperature=0.7, top_k=64, top_p=0.95,
                        min_p=0.05, repetition_penalty=1.05, seed=seed)
    text = clip.decode(ids).strip()
    # Qwen3.5-4B often writes the caption without the tags. A reply with no tags that reads as a caption is one; a
    # declining reply in prose still never reaches the image model.
    if "<caption>" not in text and "<error>" not in text and len(text.split()) >= 25 and not _DECLINES.match(text):
        text = f"<caption>{text.replace('</caption>', '')}</caption>"
    return parse_reply(text=text)


def upsample_prompt_local(
        clip,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        seed: int = 0,
        max_length: int = 400) -> Tuple[str, Optional[str]]:
    """
    upsample_prompt on the local text encoder: the same brief, the same checks, one revision.

    Args:
        clip:
            A ComfyUI CLIP holding Qwen3.5-4B.
        prompt (str):
            The user's prompt.
        height (int):
            Image height, for the frame.
        width (int):
            Image width.
        seed (int):
            Sampling seed.
        max_length (int):
            Most tokens to generate per attempt.

    Returns:
        Tuple[str, Optional[str]]:
            (caption, error), as parse_reply.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(prompt=prompt, height=height, width=width)},
    ]
    caption, error = _complete_local(clip, messages, seed, max_length)
    problems = [] if error else check_caption(caption=caption)
    if problems:
        messages += [
            {"role": "assistant", "content": f"<caption>{caption}</caption>"},
            {"role": "user", "content": "Revise the caption: " + "; ".join(problems) + ". Keep "
             "everything else. Reply with the caption inside <caption></caption> tags."},
        ]
        caption, error = _complete_local(clip, messages, seed + 1, max_length)
        problems = [] if error else check_caption(caption=caption)
        if problems:
            warnings.warn(f"upsampled caption still breaks a rule: {'; '.join(problems)}")
    return ("" if error else caption), error

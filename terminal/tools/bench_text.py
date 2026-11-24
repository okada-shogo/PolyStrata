"""The bench report's prompt text (strata-poly --bench-report --bench-text): plain English sentences made from fixed
word lists with a fixed seed, the same on every PC and every version of this file's VERSION.  No user data, nothing
from the PC.  A text that reads like prose routes a mixture-of-experts model's tokens to as many experts as a real
prompt does; random token ids route to far fewer, and their answers repeat themselves.

    python terminal/tools/bench_text.py 2000          the first 2,000 words, to look at
"""
from __future__ import annotations

import random
import sys

VERSION = 2   # change it with the lists or the generator: a report says which text it read

# Plain words, several hundred of them, so that a prompt has about as many different tokens as real prose (version 1's
# few phrases had 198 different tokens in 2,048; an English article about 670).
NOUNS = """account airport answer apple area argument army arrival article artist attention aunt autumn baby balance
ball band bank basket battery beach bean bear bed bell bicycle bird birthday blanket boat body bone book border bottle
bowl box brain branch bread breakfast brick bridge brother brush budget building bus butter cabin cake camera camp
candle captain car card carpet castle cat ceiling chair chapter cheese chest chicken child chimney church circle city
class clock cloud coast coat coffee coin collar college colour corner cottage country cousin cow crowd cup curtain
customer dance daughter desk diary dinner doctor dog door drawer dream dress driver drum dust eagle earth egg
election engine envelope evening factory family farm feather fence field film finger fire fish flag floor flower fog
forest fork fountain friend fruit garden gate gift glass glove goat government grandfather grass guitar hall hammer
harbour hat health heart hill history holiday honey horse hospital hotel house island jacket journey judge kettle key
kitchen knife ladder lake lamp language lawyer leaf lesson letter library lighthouse lion lock machine magazine map
market meadow meeting message metal minute mirror moon morning mountain mouse museum music needle neighbour
newspaper night notebook ocean office onion orange orchard painting palace paper parcel parent park passenger path
pencil pepper piano picture pillow planet plate pocket poem pond potato prison problem question rabbit radio railway
rain recipe river road roof room rope sailor salt sand school science season shadow sheep shelf shell ship shirt shoe
shop shoulder silver sister skirt sky snow soldier song soup spoon spring square stairs star station stone storm
story street student sugar summer sun supper surface sweater table teacher telephone tent theatre ticket tiger
tomato tongue tool tooth tower town toy tractor train tree truck tunnel umbrella uncle valley village violin voice
wagon wall wallet watch water weather wheel window wing winter wire wood wool world year""".split()
VERBS = """accepted admired allowed answered arranged asked attacked avoided baked balanced borrowed brought built
buried burned called carried caught changed chased checked cleaned climbed closed collected copied counted covered
crossed delivered described designed destroyed discovered divided dropped earned emptied entered escaped examined
explained fetched filled finished fixed folded followed forgot found gathered guarded guessed handed heard helped
hid hunted imagined improved invited joined kept kicked knocked learned left lifted listened locked lost loved made
marked measured mended met moved noticed obeyed offered opened ordered owned packed painted passed picked placed
planned planted played pointed polished poured praised prepared pressed printed promised protected pulled pushed
questioned reached read received recorded remembered repaired replaced rescued returned rolled ruled sailed saved
searched sent served shared showed signed sold sorted spilled spotted started stopped studied supported surprised
taught tested thanked threw tied touched traced traded trained translated trusted turned visited waited walked
warned washed watched welcomed whispered wrapped wrote""".split()
ADJECTIVES = """ancient angry anxious bitter black blue bold brave bright broad broken busy calm careful cheap cheerful
clean clever cloudy cold colourful comfortable common cool crowded curious damp dark deep delicate dirty distant dry
dusty eager early empty enormous famous fast fierce flat foolish foreign fresh friendly gentle giant glad golden
graceful green grey happy hard heavy helpful hidden high hollow honest huge hungry icy important kind large late lazy
light little lonely long loud lovely low lucky modern narrow nervous new noisy odd old orange patient plain pleasant
polite poor proud purple quick quiet rare red rich rough round rusty sad safe salty sharp shiny short shy silent
simple slow small smooth soft sour square steep sticky strange strong sudden sweet tall tame thick thin tidy tiny
tired twisted ugly unusual useful warm weak wet white wide wild wise wooden yellow young""".split()
ADVERBS = """afterwards almost always angrily badly bravely briefly calmly carefully certainly clearly closely
easily eventually finally firmly gently gladly gradually happily honestly immediately kindly lately loudly luckily
nearly neatly never often patiently politely proudly quickly quietly rarely really recently regularly roughly sadly
safely seldom slowly smoothly softly sometimes soon suddenly surely tightly today tomorrow warmly wisely yesterday""".split()
PREPS = "above across after against along among around behind below beside between beyond by during inside near over through under within".split()
LINKS = ("Later,", "Meanwhile,", "Afterwards,", "In the end,", "At first,", "However,", "For that reason,", "Then",
         "Some time later,", "On the other hand,", "Of course,", "Even so,", "Before long,", "Luckily,")


def _sentence(r: random.Random) -> str:
    n = lambda: r.choice(NOUNS)        # noqa: E731
    a = lambda: r.choice(ADJECTIVES)   # noqa: E731
    k = r.randrange(5)
    if k == 0:
        s = f"the {a()} {n()} {r.choice(VERBS)} a {n()} {r.choice(PREPS)} the {n()}"
    elif k == 1:
        s = f"my {n()} {r.choice(ADVERBS)} {r.choice(VERBS)} the {a()} {n()} and the {n()}"
    elif k == 2:
        s = f"{r.choice(PREPS)} the {a()} {n()}, a {n()} {r.choice(VERBS)} {r.choice(ADVERBS)}"
    elif k == 3:
        s = f"nobody {r.choice(VERBS)} the {n()} of the {a()} {n()} {r.choice(PREPS)} the {n()}"
    else:
        s = f"the {n()} was {a()}, so the {n()} {r.choice(VERBS)} it {r.choice(PREPS)} a {a()} {n()}"
    return s + r.choice((".", ".", ".", "!", "?") if k == 3 else (".", ".", ".", "."))


def words(n: int, seed: int = 1) -> str:
    """At least n words of the text for `seed`, in paragraphs."""
    r = random.Random(seed * 1000 + VERSION)
    out, count, in_par = [], 0, 0
    while count < n:
        s = _sentence(r)
        if in_par and r.random() < 0.3:
            s = f"{r.choice(LINKS)} {s}"
        s = s[0].upper() + s[1:]
        out.append(s)
        count += len(s.split())
        in_par += 1
        if in_par >= r.randint(4, 8):
            out.append("\n\n")
            in_par = 0
    return " ".join(out).replace(" \n\n ", "\n\n").strip()


def tokens(tokenizer, n_tokens: int, seed: int = 1) -> list[int]:
    """At least n_tokens ids of the text, by the model's own tokenizer (its encode())."""
    n_words = max(64, n_tokens * 3 // 4)
    while True:
        ids = tokenizer.encode(words(n_words, seed))
        if len(ids) >= n_tokens:
            return ids
        n_words = n_words * 3 // 2 + 64


if __name__ == "__main__":
    print(words(int(sys.argv[1]) if len(sys.argv) > 1 else 300))

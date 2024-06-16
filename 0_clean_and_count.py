#!/opt/homebrew/Caskroom/miniconda/base/envs/attention/bin/python

import re
import os
import matplotlib.pyplot as plt

print("==> walking files.")

# Script is in the root dir
IN_DIR = os.getcwd() + "/dirty-data/ende/"
OUT_DIR = os.getcwd() + "/clean-data/ende/"

file_tree = {}

_, dirty_test, dirty_train, dirty_val = os.walk(IN_DIR)

# dirty_val, e.g., looks like this:
# ('/Users/henryjosephson/personal/Projects/Attention-Is-All-I-Need/dirty-data/ende/val',
# [],
# ['newstest2013.en', 'newstest2013.de'])

for directory in (dirty_test, dirty_train, dirty_val):
    file_tree[directory[0].split("/")[-1]] = directory[2]

# file_tree looks like this:
# {
#   'test': [
#       'newstest2014-deen-src.en.sgm',
#       'newstest2014-deen-ref.de.sgm',
#       'newstest2014-deen-ref.en.sgm',
#       'newstest2014-deen-src.de.sgm',
#   ],
# 'train': [
#       'commoncrawl.de-en.en',
#       'commoncrawl.de-en.de',
#       'news-commentary-v9.de-en.en',
#       'europarl-v7.de-en.en',
#       'commoncrawl.de-en.annotation',
#       'europarl-v7.de-en.de',
#       'news-commentary-v9.de-en.de',
#   ],
# 'val': [
#       'newstest2013.en',
#       'newstest2013.de',
#   ]
# }

print("==> initializing cleaner.")

remap_dict = {
    "„ ": '"',  # fix non-aligned beginnings
    " “": '"',  # fix non-aligned beginnings
    "\u0093": '"',
    "\u0094": '"',
    "\u0097": " ",
    " “": '"',  # fix non-aligned beginnings
    "\u00a0": " ",  # non-breaking white space
    "\u202f": " ",  # narrow non-breaking white space
    "Ã¶": "ö",  # german oe
    "Ã¼": "ü",  # german ue
    "Ã¤": "ä",  # german ae
    "„": '"',
    "“": '"',
    "‟": '"',
    "”": '"',
    "″": '"',
    "‶": '"',
    "”": '"',
    "‹": '"',
    "›": '"',
    "’": "'",
    "′": "'",
    "′": "'",
    "‛": "'",
    "‘": "'",
    "`": "'",
    "–": "--",
    "‐": "-",
    "»": '"',
    "«": '"',
    "≪": '"',
    "≫": '"',
    "》": '"',
    "《": '"',
    "？": "?",
    "！": "!",
    "…": " ... ",
    "\t": " ",
    "。": ".",  # chinese period
    "︰": ":",
    "〜": "~",
    "；": ";",
    "）": ")",
    "（": "(",
    "ﬂ": "fl",  # small ligature characters
    "ﬁ": "fi",
    "¶": " ",
}

filter_unicode_ranges = (
    "\u4e00-\u9fff",  # chinese
    "\u3040-\u309f",  # japanese (Hiragana)
    "\u30a0-\u30ff",  # japanese2 (Hiragana)
    "\u0400-\u04ff",  # cyrillic
    "\u0900-\u0954",  # devanagari, but hindi is here
    "\uac00-\ud7a3",  # korean1
    "\u1100-\u11ff",  # korean2
    "\u3130-\u318f",  # korean3
    "\ua960-\ua97f",  # korean4
    "\ud7b0-\ud7ff",  # korean5
    "\u0d00-\u0d7f",  # malayalam
    "\u0600-\u06ff",  # arabic1
    "\u0750-\u077f",  # arabic2
    "\u0870-\u089f",  # arabic3
    "\u08a0-\u08ff",  # arabic4
    "\ufb50-\ufdff",  # arabic5
    "\ufe70-\ufeff",  # arabic6
    "\u0590-\u05ff",  # hebrew
    "\u1200-\u137f",  # ethiopic
    "\u4e00-\u4fff",  # chinese1
    "\u5000-\u57ff",  # chinese2
    "\u5800-\u5fff",  # chinese3
    "\u6000-\u67ff",  # chinese4
    "\u6800-\u6fff",  # chinese5
    "\u7000-\u77ff",  # chinese6
    "\u7800-\u7fff",  # chinese7
    "\u8000-\u87ff",  # chinese8
    "\u8800-\u8fff",  # chinese9
    "\u9000-\u97ff",  # chinese10
    "\u9800-\u9fff",  # chinese11
    "\u3100-\u312f",  # chinese12
    "\u31a0-\u31bf",  # chinese13
    "\u2c00-\u2c5f",  # glagolitic1
    "\u0980-\u09ff",  # bengali
    "\u0c00-\u0c7f",  # telugu
    "\U00010e60-\U00010e7e",  # rumi
    "\U00010ec0-\U00010eff",  # arabic7
    "\U0001ec70-\U0001ecbf",  # indic
    "\U0001ed00-\U0001ed4f",  # ottoman
    "\U0001ee00-\U0001eeff",  # arabic-m
    "\U0001e000-\U0001e02f",  # glagolitic2
)


def normalize_text(text: str, remove_unicode=True) -> str:
    """
    adapted from Martin Dittgen's function in
    https://github.com/Montinger/Transformer-Workbench/blob/main/transformer-
    from-scratch/common.py.
    ===
    takes str input and replaces special characters with

    Optionally replaces unicode characters with '[UNK]'.
    """
    for old, new in remap_dict.items():
        text = text.replace(old, new)

    if remove_unicode:
        for unicode_range in filter_unicode_ranges:
            text = re.sub(rf"[{unicode_range}]+", "[UNK]", text, flags=re.U)

    while text.find("  ") >= 0:
        text = text.replace("  ", " ").replace("  ", " ")

    return text


print("==> cleaning files.")
unique_chars = set()

# this loop cleans each file, writes the clean file to the clean-data dir, and
# indexes unique characters.
for file_type, path_tails in file_tree.items():
    for path_tail in path_tails:
        print("\tprocessing " + path_tail)

        with open(
            IN_DIR + file_type + "/" + path_tail, "r", encoding="utf-8"
        ) as f_in:
            full_text = f_in.read()
            if file_type == "test":
                full_text = normalize_text(full_text, remove_unicode=False)
                # the test files have weird html-esque tags that the training
                # and validation files lack. The block below deals with this.

                full_text = re.sub(r"<[^>]*?>", "", full_text)
                # this regex looks for '<', followed by anything that isn't '>'
                # zero or more times, and then followed by '>'. The ? means
                # that the search isn't greedy, i.e. it looks for the '>'
                # nearest to the opening '<' instead of the '>' closest to the
                # end of the line.
                while full_text.find("\n\n") >= 0:
                    full_text = full_text.replace("\n\n", "\n")

            else:
                full_text = normalize_text(full_text)

            with open(
                OUT_DIR + file_type + "/" + path_tail, "w", encoding="utf-8"
            ) as f_out:
                f_out.write(full_text)

            unique_chars.update(set(full_text))

print("==> getting char counts")
char_count_dict = {key: 0 for key in unique_chars}

for file_type, path_tails in file_tree.items():
    for path_tail in path_tails:
        print("\tcounting chars in " + path_tail)

        with open(
            OUT_DIR + file_type + "/" + path_tail, "r", encoding="utf-8"
        ) as f:
            full_text = f.read()
            for char in unique_chars:
                char_count_dict[char] += full_text.count(char)

plot_var = list(
    zip(
        *sorted(
            list(char_count_dict.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:20]
    )
)

print("==> plotting")
fig, ax = plt.subplots()
ax.bar(plot_var[0], height=plot_var[1], color="g")
ax.set(
    title="Character Counts",
    xlabel="Character",
    ylabel="Count",
)

plt.savefig("./imgs/char_count.png")
plt.close()
print("==> done")

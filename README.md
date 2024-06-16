# Attention Is All I Need
This is my attempt to replicate the classic 2017 paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), which introduced the transformer architecture.

Why? Because it's fun, because I want to understand how transformers actually work, and because I can't do anything for my internship until after a meeting later this afternoon.

I'm far from the first person to do this — see, e.g., previous replications by [Martin Dittgen](https://medium.com/@martin.p.dittgen/reproducing-the-attention-is-all-you-need-paper-from-scratch-d2fb40bb25d4), person2, and person3, all of which I cross-referenced when doing my own replication.

## 0. Prep/What am I doing?
The original paper references base models and "big" models, with the base model training in around 12 hours and the big models training in around 3.5 days. (In both cases, they trained one machine with 8 NVIDIA P100 GPUs.)[^1]

I'm lucky enough to have access to UChicago's remote computing cluster, so I'll be able to use 4 A100s, which should breeze through anything I throw at them.

I'll definitely replicate the base model, and I'll decide whether to do the big model after I'm done with base.

## 1. Training data
### Acquiring
The original *Attention* paper was part of the 2014 Workshop on Machine Translation, so we can download all the training, evaluation, and testing data from the [WMT's website](https://www.statmt.org/wmt14/translation-task.html). For training, I downloaded every one of their datasets that had a DE-EN set (there's a convenient table around $\frac{2}{3}$ of the way down the page):
- *[Europarl v7](https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz)*,
- *[Common Crawl corpus](https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz)*, and
- *[News Commentary](https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz)*.

The site recommends using the 2013 test set for validation and the 2014 test set for scoring, so that's what I'll do. Perhaps more importantly, this is also what the original *Attention* paper did.[^2] I downloaded the 2013 test set (i.e. my evaluation set) as part of this larger [dev](https://www.statmt.org/wmt14/dev.tgz) tar file. For testing, I used the [filtered test sets](https://www.statmt.org/wmt14/test-filtered.tgz), since that's what the site recommends for anyone who "want\[s\] to reproduce results from the campaign."

I structured the files as follows: 
```
.
├── clean-data
│   └── ende
│       ├── test
|       ├── train
│       └── val
└── dirty-data
    └── ende
        ├── test
        ├── train
        └── val
```

All the scripts I write will live in the root directory. The extra `ende` directory is so that I don't have to massively reorganize if I want to train on other languages beyond German.  All the data I've downloaded here is going into `dirty-data`, of course. My cleaning script will write the cleaned version of each file into the corresponding `clean-data` subdirectory. Speaking of cleaning...

### Cleaning
After downloading all this text, I had to make sure is was all actually workable. A big part of this is removing the non-ASCII German characters, like ö, ä, ü, and ß. Germans also do quotes „like this“, which is another thing I'll have to convert. 

The WMT site that provided all the original data also provides a perl script for normalizing the punctuation, but I adapted [some python](https://github.com/Montinger/Transformer-Workbench/blob/main/transformer-from-scratch/0-Cleans-Data-and-Tokenize.py) that does the same thing from Martin Dittgen's replication of the paper.[^3]

You can read the script I used for this part at [0_clean_and_count.py](./0_clean_and_count.py)...
- <details><summary>Or you can here to un-collapse the same script, broken down into chunks.</summary>

    - <details><summary>This part initializes all the filepaths.</summary>

        ```python
        #!/opt/homebrew/Caskroom/miniconda/base/envs/attention/bin/python

        import re
        import os
        import itertools

        # Script is in the root dir
        IN_DIR = os.getcwd() + '/dirty-data/ende/'
        OUT_DIR = os.getcwd() + '/clean-data/ende/'

        file_tree = {}

        _, dirty_test, dirty_train, dirty_val = os.walk(IN_DIR)

        # dirty_val, e.g., looks like this:
        # ('/Users/henryjosephson/personal/Projects/Attention-Is-All-I-Need/dirty-data/ende/val',
        # [],
        # ['newstest2013.en', 'newstest2013.de'])

        for dir in (dirty_test, dirty_train, dirty_val):
            file_tree[dir[0].split("/")[-1]] = dir[2]

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
        ```
    </details>

    - <details><summary>This part contains dictionaries of stuff to replace and what to replace it with.</summary>
    
        ```python
        remap_dict = {
                '„ ' : '"', # fix non-aligned beginnings
                ' “' : '"', # fix non-aligned beginnings
                '\u0093' : '"',
                '\u0094' : '"',
                '\u0097' : ' ',
                ' “' : '"', # fix non-aligned beginnings
                '\u00a0' : ' ', # non-breaking white space
                '\u202f' : ' ', # narrow non-breaking white space
                'Ã¶' : 'ö', # german oe
                'Ã¼' : 'ü', # german ue
                'Ã¤' : 'ä', # german ae
                '„'  : '"',
                '“'  : '"',
                '‟'  : '"',
                '”'  : '"',
                '″'  : '"',
                '‶'  : '"',
                '”'  : '"',
                '‹'  : '"',
                '›'  : '"',
                '’'  : "'",
                '′'  : "'",
                '′'  : "'",
                '‛'  : "'",
                '‘'  : "'",
                '`'  : "'",
                '–'  : '--',
                '‐'  : '-',
                '»'  : '"',
                '«'  : '"',
                '≪'  : '"',
                '≫'  : '"',
                '》' : '"',
                '《' : '"',
                '？' : '?',
                '！' : '!',
                '…'  : ' ... ',
                '\t' : ' ',
                '。' : '.', # chinese period
                '︰' : ':',
                '〜' : '~',
                '；' : ';',
                '）' : ')',
                '（' : '(',
                'ﬂ'  : 'fl', # small ligature characters
                'ﬁ'  : 'fi',
                '¶'  : ' ',
            }

        filter_unicode_ranges = (
                "\u4e00-\u9fff", # chinese
                "\u3040-\u309f", # japanese (Hiragana)
                "\u30a0-\u30ff", # japanese2 (Hiragana)
                "\u0400-\u04ff", # cyrillic
                "\u0900-\u0954", # devanagari, but hindi is here
                "\uac00-\ud7a3", # korean1
                "\u1100-\u11ff", # korean2
                "\u3130-\u318f", # korean3
                "\ua960-\ua97f", # korean4
                "\ud7b0-\ud7ff", # korean5
                "\u0d00-\u0d7f", # malayalam
                "\u0600-\u06ff", # arabic1
                "\u0750-\u077f", # arabic2
                "\u0870-\u089f", # arabic3
                "\u08a0-\u08ff", # arabic4
                "\ufb50-\ufdff", # arabic5
                "\ufe70-\ufeff", # arabic6
                "\u0590-\u05ff", # hebrew
                "\u1200-\u137f", # ethiopic
                "\u4e00-\u4fff", # chinese1
                "\u5000-\u57ff", # chinese2
                "\u5800-\u5fff", # chinese3
                "\u6000-\u67ff", # chinese4
                "\u6800-\u6fff", # chinese5
                "\u7000-\u77ff", # chinese6
                "\u7800-\u7fff", # chinese7
                "\u8000-\u87ff", # chinese8
                "\u8800-\u8fff", # chinese9
                "\u9000-\u97ff", # chinese10
                "\u9800-\u9fff", # chinese11
                "\u3100-\u312f", # chinese12
                "\u31a0-\u31bf", # chinese13
                "\u2c00-\u2c5f", # glagolitic1
                "\u0980-\u09ff", # bengali
                "\u0c00-\u0c7f", # telugu
                "\U00010e60-\U00010e7e", # rumi
                "\U00010ec0-\U00010eff", # arabic7
                "\U0001ec70-\U0001ecbf", # indic
                "\U0001ed00-\U0001ed4f", # ottoman
                "\U0001ee00-\U0001eeff", # arabic-m
                "\U0001e000-\U0001e02f", # glagolitic2
        )
        ```
    </details>

    - <details><summary>This part actually cleans the files.</summary>

        ```py
        print("==> cleaning files.")
        unique_chars = set()

        # this loop cleans each file, writes the clean file to the clean-data dir, and
        # indexes unique characters.
        for file_type, path_tails in file_tree.items():
            for path_tail in path_tails:
                print("\tprocessing " + path_tail)

                with open(IN_DIR + file_type + "/" + path_tail, 'r') as f_in:
                    full_text = f_in.read()
                    if file_type == 'test':
                        full_text = normalize_text(full_text, remove_unicode=False)
                        # the test files have weird html-esque tags that the training
                        # and validation files lack. The block below deals with this.
                        
                        full_text = re.sub(r'<[^>]*?>', '', full_text)
                        # this regex looks for '<', followed by anything that isn't '>'
                        # zero or more times, and then followed by '>'. The ? means that
                        # the search isn't greedy, i.e. it looks for the '>' nearest to
                        # the opening '<' instead of the '>' closest to the end of the
                        # line.
                        while full_text.find('\n\n') >= 0:
                            full_text = full_text.replace('\n\n', '\n')

                    else :
                        full_text = normalize_text(full_text)

                    with open(
                        OUT_DIR + file_type + "/" + path_tail, 'w', encoding='utf-8'
                    ) as f_out:
                        f_out.write(full_text)

                    unique_chars.update(set(full_text))
        ```
    </details>

    - <details><summary>This part gets and graphs character counts.</summary>

        ```py
        print("==> getting char counts")
        char_count_dict = {key: 0 for key in unique_chars}

        for file_type, path_tails in file_tree.items():
            for path_tail in path_tails:
                print("\tcounting chars in " + path_tail)

                with open(OUT_DIR + file_type + "/" + path_tail, 'r') as f:
                    full_text = f.read()
                    for char in unique_chars:
                        char_count_dict[char] += full_text.count(char)

        plot_var = [x for x in zip(*sorted(
            list(char_count_dict.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:20])]

        fig,ax = plt.subplots()
        ax.bar(
            plot_var[0],
            height = plot_var[1],
            color='g'
        )
        ax.set(
            title='Character Counts',
            xlabel='Character',
            ylabel='Count',
        )

        plt.savefig('./imgs/char_count.png')
        ```
    </details>

    </details>


Here are the top twenty characters, along with their counts:
![](/imgs/char_count.png)

The most frequently-appearing character is the space. It doesn't quite match the $\dfrac{1}{n}$ distribution we'd expect from [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law), but it's kinda close.

### Tokenizing

Of course, the cold, hard machine gods don't actually speak English. They don't speak German, either — they speak number. So let's convert our natural language data into 

<details>
<summary>Click here to expand the code I used</summary>

```ruby
def some_code
    puts "Rails is so cool"
end
```

</details>



[^1]: This info is all from section 5.2 of the *Attention* paper.

[^2]: See *Attention* section 6.2.

[^3]: Normally I'd feel a little bad about skipping a step like this, but I already know how to do find-and-replace — the point of this whole thing is to better understand how transformers work, not how to replace *ö*s with *o*s. This is just to save time.
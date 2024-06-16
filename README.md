# Attention Is All I Need
This is my attempt to replicate the classic 2017 paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), which introduced the transformer architecture.

Why? Because it's fun, because I want to understand how transformers actually work, and because 

I'm far from the first person to do this — see, e.g., previous replications by [Martin Dittgen](https://medium.com/@martin.p.dittgen/reproducing-the-attention-is-all-you-need-paper-from-scratch-d2fb40bb25d4), person2, and person3, all of which I cross-referenced when doing my own replication.

## 0. Prep/What am I doing?
The original paper references base models and "big" models, with the base model training in around 12 hours and the big models training in around 3.5 days. (In both cases, they trained one machine with 8 NVIDIA P100 GPUs.)[^1]

In the best-case scenario, I'll be able to use 4 A100s from UChicago's compute cluster, which should breeze through anything I throw at them. In the worst-case, I'll be working with the M2 chip on my mac (which isn't the worst, but is a far cry).

I'll definitely replicate the base model, and whether I try to reproduce the big model will depend on whether I get compute time.

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

The WMT site that provided all the original data also provides a perl script for normalizing the punctuation, but I stole [some python](https://github.com/Montinger/Transformer-Workbench/blob/main/transformer-from-scratch/0-Cleans-Data-and-Tokenize.py) that does the same thing from Martin Dittgen's replication of the paper.[^3]

<details>
<summary>Click here for the normalization script</a></summary>

```python
#!/opt/homebrew/Caskroom/miniconda/base/envs/attention/bin/python

import re
import os
import itertools

# Script is in the root dir
IN_DIR = os.getcwd() + '/dirty-data/ende/'
OUT_DIR = os.getcwd() + '/clean-data/ende/'

dirty_files = {}

_, dirty_test, dirty_train, dirty_val = os.walk(IN_DIR)

# dirty_val, e.g., looks like this:
# ('/Users/henryjosephson/personal/Projects/Attention-Is-All-I-Need/dirty-data/ende/val',
# [],
# ['newstest2013.en', 'newstest2013.de'])

for dir in (dirty_test, dirty_train, dirty_val):
    dirty_files[dir[0].split("/")[-1]] = dir[2]

# dirty_files looks like this:
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

<details>
<summary>Click here to expand the dictionary of characters to replace. It's hidden by default because it's long (~50 lines) and tedious.</a></summary>
Like I mentioned above, I took this from Martin Dittgen.

```py
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
```



test
</details>


</details>

### Tokenizing


 

<details>
<summary>Click here to expand the code I used</a></summary>

```ruby
def some_code
    puts "Rails is so cool"
end
```

</details>



[^1]: This info is all from section 5.2 of the *Attention* paper.

[^2]: See *Attention* section 6.2.

[^3]: Normally I'd feel a little bad about skipping a step like this, but I already know how to do find-and-replace — the point of this whole thing is to better understand how transformers work, not how to replace *ö*s with *o*s. This is just to save time.
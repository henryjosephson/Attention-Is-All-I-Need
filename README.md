# Attention Is All I Need
This is my attempt to replicate the classic 2017 paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), which introduced the transformer architecture.

Why? Because it's fun, because I want to understand how transformers actually work, and because 

I'm far from the first person to do this — see, e.g., previous replications by [Martin Dittgen](https://medium.com/@martin.p.dittgen/reproducing-the-attention-is-all-you-need-paper-from-scratch-d2fb40bb25d4), person2, and person3, all of which I cross-referenced when doing my own replication.

## 0. Prep/What am I doing?
The original paper references base models and "big" models, with the base model training in around 12 hours and the big models training in around 3.5 days. (In both cases, they trained one machine with 8 NVIDIA P100 GPUs.)[^1]

In the best-case scenario, I'll be able to use 4 A100s from UChicago's compute cluster, which should breeze through anything I throw at them. In the worst-case, I'll be working with the M2 chip on my mac (which isn't the worst, but is a far cry).

I'll definitely replicate the base model, and whether I try to reproduce the big model will depend on whether I get compute time.

## 1. Training data
All the models are evaluated based on the standard Workshop on Machine Translation 2014 English-German dataset, so we'll work with that. Fortunately, the [WMT's website](https://www.statmt.org/wmt14/translation-task.html) labels which datasets correspond to which languages, so I downloaded everything that had a DE-EN set:
- *[Europarl v7](https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz)*,
- *[Common Crawl corpus](https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz)*, and
- *[News Commentary](https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz)*.

They also conveniently include their test sets, so I downloaded those, too.

[^1]: This info is all from section 5.2 of the *Attention* paper.
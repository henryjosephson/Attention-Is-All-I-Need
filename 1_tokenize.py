#!/opt/homebrew/Caskroom/miniconda/base/envs/attention/bin/python

import os
import tokenizers

print("==> walking files.")

CLEAN_DIR = os.getcwd() + "/clean-data/ende/"

file_tree = {}

_, clean_test, clean_train, clean_val = os.walk(CLEAN_DIR)

for directory in (clean_test, clean_train, clean_val):
    file_tree[directory[0].split("/")[-1]] = directory[2]

print("==> checking if pre-trained tokenizer in root dir.")

if not 
    print("==> training tokenizer.")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))

    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"], # START and END are for the target sentence only
        vocab_size=25_000,
        min_frequency=3,
        show_progress=True
    )

    tokenizer.normalizer = tokenizers.normalizers.Sequence([tokenizers.normalizers.NFKC()]) # , tokenizers.normalizers.StripAccents()]) -> destorys German Umlaute
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
    tokenizer.decoder = tokenizers.decoders.Metaspace()

    LIST_OF_LISTS_OF_NONTEST_PATHS = [[CLEAN_DIR + x[0] + "/" + y for y in x[1]] for x in file_tree.items()][1:]
    LIST_OF_NONTEST_PATHS = LIST_OF_LISTS_OF_NONTEST_PATHS[0] + LIST_OF_LISTS_OF_NONTEST_PATHS[1]

    del LIST_OF_LISTS_OF_NONTEST_PATHS


    tokenizer.train(LIST_OF_NONTEST_PATHS, trainer)

    # save tokenizer
    tokenizer.save(str(os.getcwd() + '/tokenizer.json'))

    print(tokenizer.get_vocab_size())

print("==> tokenizing data.")


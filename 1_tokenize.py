#!/opt/homebrew/Caskroom/miniconda/base/envs/attention/bin/python

import os
import gc
import tokenizers # type: ignore

print("==> walking files.")

CLEAN_DIR = os.getcwd() + "/clean-data/ende/"

file_tree = {}

_, clean_test, clean_train, clean_val = os.walk(CLEAN_DIR)

for directory in (clean_test, clean_train, clean_val):
    file_tree[directory[0].split("/")[-1]] = directory[2]

print("==> checking if pre-trained tokenizer in root dir.")

if not 'tokenizer.json' in os.listdir():
    print("==> training tokenizer.")

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))

    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"],
        vocab_size=25_000,
        min_frequency=3,
        show_progress=True
    )

    tokenizer.normalizer = tokenizers.normalizers.Sequence(
        [tokenizers.normalizers.NFKC()] #,
        ) #, tokenizers.normalizers.StripAccents()]) # destorys German Umlaute
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
    tokenizer.decoder = tokenizers.decoders.Metaspace()

    LISTS_OF_PATHS = [
        [CLEAN_DIR + x[0] + "/" + y for y in x[1]] for x in file_tree.items()
    ]

    tokenizer.train(LISTS_OF_PATHS[1] + LISTS_OF_PATHS[2], trainer)
    # i.e. not training on test data

    print("==> saving tokenizer.")
    tokenizer.save(os.getcwd() + '/tokenizer.json')

    print(tokenizer.get_vocab_size())
else:
    print("==> using pre-trained tokenizer.")
    tokenizer = tokenizers.Tokenizer.from_file(os.getcwd() + '/tokenizer.json')

print("==> tokenizing data.")
all_files = LISTS_OF_PATHS[0] + LISTS_OF_PATHS[1] + LISTS_OF_PATHS[2]

de_files = sorted([x for x in all_files if ".de" in x])
en_files = sorted([x for x in all_files if ".en" in x])

gc.collect() 
# wouldn't have thought to do this if not for Martin doing it in his replication.
# thanks, martin.

max_token_length = 0 
# we're just initializing this -- we'll set it to the length of the longest
# test item.

# this part is taken pretty much line-for-line from martin's code.
tokenized_dict = {
    'train':     {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
    'val':       {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
    'test-src':  {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
    'test-ref':  {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
}


for idx, (de_file, en_file) in enumerate(zip(de_files, en_files)):
    print(f"Processing step {idx}")
    print(f"\tde_file: {de_file}")
    print(f"\ten_file: {en_file}")
    
    file_name = de_file.split('/')[-1]
    print(f"\tfile_type: {file_name}")
    
    if "newstest2013" in file_name:
        save_type = "val"
    elif "newstest2014" in file_name and '-ref' in file_name:
        save_type = "test-ref"
    elif "newstest2014" in file_name and '-src' in file_name:
        save_type = "test-src"
    else:
        save_type = "train"
    
    print(f"\tsave_type: {save_type}")
    
    with open(
        de_file, 'r', encoding='utf-8'
    ) as de_ff, open(
        en_file, 'r', encoding='utf-8'
    ) as en_ff:
        for de_line, en_line in zip(de_ff, en_ff):
            de_line = de_line.strip()
            en_line = en_line.strip()
            if len(de_line)==0 or len(en_line)==0:
                continue
            # add START and END token to de_string (as it is target)
            de_line = ('[START]' +  de_line + '[END]')
            en_line = ('[START]' +  en_line + '[END]')

            de_token_obj = tokenizer.encode(de_line)
            en_token_obj = tokenizer.encode(en_line)

            de_token_ids = de_token_obj.ids[:max_token_length]
            en_token_ids = en_token_obj.ids[:max_token_length]

            if "test" in save_type:
                candidate_max = len(de_token_obj.ids)
                if candidate_max > max_token_length:
                    max_token_length = candidate_max
                    print(f"Updating max_token_length to {max_token_length}")

            tokenized_dict[save_type]['de'].append( de_token_ids )
            tokenized_dict[save_type]['en'].append( en_token_ids )
            tokenized_dict[save_type]['de_length'].append( len(de_token_ids) )
            tokenized_dict[save_type]['en_length'].append( len(en_token_ids) )
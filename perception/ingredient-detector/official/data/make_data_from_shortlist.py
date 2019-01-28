import pickle
import json
import os
from shutil import copy2
import numpy as np

VOCAB_PATH = 'ingrs_short_list.txt'
SRC_IMG_PKL_PATH = 'salad1_id_to_img.pkl'
TRAIN_PATH = 'TRAIN2'
VAL_PATH = 'VALIDATION2'
if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)
if not os.path.exists(VAL_PATH):
    os.makedirs(VAL_PATH)

IMG_PATH_PREFIX = 'train'
cpy_imgs = True
ignore_blank_examples = True

val_data_pct = 0.2

det_ingrs_path = 'det_ingrs.json'
salad_id_path = 'salad1.txt'
salad_ids = [line.strip() for line in open(salad_id_path, 'r')]
assert len(salad_ids) == 24666

vocab_map = [line.strip().split(':') for line in open(VOCAB_PATH, 'r')]
vocab_map = {k: v.split(',') for [k, v] in vocab_map}
vocab = set([word for phrase_map in vocab_map.values() for word in phrase_map])
# TODO(sai): pickle this
vocab = list(vocab)
VOCAB_PKL_PATH = 'vocab_short_list.pkl'
pickle.dump(vocab, open(VOCAB_PKL_PATH, 'wb'))
assert len(vocab) == 87
vocab_to_idx = {word: i for i, word in enumerate(vocab)}
phrase_to_idx = {k: [vocab_to_idx[word] for word in v] for (k, v) in vocab_map.items()}

id_to_img_paths = pickle.load(open(SRC_IMG_PKL_PATH, 'rb'))
f = open(det_ingrs_path, 'r')
parsed_json = json.load(f)
for recipe in parsed_json:
    Id = recipe['id']
    if (Id in salad_ids) and (Id in id_to_img_paths):
        print(Id)
        ingrs = []
        ingr_names = []
        for ingr in recipe['ingredients']:
            ingr = ingr['text']
            if ingr in phrase_to_idx:
                ingrs.extend(phrase_to_idx[ingr])
        ingrs = list(set(ingrs))
        if ignore_blank_examples and len(ingrs) == 0:
            continue
        ingrs = np.array(sorted(ingrs), dtype=np.int64)
        ingr_names = [vocab[i] for i in ingrs]
        img_paths = id_to_img_paths[Id]
        for img_path in img_paths:
            img_base = os.path.basename(img_path)
            img_id = os.path.splitext(img_base)[0]
            if cpy_imgs:
                DST_PREFIX = TRAIN_PATH
                if np.random.random() <= val_data_pct:
                    DST_PREFIX = VAL_PATH
                pickle.dump([ingrs, ingr_names],
                        open(os.path.join(DST_PREFIX, img_id+'.pkl'), 'wb'))
                src_img_path = os.path.join(img_path)
                dst_img_path = os.path.join(DST_PREFIX, img_base)
                copy2(src_img_path, dst_img_path)
            else:
                DST_PREFIX = TRAIN_PATH
                if not os.path.exists(os.path.join(DST_PREFIX, img_id+'.pkl')):
                    DST_PREFIX = VAL_PATH
                pickle.dump([ingrs, ingr_names],
                        open(os.path.join(DST_PREFIX, img_id+'.pkl'), 'wb'))


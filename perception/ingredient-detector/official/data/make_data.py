import pickle
import json
import os
from shutil import copy2
import numpy as np

PKL_PATH = 'salad1_vocab_cutoff.pkl'
SRC_IMG_PKL_PATH = 'salad1_id_to_img.pkl'
TRAIN_PATH = 'TRAIN'
VAL_PATH = 'VALIDATION'
IMG_PATH_PREFIX = 'train'
cpy_imgs = False

val_data_pct = 0.1

det_ingrs_path = 'det_ingrs.json'
salad_id_path = 'salad1.txt'
salad_ids = [line.strip() for line in open(salad_id_path, 'r')]
assert len(salad_ids) == 24666

vocab, _ = pickle.load(open(PKL_PATH, 'rb'))
N = len(vocab)
assert N == 1000
vocab_to_idx = {word: i for i, word in enumerate(vocab)}

id_to_img_paths = pickle.load(open(SRC_IMG_PKL_PATH, 'rb'))
f = open(det_ingrs_path, 'r')
parsed_json = json.load(f)
for recipe in parsed_json:
    Id = recipe['id']
    if (Id in salad_ids) and (Id in id_to_img_paths):
        print(Id)
        ingrs = []
        for ingr in recipe['ingredients']:
            ingr = ingr['text']
            if ingr in vocab_to_idx:
                ingrs.append(vocab_to_idx[ingr])
        ingrs = np.array(sorted(ingrs), dtype=np.int64)
        img_paths = id_to_img_paths[Id]
        for img_path in img_paths:
            img_base = os.path.basename(img_path)
            img_id = os.path.splitext(img_base)[0]
            if cpy_imgs:
                DST_PREFIX = TRAIN_PATH
                if np.random.random() <= val_data_pct:
                    DST_PREFIX = VAL_PATH
                pickle.dump(ingrs, open(os.path.join(DST_PREFIX, img_id+'.pkl'), 'wb'))
                src_img_path = os.path.join(img_path)
                dst_img_path = os.path.join(DST_PREFIX, img_base)
                copy2(src_img_path, dst_img_path)
            else:
                DST_PREFIX = TRAIN_PATH
                if not os.path.exists(os.path.join(DST_PREFIX, img_id+'.pkl')):
                    DST_PREFIX = VAL_PATH
                pickle.dump(ingrs, open(os.path.join(DST_PREFIX, img_id+'.pkl'), 'wb'))


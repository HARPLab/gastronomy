import json
import pickle
import numpy as np

"""
check_ingredients
23112
30.57338179300796 (OBSELETE - SEE IMG COUNT INSTEAD)
322.89118474104436

else
10719
38.460210840563484
300.01147640764776
"""

check_ingredients = False

det_ingrs_path = 'det_ingrs.json'
salad_id_path = 'salad1.txt'
id_to_img_pkl_path = 'salad1_id_to_img.pkl'
id_to_img_paths = pickle.load(open(id_to_img_pkl_path, 'rb'))
out_path = 'salad1_vocab.pkl'
if check_ingredients:
    salad_id_path = 'salad2.txt'
    out_path = 'salad2_vocab.pkl'

ingrs = set([])
salad_ids = [line.strip() for line in open(salad_id_path, 'r')]
salad_ids = set(salad_ids)
ingr_counts = {}

f = open(det_ingrs_path, 'r')

parsed_json = json.load(f)
for recipe in parsed_json:
    if (recipe['id'] in salad_ids) and (recipe['id'] in id_to_img_paths):
        _ingrs = []
        for ingr in recipe['ingredients']:
            ingr = ingr['text']
            _ingrs.append(ingr)
            if ingr not in ingr_counts:
                ingr_counts[ingr] = len(id_to_img_paths[recipe['id']])
            else:
                ingr_counts[ingr] += len(id_to_img_paths[recipe['id']])
        ingrs |= set(_ingrs)

f.close()
pickle.dump([ingrs, ingr_counts], open(out_path, 'wb'))
print(len(ingrs))
print(np.mean(ingr_counts.values()))
print(np.std(ingr_counts.values()))

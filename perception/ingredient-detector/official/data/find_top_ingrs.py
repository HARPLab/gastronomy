import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import operator
import bisect

cut_off = 1
in_path = 'salad1_vocab.pkl'
out_path = 'salad1_vocab_cutoff.pkl'
ingrs, ingr_counts = pickle.load(open(in_path, 'rb'))
id_to_img_pkl_path = 'salad1_id_to_img.pkl'
id_to_img_paths = pickle.load(open(id_to_img_pkl_path, 'rb'))

IMG_COUNT = np.sum(map(lambda x: len(x), id_to_img_paths.values()))
ingr_f = {k: 100.*float(v)/IMG_COUNT for k, v in ingr_counts.items()}
ingr_f_sorted = sorted(ingr_f.items(), key=operator.itemgetter(1))
ingr_f_fs = [v[1] for v in ingr_f_sorted]#[::-1]
ingr_f_ingrs = [v[0] for v in ingr_f_sorted]

pickle.dump([ingr_f_ingrs[-300:], 1], open(out_path, 'wb'))

#plt.bar(range(len(ingr_f_only)), ingr_f_only)
#plt.savefig('ingr_f_only.png')
#plt.close()

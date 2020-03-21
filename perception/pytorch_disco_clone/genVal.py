import os
import random
import glob
TFR_MOD = "aa"
# out_dir_base = "/projects/katefgroup/datasets/clevr_veggies/tfrs_tf2"
out_dir_base = "/projects/katefgroup/datasets/clevr_veggies/tfrs_tf2/"
folder = "{}/{}/*".format(out_dir_base,TFR_MOD)
files = glob.glob(folder)

random.shuffle(files)
print(len(files))
split = int(len(files)*0.8)
with open(out_dir_base + f'/{TFR_MOD}t.txt', 'w') as f:
	for item in files[:split]:
		f.write("%s/%s\n" % (TFR_MOD,os.path.basename(item)))

with open(out_dir_base + f'/{TFR_MOD}v.txt', 'w') as f:
	for item in files[split:]:
		f.write("%s/%s\n" % (TFR_MOD,os.path.basename(item)))

import ipdb
import socket
import random
from itertools import permutations
from collections import defaultdict
import numpy as np
import pickle
hostname = socket.gethostname()
st = ipdb.set_trace

if 'compute' in hostname:
    root_location = "/projects/katefgroup/datasets"
elif 'Alien' in hostname:
    root_location = "/media/mihir/dataset"

data_mod = "cg_F"
# new_mod = data_mod +"_F"

txt_file_train = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
txt_file_val = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
train_data = open(txt_file_train,"r").readlines()
# val_data = open(txt_file_val,"r").readlines()
data = train_data 
import pickle
filtered_filenames = []
words = []
tree_files = []
for example in data:
	example = example[:-1]
	example_npy = pickle.load(open(f"{root_location}/clevr_veggies/npys/{example}","rb"))
	tree_seq_filename = example_npy['tree_seq_filename']
	tree = pickle.load(open(f"{root_location}/clevr_veggies/{tree_seq_filename}","rb"))
	if tree.word not in ["cylinder","cube","sphere"]:
		filtered_filenames.append(f"{root_location}/clevr_veggies/npys/{example}")
		words.append(tree.word)
		tree_files.append(f"{root_location}/clevr_veggies/{tree_seq_filename}")

file_dicts = defaultdict(lambda: [])
secret_words = set(words)

for index,tree_file in enumerate(tree_files):
	tree_temp = pickle.load(open(f"{tree_file}","rb"))
	class_name = tree_temp.word
	file_dicts[class_name].append(filtered_filenames[index])

for key,value in file_dicts.items():
	value = list(permutations(value,2))
	file_dicts[key] = value

all_file_pairs = []
for key,value in file_dicts.items():
	if len(value) > 0:
		for val in value:
			all_file_pairs.append(val)

all_file_pairs = np.stack(all_file_pairs,axis=0)

pickle.dump(all_file_pairs,open("all_file_pairs.p","wb"))
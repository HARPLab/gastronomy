import ipdb
import socket
from collections import defaultdict
import numpy as np
from scipy.misc import imsave
import random

hostname = socket.gethostname()
st = ipdb.set_trace

if 'compute' in hostname:
    root_location = "/projects/katefgroup/datasets"
elif 'Alien' in hostname:
    root_location = "/media/mihir/dataset"

data_mod = "cg"
new_mod = data_mod +"_F"

txt_file_train = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
txt_file_val = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
train_data = open(txt_file_train,"r").readlines()
# val_data = open(txt_file_val,"r").readlines()
data = train_data 
import pickle
colors =['brown','yellow','cyan','blue','gray','purple','green','red']

default_dict = defaultdict(list)
def add_color(tree):
	color = None
	if "_" not in tree.word:
		if tree.children[0].word in colors:
			color = tree.children[0].word
		elif tree.children[0].children[0].word in colors:
			color = tree.children[0].children[0].word
		else:
			color = tree.children[0].children[0].children[0].word
			assert color in colors
		tree.word = tree.word + "_" + color
	return tree

word_files = {}
for example in data:
	example = example[:-1]
	example_npy = pickle.load(open(f"{root_location}/clevr_veggies/npys/{example}","rb"))
	tree_seq_filename = example_npy['tree_seq_filename']
	tree = pickle.load(open(f"{root_location}/clevr_veggies/{tree_seq_filename}","rb"))
	if tree.word in  ["cylinder","cube","sphere"]:
		tree = add_color(tree)
		pickle.dump(tree,open(f"{root_location}/clevr_veggies/{tree_seq_filename}","wb"))
	default_dict[tree.word].append(f"{root_location}/clevr_veggies/npys/{example}")
number_times = {key:len(item) for key,item in  default_dict.items()}
for key,item in default_dict.items():
	npy_file = item[0]
	example_npy = pickle.load(open(npy_file,"rb"))
	rgb = example_npy['rgb_camXs_raw'][0][...,:3]
	imsave(f"dump_out/{key}.png",rgb)
	print("hello")
st()
# check = np.array([len(item) for key,item in  default_dict.items()])
print("done")
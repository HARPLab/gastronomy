import ipdb
import socket
import random
hostname = socket.gethostname()
st = ipdb.set_trace

if 'compute' in hostname:
    root_location = "/projects/katefgroup/datasets"
elif 'Alien' in hostname:
    root_location = "/media/mihir/dataset"

data_mod = "cg"
new_mod = data_mod +"_shapes"

txt_file_train = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
txt_file_val = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
train_data = open(txt_file_train,"r").readlines()
# val_data = open(txt_file_val,"r").readlines()
data = train_data 
import pickle
filtered_filenames = []
words = []
for example in data:
	example = example[:-1]
	example_npy = pickle.load(open(f"{root_location}/clevr_veggies/npys/{example}","rb"))
	tree_seq_filename = example_npy['tree_seq_filename']
	tree = pickle.load(open(f"{root_location}/clevr_veggies/{tree_seq_filename}","rb"))
	if "cylinder" in tree.word.lower() or "cube"  in tree.word.lower()  or "sphere"  in tree.word.lower():
		filtered_filenames.append(example)
		words.append(tree.word)
random.shuffle(filtered_filenames)
print("original size",len(data),"filtered size",len(filtered_filenames))
st()
split = int(len(filtered_filenames)*0.95)
out_dir_base = f"{root_location}/clevr_veggies/npys"
with open(out_dir_base + '/%st.txt' % new_mod, 'w') as f:
    for item in filtered_filenames[:split]:
        f.write("%s\n" % item)

if 'Alien' in hostname:
	with open(out_dir_base + '/%sv.txt' % new_mod, 'w') as f:
	    for item in filtered_filenames[:split]:
	        f.write("%s\n" % item)	
else:
	with open(out_dir_base + '/%sv.txt' % new_mod, 'w') as f:
	    for item in filtered_filenames[split:]:
	        f.write("%s\n" % item)
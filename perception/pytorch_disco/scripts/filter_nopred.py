import ipdb
import socket
import random
hostname = socket.gethostname()
st = ipdb.set_trace

if 'compute' in hostname:
    root_location = "/home/mprabhud/dataset"
elif 'Alien' in hostname:
    root_location = "/media/mihir/dataset"

data_mod = "be_l"
new_mod = data_mod +"_shapes"

txt_file_train = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
# txt_file_val = f"{root_location}/clevr_veggies/npys/{data_mod}t.txt"
train_data = open(txt_file_train,"r").readlines()
# val_data = open(txt_file_val,"r").readlines()
data = train_data 
import pickle
filtered_filenames = []
words = []
nums_not = 0
nums = 0
for example in data:
	example = example[:-1]
	example_npy = pickle.load(open(f"{root_location}/clevr_veggies/npys/{example}","rb"))
	tree_seq_filename = example_npy['tree_seq_filename']
	tree = pickle.load(open(f"{root_location}/clevr_veggies/{tree_seq_filename}","rb"))
	if not hasattr(tree, 'bbox_det'):
		nums_not += 1
	else:
		nums += 1
print(nums_not,nums)
# random.shuffle(filtered_filenames)
# print("original size",len(data),"filtered size",len(filtered_filenames))
# st()
# split = int(len(filtered_filenames)*0.95)
# out_dir_base = f"{root_location}/clevr_veggies/npys"
# with open(out_dir_base + '/%st.txt' % new_mod, 'w') as f:
#     for item in filtered_filenames[:split]:
#         f.write("%s\n" % item)

# if 'Alien' in hostname:
# 	with open(out_dir_base + '/%sv.txt' % new_mod, 'w') as f:
# 	    for item in filtered_filenames[:split]:
# 	        f.write("%s\n" % item)	
# else:
# 	with open(out_dir_base + '/%sv.txt' % new_mod, 'w') as f:
# 	    for item in filtered_filenames[split:]:
# 	        f.write("%s\n" % item)
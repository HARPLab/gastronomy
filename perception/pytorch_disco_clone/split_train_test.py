import ipdb
st = ipdb.set_trace
data_mod = "bb_tv"
new_mod_1 = data_mod +"_a"
new_mod_2 = data_mod +"_b"
root_location = "/projects/katefgroup/datasets/"
txt_file_train = f"{root_location}/carla/npys/{data_mod}t.txt"

txt_file_train_1 = f"{root_location}/carla/npys/{new_mod_1}t.txt"
txt_file_train_2 = f"{root_location}/carla/npys/{new_mod_2}t.txt"

train_data = open(txt_file_train,"r").readlines()
num = int(len(train_data)*0.5)
train_data_1 = train_data[:num]
train_data_2 = train_data[num:]


with open(txt_file_train_1, 'w') as f:
	for item in train_data_1:
		if "*" not in item:
			f.write("%s" % item)

with open(txt_file_train_2, 'w') as f:
	for item in train_data_2:
		if "*" not in item:
			f.write("%s" % item)
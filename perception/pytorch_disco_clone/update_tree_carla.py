import ipdb
import pickle
st = ipdb.set_trace
data_mod = "bb"

root_location = "/projects/katefgroup/datasets/"
txt_file_train = f"{root_location}/carla/npys/{data_mod}t.txt"


train_data = open(txt_file_train,"r").readlines()
for file_name in train_data:
	pickle.load(open(file_name))

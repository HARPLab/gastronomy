import pickle 
import sys 
import os 
pname = sys.argv[1]

basepath = "/hdd/shamit/kinect/processed/single_obj"
p = pickle.load(open(os.path.join(basepath, pname), 'rb'))
p['shape_list'] = [] 
p['color_list'] = [] 
with open(os.path.join(basepath, pname), "wb") as fwrite:
    pickle.dump(p, fwrite)
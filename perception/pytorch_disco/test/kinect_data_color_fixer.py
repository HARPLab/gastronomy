import pickle 
import os
import ipdb 
st = ipdb.set_trace
basepath = "/projects/katefgroup/datasets/kinect/npys/nips_bbox"
pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]
allshapes = set()
allowed_shapes = ['onion', 'avocado', 'banana', 'tomato', 'apple']
sh2co = {'banana':'yellow', 'apple':'red', 'onion':'yellow', 'tomato':'red', 'avocado':'black'}
sh2actualco = {'banana':'yellow', 'apple':'red', 'onion':'pink', 'tomato':'orange', 'avocado':'black'}

color_list = []
actual_color_list = []
for pfile in pfiles:
    p = pickle.load(open(os.path.join(basepath, pfile), 'rb'))
    # st()
    actual_color_list = []
    color_list = []
    shape = p['shape_list']
    for s in shape:
        if s not in allowed_shapes:
            print(s)
            print(pfile)
            # st()
        color_list.append(sh2co[s])
        actual_color_list.append(sh2actualco[s])
        allshapes.add(s)
    p['color_list'] = color_list
    p['actual_color_list'] = actual_color_list
    with open(os.path.join(basepath, pfile), "wb") as fwrite:
        pickle.dump(p, fwrite)

print(allshapes)
print(len(allshapes))


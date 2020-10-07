import pickle
import glob
import ipdb
import numpy as np
st = ipdb.set_trace

def get_boxes_and_trees(trees):
	boxes = []
	for tree in trees:
		if tree.function == "describe":
			boxes.append((tree.bbox[3:]/128)*40)
	z,y,x = np.split(np.array(boxes),3,1)
	print("zmin",np.min(z),"zmax",np.max(z))
	print("ymin",np.min(y),"ymax",np.max(y))
	print("xmin",np.min(x),"xmax",np.max(x))

def get_all_boxes(trees):
	boxes = []
	for tree in trees:
		tree,box_tree = gen_list_of_bboxes(tree,boxes=[])
		boxes = boxes + box_tree
	boxes = np.stack(boxes)
	x,y,z = np.split(boxes,3,axis=1) 
	st()
	print("zmin",np.min(z),"zmax",np.max(z))
	print("ymin",np.min(y),"ymax",np.max(y))
	print("xmin",np.min(x),"xmax",np.max(x))

# zmin 5.5875 zmax 18.4625
# ymin -2.5625 ymax -0.0625
# xmin -6.8374996 xmax 7.4125004

# zmin 5 zmax 20
# ymin -3.0 ymax -0.0
# xmin -7.5 xmax 7.5

def gen_list_of_bboxes(tree,boxes= [],ref_T_mem=None):
	for i in range(0, tree.num_children):
		updated_tree,boxes = gen_list_of_bboxes(tree.children[i],boxes=boxes,ref_T_mem=ref_T_mem)
		tree.children[i] = updated_tree		
	if tree.function == "describe":
		bbox_origin = tree.bbox_origin
		boxes.append(bbox_origin[:3])
		boxes.append(bbox_origin[3:])		
	return tree,boxes


# between 3 and 7
folder = "/home/nel/Documents/datasets/katefgroup/datasets/clevr_veggies/CLEVR_DATASET_DEFAULT_C/trees_updated/train/*"
all_files = glob.glob(folder)
trees = [pickle.load(open(i,"rb")) for i in all_files]
get_all_boxes(trees)

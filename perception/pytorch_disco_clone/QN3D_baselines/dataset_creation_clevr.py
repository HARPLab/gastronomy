import numpy as np
import os 
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "trainer_quantize_object_no_detach_rotate"
os.environ["run_name"] = "check"
import lib_classes.Nel_Utils as nlu
import torch 
import pickle
import sys
import ipdb 
st = ipdb.set_trace

class create_baseline_dataset():
    def __init__(self):
        self.data_file = "/projects/katefgroup/datasets/clevr_veggies/npys/cgt.txt"
        self.num_samples_per_class = 2
        self.num_samples_per_class_test = 2
        self.train_sample_list = {}
        self.test_sample_list = {}
        # self.files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.p')]
        self.root_location = "/projects/katefgroup/datasets/clevr_veggies"
        self.save_path = "/home/shamitl/datasets/clevr_veggie_baseline"
        self.train_classes = []
        self.train_rgbs = []
        self.train_bbox = []
        self.train_origin_T_camX = []
        self.train_pix_T_camX = []
        self.train_camR_T_origin = []

        self.test_classes = []
        self.test_rgbs = []
        self.test_bbox = []
        self.test_origin_T_camX = []
        self.test_pix_T_camX = []
        self.test_camR_T_origin = []

    
    def process_rgb(self, rgbs):
        random_idx = np.random.permutation(rgbs.shape[0])[0]
        random_rgb = rgbs[random_idx][:,:,:3]/255.
        random_rgb = random_rgb.transpose(2, 0, 1)
        #Normalize using imagenet mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        random_rgb = (random_rgb-mean)/std
        return random_rgb, random_idx

    
    def process(self):
        datafile = open(self.data_file, "r")
        for cnt, line in enumerate(datafile.readlines()):
            # st()
            line = line.strip()
            print("Processing file {} with number {}".format(line, cnt))
            p = pickle.load(open(os.path.join(self.root_location, 'npys', line), "rb"))
            rgbs = p['rgb_camXs_raw']

            random_rgb, random_idx = self.process_rgb(rgbs)
            

            tree_seq_filename = p['tree_seq_filename']
            tree_filenames = os.path.join(self.root_location, tree_seq_filename)
            trees = [pickle.load(open(tree_filenames,"rb"))]
            gt_boxesR,scores,classes = nlu.trees_rearrange(trees)
            # st()

            origin_T_camX = p['origin_T_camXs_raw'][random_idx]
            pix_T_camX = p['pix_T_cams_raw'][random_idx]
            camR_T_origin = p['camR_T_origin_raw'][random_idx]

            objtype = classes[0][0]
            if objtype not in self.train_sample_list:
                self.train_sample_list[objtype] = 1
                self.train_classes.append(objtype)
                self.train_rgbs.append(random_rgb)

                self.train_bbox.append(gt_boxesR)
                self.train_origin_T_camX.append(origin_T_camX)
                self.train_pix_T_camX.append(pix_T_camX)
                self.train_camR_T_origin.append(camR_T_origin)

            elif self.train_sample_list[objtype] < self.num_samples_per_class:
                self.train_sample_list[objtype] += 1
                self.train_classes.append(objtype)
                self.train_rgbs.append(random_rgb)

                self.train_bbox.append(gt_boxesR)
                self.train_origin_T_camX.append(origin_T_camX)
                self.train_pix_T_camX.append(pix_T_camX)
                self.train_camR_T_origin.append(camR_T_origin)
            
            elif objtype not in self.test_sample_list:
                self.test_sample_list[objtype] = 1
                self.test_classes.append(objtype)
                self.test_rgbs.append(random_rgb)

                self.test_bbox.append(gt_boxesR)
                self.test_origin_T_camX.append(origin_T_camX)
                self.test_pix_T_camX.append(pix_T_camX)
                self.test_camR_T_origin.append(camR_T_origin)

            elif self.test_sample_list[objtype] < self.num_samples_per_class_test:
                self.test_sample_list[objtype] += 1
                self.test_classes.append(objtype)
                self.test_rgbs.append(random_rgb)

                self.test_bbox.append(gt_boxesR)
                self.test_origin_T_camX.append(origin_T_camX)
                self.test_pix_T_camX.append(pix_T_camX)
                self.test_camR_T_origin.append(camR_T_origin)

            #Remove this after testing
            # if len(self.train_classes) > 20:
            #     break
                
        print("Num train objects %d and num test objects %d " %(len(self.train_classes), len(self.test_classes)))
        # pdict = {'train_class': self.train_classes, 'train_rgbs': self.train_rgbs, 'test_class': self.test_classes, 'test_rgbs': self.test_rgbs}
        pdict = {'train_camR_T_origin': self.train_camR_T_origin, 'train_bbox': self.train_bbox, 'train_origin_T_camX': self.train_origin_T_camX, 'train_pix_T_camX': self.train_pix_T_camX, 'train_class': self.train_classes, 'train_rgbs': self.train_rgbs, 'test_camR_T_origin': self.test_camR_T_origin,'test_bbox': self.test_bbox, 'test_origin_T_camX': self.test_origin_T_camX, 'test_pix_T_camX': self.test_pix_T_camX, 'test_class': self.test_classes, 'test_rgbs': self.test_rgbs}
  
        with open(os.path.join(self.save_path, 'clever_fewshot.p'), 'wb') as f:
            pickle.dump(pdict, f)

if __name__ == '__main__':
    cbd = create_baseline_dataset()
    cbd.process()
        
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
        self.data_file = "/home/mprabhud/dataset/carla/npy/bb"
        self.pickle_files = [os.path.join(self.data_file, f) for f in os.listdir(self.data_file) if f.endswith('.p')]

        self.num_samples_per_class = 2
        self.num_samples_per_class_test = 2
        self.train_sample_list = {}
        self.test_sample_list = {}
        # self.files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.p')]
        self.save_path = "/home/mprabhud/dataset/carla_baseline"
        self.train_classes = []
        self.train_rgbs = []
        self.train_bbox = []
        self.train_origin_T_camX = []
        self.train_pix_T_camX = []

        self.test_classes = []
        self.test_rgbs = []
        self.test_bbox = []
        self.test_origin_T_camX = []
        self.test_pix_T_camX = []

    
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

        for cnt, line in enumerate(self.pickle_files):
            # st()
            line = line.strip()
            print("Processing file {} with number {}".format(line, cnt))
            p = pickle.load(open(line, "rb"))
            rgbs = p['rgb_camXs_raw']

            random_rgb, random_idx = self.process_rgb(rgbs)
            origin_T_camX = p['origin_T_camXs_raw'][random_idx]
            pix_T_camX = p['pix_T_cams_raw'][random_idx]
            bbox_origin = p['bbox_origin']
            objtype = p['obj_name']
        
            if objtype not in self.train_sample_list:
                self.train_sample_list[objtype] = 1
                self.train_classes.append(objtype)
                self.train_rgbs.append(random_rgb)

                self.train_bbox.append(bbox_origin)
                self.train_origin_T_camX.append(origin_T_camX)
                self.train_pix_T_camX.append(pix_T_camX)

            elif self.train_sample_list[objtype] < self.num_samples_per_class:
                self.train_sample_list[objtype] += 1
                self.train_classes.append(objtype)
                self.train_rgbs.append(random_rgb)

                self.train_bbox.append(bbox_origin)
                self.train_origin_T_camX.append(origin_T_camX)
                self.train_pix_T_camX.append(pix_T_camX)
            
            elif objtype not in self.test_sample_list:
                self.test_sample_list[objtype] = 1
                self.test_classes.append(objtype)
                self.test_rgbs.append(random_rgb)

                self.test_bbox.append(bbox_origin)
                self.test_origin_T_camX.append(origin_T_camX)
                self.test_pix_T_camX.append(pix_T_camX)

            elif self.test_sample_list[objtype] < self.num_samples_per_class_test:
                self.test_sample_list[objtype] += 1
                self.test_classes.append(objtype)
                self.test_rgbs.append(random_rgb)

                self.test_bbox.append(bbox_origin)
                self.test_origin_T_camX.append(origin_T_camX)
                self.test_pix_T_camX.append(pix_T_camX)

            #Remove this after testing
            # if len(self.train_classes) > 20:
            #     break
                
        print("Num train objects %d and num test objects %d " %(len(self.train_classes), len(self.test_classes)))
        pdict = {'train_bbox': self.train_bbox, 'train_origin_T_camX': self.train_origin_T_camX, 'train_pix_T_camX': self.train_pix_T_camX, 'train_class': self.train_classes, 'train_rgbs': self.train_rgbs, 'test_bbox': self.test_bbox, 'test_origin_T_camX': self.test_origin_T_camX, 'test_pix_T_camX': self.test_pix_T_camX, 'test_class': self.test_classes, 'test_rgbs': self.test_rgbs}
                
        with open(os.path.join(self.save_path, 'carla_fewshot.p'), 'wb') as f:
            pickle.dump(pdict, f)

if __name__ == '__main__':
    cbd = create_baseline_dataset()
    cbd.process()
        
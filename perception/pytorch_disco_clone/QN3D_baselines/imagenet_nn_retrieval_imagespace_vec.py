'''
Crops boxes from images, resized individual boxes, passes them thru resnet and calculates similarity
'''
import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "trainer_quantize_object_no_detach_rotate"
os.environ["run_name"] = "check"
import numpy as np 
import torch 
import torchvision.models as models
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys 
import ipdb 
import pickle
import utils_geom
import time
import random
import lib_classes.Nel_Utils as nlu
import cv2
import cross_corr
import getpass
username = getpass.getuser()
st = ipdb.set_trace

device = torch.device("cuda")
from DoublePool import ClusterPool
class imagenet_nn_retrieval():
    def __init__(self, root, modfile, crop_obj, mod_folder, hyp_N, dataset_name, test_rotations, exp_name):
        self.test_rotations = test_rotations
        self.dataset_name = dataset_name
        self.H = 224
        self.W = 224
        self.mbr = cross_corr.meshgrid_based_rotation(self.H, self.H, self.W, angleIncrement=50)
        self.root = root
        self.modfile = modfile
        self.mod_folder = mod_folder
        self.crop_obj = crop_obj
        self.model = models.resnet18(pretrained=True)
        
        self.highest_recall = 20
        self.exp_name = exp_name
        if self.dataset_name == "carla":
            self.num_views = 17
        elif self.dataset_name == "clevr":
            self.num_views = 24
        elif self.dataset_name == "bigbird":
            self.num_views = 2

        self._data = [] # Names of all pickle files in the {mod}t.txt file
        self.hyp_N = hyp_N
        self.hyp_B = 1
        self.class_labels = {}
        self.images = []
        self.labels = []
        start_time = time.time()
        self.create_pool()
        end_time = time.time()
        print("Took {} seconds to create dataset".format(end_time - start_time))
        train_images, train_labels, test_images, test_labels = self.generate_train_test_datapts()
        if self.exp_name == "fewshot_nearest_neighbor" or self.exp_name == "full_nearest_neighbor":
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
            self.model.eval()
            self.model = self.model.to(device)
            self.evaluate_nearest_neighbor(train_images, train_labels, test_images, test_labels)
        elif self.exp_name == "fewshot_classification":
            self.evaluate_classification(train_images, train_labels, test_images, test_labels)


    def evaluate_classification(self, train_images, train_labels, test_images, test_labels):
        num_labels = torch.unique(train_labels).shape[0]
        self.model.fc = nn.Linear(512, num_labels)
        self.model = self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        st()
        epochs = 1000
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Forward pass
            output = self.model(train_images)
            # Calculate the loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, train_labels)
            loss.backward()
            # Optimizer takes one step
            optimizer.step()

            # # Log info
            # if epoch % param['log_every'] == 0:
            #     # todo: add your visualization code
            #     writer.add_scalar('Train Loss', loss.item(), epoch)

            
            # Validation iteration
            self.model.eval()
            with torch.no_grad():
                test_output = self.model(test_images)
                test_preds = torch.argmax(test_output, dim=1)
                corr_preds = test_preds == test_labels
                test_acc = 1.*torch.sum(corr_preds)/len(test_labels)

                train_preds = torch.argmax(output, dim=1)
                corr_preds = train_preds == train_labels
                train_acc = torch.sum(corr_preds)/len(train_labels)
                # st()
            # writer.add_scalar('Test accuracy', test_acc.item(), epoch)
            # writer.add_scalar('Train accuracy', train_acc.item(), epoch)
            # print("Test accuracy is ",test_acc)
            # print("Train accuracy is: ", train_acc)
            print('Train Epoch: {}, Train Loss {} Train accuracy {} Test accuracy {}'.format(str(epoch), str(loss.item()), str(train_acc.item()), str(test_acc.item())))
            self.model.train()
            
            # if epoch % param['checkpoint_every'] == 0:
            #     save_model(epoch, model, optimizer, epoch, exp_name, dataset_name)


    def generate_train_test_datapts(self):
        if self.exp_name == "full_nearest_neighbor":
            # st()
            if len(self.images) > 1500:
                self.images = self.images[:1500]
                self.labels = self.labels[:1500]
            train_num = int(len(self.images)*0.7)
            test_num = int(len(self.images) - train_num)
            # st()
            train_images = self.process_rgb(np.array(self.images[:train_num])).to(device)
            train_labels = torch.tensor(np.array(self.labels[:train_num])).to(device)

            test_images = self.process_rgb(np.array(self.images[train_num:])).to(device)
            test_labels = torch.tensor(np.array(self.labels[train_num:])).to(device)
        
        elif self.exp_name == "fewshot_nearest_neighbor" or self.exp_name == "fewshot_classification":
            train_images, train_labels = [], []
            test_images, test_labels = [], []

            label_cnt = {}
            for image, label in zip(self.images, self.labels):
                if label not in label_cnt:
                    label_cnt[label] = 0
                if label_cnt[label]<2:
                    train_images.append(image)
                    train_labels.append(label)
                    label_cnt[label] += 1
                elif label_cnt[label]<4:
                    test_images.append(image)
                    test_labels.append(label)
                    label_cnt[label] += 1
            
            # st()
            train_images = self.process_rgb(np.stack(train_images, axis=0)).float().to(device)
            train_labels = torch.tensor(train_labels).to(device)

            test_images = self.process_rgb(np.stack(test_images, axis=0)).float().to(device)
            test_labels = torch.tensor(test_labels).to(device)

        return train_images, train_labels, test_images, test_labels

        
    def evaluate_nearest_neighbor(self, train_images, train_labels, test_images, test_labels):
        

        with torch.no_grad():
            train_features = self.model(train_images) #torch.Size([100, 512, 1, 1]) (resnet)
            # st()
            B, _, _, _ = train_features.shape
            train_features = train_features.reshape(B, -1)
        
        R = 1
        if self.test_rotations:
            rotated_test = self.mbr.rotate2D(test_images) # torch.Size([56, 3, 36, 224, 224])
            rotated_test = rotated_test.permute(0, 2, 1, 3, 4)
            test_labels = test_labels.reshape(-1,1).repeat(1, rotated_test.shape[1])
            
            B, R, C, H, W = rotated_test.shape
            test_images = rotated_test.reshape(B*R, C, H, W)
            test_labels = test_labels.reshape(-1)
        test_features_list = []
        for i in range(0, test_images.shape[0], 900):
            with torch.no_grad():
                j=min(i+900, test_images.shape[0])
                test_features = self.model(test_images[i:j]) #torch.Size([100, 512, 1, 1]) (resnet)
                B, _, _, _ = test_features.shape
                test_features = test_features.reshape(B, -1)
                test_features_list.append(test_features)
        
        # st()
        test_features = torch.cat(test_features_list, dim=0)
        cnt = 0
        # predicted_test_labels = []
        train_feature_to_score_map = torch.zeros(train_features.shape[0]).to(device)
        best_test_retrievals = []
        precision_recall = []
        for test_feature, test_label in zip(test_features, test_labels):
            # st()    
            print("Processing test feature number and label :", cnt, test_label)
            
            mini = 1000000000000
            mini_idx = -1
            if cnt%R == 0:                        
                # Clear and assign infinity value
                train_feature_to_score_map = torch.zeros(train_features.shape[0]).to(device) + mini
            # st()
            test_feature = test_feature.reshape(1,-1)
            dist = torch.norm(test_feature - train_features, dim=1)

            update_idxs = torch.where(dist<train_feature_to_score_map)
            train_feature_to_score_map[update_idxs] = dist[update_idxs]
            
            # if False: # Remove this to save images
            #     trainsave = original_train_images[mini_idx]
            #     testsave = original_test_images[int((cnt)//R)]
            #     image_to_save = np.concatenate((trainsave, testsave), axis=0)
            #     if "mprabhud" in username:
            #         plt.imsave("/home/mprabhud/shamit/clevr_imagenet/{}_{}.jpg".format(cnt, predicted_test_labels[-1]), image_to_save)
            #     elif "shamitl" in username:
            #         plt.imsave("/home/shamitl/vis/clevr/{}_{}.jpg".format(cnt, predicted_test_labels[-1]), image_to_save)
            
            cnt += 1
            if cnt%R == 0:
                precision_recall_sample = []
                best_test_retrievals = []
                sorted_val, sorted_idxs = train_feature_to_score_map.sort()
                sorted_val = sorted_val[: self.highest_recall]
                sorted_idxs = sorted_idxs[: self.highest_recall]
                # best_test_retrievals = best_test_retrievals[: self.highest_recall]
                # st()
                for idx in sorted_idxs:
                    # _, idx = j
                    if train_labels[idx] == test_label:
                        precision_recall_sample.append(1)
                    else:
                        precision_recall_sample.append(0)
                precision_recall.append(precision_recall_sample)

        precision_recall = np.array(precision_recall)
        cummulative = np.cumsum(precision_recall, axis=1)
        deno = np.arange(cummulative.shape[1]).reshape(1,-1) + 1
        deno = deno.astype(np.float32)
        mean_cummulative = cummulative/deno 
        precisions = np.mean(mean_cummulative, axis=0)
        print("Precisions are: ", precisions)
        # st()
        # predicted_test_labels = np.stack(predicted_test_labels)
        # predicted_test_labels = torch.tensor(predicted_test_labels.reshape(-1,R))
        # predicted_test_labels = torch.clamp(torch.sum(predicted_test_labels, dim=1), 0, 1)
        # accuracy = torch.sum(predicted_test_labels)/(1.0*len(predicted_test_labels))
        # print("Accuracy is : ", accuracy)
        

    def process_rgb(self, rgbs):
        rgbs = rgbs[:,:,:,:3]/255.
        rgbs = rgbs.transpose(0, 3, 1, 2)
        #Normalize using imagenet mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        rgbs = (rgbs-mean)/std
        return torch.tensor(rgbs).float()

    def create_pool(self):
        with open(self.modfile, 'r') as f:
            for filename_i in f.readlines():
                filename = filename_i[:-1]
                filepath = f"{self.mod_folder}/{filename}"
                self._data.append(filepath)
        # st()
        random.shuffle(self._data)
        print("Total size of data: ", len(self._data))
        for i in range(len(self._data)):
            print("Processing index: ", i)
            rgb_images_ret, rgb_labels_ret = self.get_item(i)
            for rgb_image, label in zip(rgb_images_ret, rgb_labels_ret):
                if label in self.class_labels:
                    label = self.class_labels[label]
                else:
                    val = len(self.class_labels.keys())
                    self.class_labels[label] = val
                    label = val
                self.images.append(rgb_image)
                self.labels.append(label)
        # st()
        c = list(zip(self.images, self.labels))
        random.shuffle(c)
        self.images, self.labels = zip(*c)
        # Uncomment this to debug label associations.
        # cnt = 0
        # for image, label in zip(self.images, self.labels):
        #     cnt+=1
        #     if "mprabhud" in username:
        #         plt.imsave("/home/mprabhud/shamit/vis/{}/{}_label-{}.jpg".format(self.dataset_name,cnt, label), image)
        # st()
                
    def get_item(self, index):
        # st()
        data = pickle.load(open(self._data[index],"rb"))
        view_to_take = np.random.randint(0,self.num_views)
        random_rgb = data['rgb_camXs_raw'][view_to_take][...,:3]
        rgb_to_return = np.copy(random_rgb) # (256, 256, 3)
        # st()
        camR_T_origin_raw = torch.from_numpy(data["camR_T_origin_raw"][view_to_take:view_to_take+1]).unsqueeze(dim=0).float()
        pix_T_cams_raw = torch.from_numpy(data["pix_T_cams_raw"][view_to_take:view_to_take+1]).unsqueeze(dim=0).float()
        origin_T_camXs_raw = torch.from_numpy(data["origin_T_camXs_raw"][view_to_take:view_to_take+1]).unsqueeze(dim=0).float()
        
        #Crop out the main object
        __pb = lambda x: self.pack_boxdim(x, self.hyp_N)
        __ub = lambda x: self.unpack_boxdim(x, self.hyp_N)            
        __p = lambda x: self.pack_seqdim(x, self.hyp_B)
        __u = lambda x: self.unpack_seqdim(x, self.hyp_B)

        pix_T_cams = pix_T_cams_raw
        # cam_T_velos = feed["cam_T_velos"]
        origin_T_camRs = __u(utils_geom.safe_inverse(__p(camR_T_origin_raw)))
        origin_T_camXs = origin_T_camXs_raw


        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
        camX0_T_camRs = camXs_T_camRs[:,0]
        camR_T_camX0  = utils_geom.safe_inverse(camX0_T_camRs)

        tree_file_name = data['tree_seq_filename']
        tree_file_path = f"{self.root}/{tree_file_name}"
        trees = pickle.load(open(tree_file_path,"rb"))
        # st()
        if self.dataset_name == "clevr":
            gt_boxesR,scores,classes = self.trees_rearrange(trees)
        elif self.dataset_name == "bigbird":
            gt_boxesR,scores,classes = self.trees_rearrange(trees)
            gt_boxesR = np.array([-100,-100,-100,100,100,100]).reshape(1,1,6)
        elif self.dataset_name == "carla":
            classes = np.array([[data['obj_name']]])
            scores = None
            gt_boxesR = data['bbox_origin'].reshape(1,1,-1) # This is actually origin.
            camX0_T_origin = utils_geom.safe_inverse(__p(origin_T_camXs))
            camX0_T_camRs = camX0_T_origin.float()

        gt_boxesR = torch.from_numpy(gt_boxesR).float()
        gt_boxesR_end = torch.reshape(gt_boxesR,[self.hyp_B,self.hyp_N,2,3])

        gt_boxesR_theta = nlu.get_alignedboxes2thetaformat(gt_boxesR_end)
        gt_boxesR_corners = utils_geom.transform_boxes_to_corners(gt_boxesR_theta).float()

        gt_boxesX0_corners = __ub(utils_geom.apply_4x4(camX0_T_camRs, __pb(gt_boxesR_corners)))
        gt_cornersX0_pix = __ub(utils_geom.apply_pix_T_cam(pix_T_cams[:,0], __pb(gt_boxesX0_corners)))
        boxes_pix = nlu.get_ends_of_corner(gt_cornersX0_pix)
        boxes_pix = torch.clamp(boxes_pix,min=0)            

        rgb_to_return_list = []
        obj_classes = []
        # st()
        for ii in range(len(classes[0])):
            # st()
            if classes[0,ii]=='0':
                break
            tmp_box = boxes_pix[0,ii]
            obj_class = classes[0,ii]

            lower,upper = torch.unbind(tmp_box)
            xmin,ymin = torch.floor(lower).to(torch.int16)
            xmax,ymax = torch.ceil(upper).to(torch.int16)
            object_rgb = random_rgb[ymin:ymax,xmin:xmax]
            # st()
            # if "mprabhud" in username:
            #     plt.imsave("/home/mprabhud/shamit/vis/{}/index-{}_ii-{}.jpg".format(self.dataset_name,index, ii), object_rgb)
            
            # if self.crop_obj:
            rgb_to_return = object_rgb
            try:
                rgb_to_return = cv2.resize(rgb_to_return,(int(self.H),int(self.W)))
                rgb_to_return_list.append(rgb_to_return)
                obj_classes.append(obj_class)
            except Exception as e:
                print("Got exception :", e)
            

        return rgb_to_return_list, obj_classes
        # return image, label

    def trees_rearrange(self,tree):
        tree,boxes,classes = self.bbox_rearrange(tree,boxes=[],classes=[])
        # st()
        boxes = np.stack(boxes)
        classes = np.stack(classes)
        N,_  = boxes.shape 
        scores = np.pad(np.ones([N]),[0,self.hyp_N-N])
        boxes = np.pad(boxes,[[0,self.hyp_N-N],[0,0]])
        # st()
        classes = np.pad(classes,[0,self.hyp_N-N])
        scores = np.expand_dims(scores,axis=0)
        boxes = np.expand_dims(boxes,axis=0)
        classes = np.expand_dims(classes,axis=0)
        return boxes,scores,classes

    def pack_boxdim(self,tensor, N):
        shapelist = list(tensor.shape)
        B, N_, C = shapelist[:3]
        assert(N==N_)
        # assert(C==8)
        otherdims = shapelist[3:]
        tensor = torch.reshape(tensor, [B,N*C]+otherdims)
        return tensor

    def unpack_boxdim(self,tensor, N):
        shapelist = list(tensor.shape)
        B,NS = shapelist[:2]
        assert(NS%N==0)
        otherdims = shapelist[2:]
        S = int(NS/N)
        tensor = torch.reshape(tensor, [B,N,S]+otherdims)
        return tensor
    
    def pack_seqdim(self,tensor, B):
        shapelist = list(tensor.shape)
        B_, S = shapelist[:2]
        assert(B==B_)
        otherdims = shapelist[2:]
        tensor = torch.reshape(tensor, [B*S]+otherdims)
        return tensor


    def unpack_seqdim(self,tensor, B):
        shapelist = list(tensor.shape)
        BS = shapelist[0]
        assert(BS%B==0)
        otherdims = shapelist[1:]
        S = int(BS/B)
        tensor = torch.reshape(tensor, [B,S]+otherdims)
        return tensor
    
    def bbox_rearrange(self,tree,boxes= [],classes=[]):
        for i in range(0, tree.num_children):
            updated_tree,boxes,classes = self.bbox_rearrange(tree.children[i],boxes=boxes,classes=classes)
            tree.children[i] = updated_tree     
        if tree.function == "describe":
            xmax,ymax,zmin,xmin,ymin,zmax = tree.bbox_origin
            box = np.array([xmin,ymin,zmin,xmax,ymax,zmax])
            tree.bbox_origin = box
            boxes.append(box)
            classes.append(tree.word)
        return tree,boxes,classes

if __name__ == '__main__':
    dataset = sys.argv[1]
    exp_name = sys.argv[2]
    if dataset == "clevr":
        if "shamitl" in username:
            root = '/projects/katefgroup/datasets/clevr_veggies'
            mod = 'cg'
            mod_folder = f"{root}/npys"
            modfile = f"{mod_folder}/{mod}v.txt"
            hyp_N = 3
            test_rotations = False
        elif "mprabhud" in username:
            root = '/home/mprabhud/dataset/clevr_veggies'
            mod = 'be_l'
            mod_folder = f"{root}/npys"
            modfile = f"{mod_folder}/{mod}t.txt"
            hyp_N = 3
            test_rotations = True
    elif dataset == "carla":
        if "mprabhud" in username:
            root = '/home/mprabhud/dataset/carla'
            mod = 'bb'
            mod_folder = f"{root}/npy"
            modfile = f"{mod_folder}/{mod}t.txt"
            hyp_N = 1
            test_rotations = True
    elif dataset == "bigbird":
        if "mprabhud" in username:
            root = '/projects/katefgroup/datasets/bigbird_processed'
            mod = 'gg'
            mod_folder = f"{root}/npy"
            modfile = f"{mod_folder}/{mod}t.txt"
            hyp_N = 1
            test_rotations = True


    nn = imagenet_nn_retrieval(root, modfile, True, mod_folder, hyp_N, dataset, test_rotations, exp_name)
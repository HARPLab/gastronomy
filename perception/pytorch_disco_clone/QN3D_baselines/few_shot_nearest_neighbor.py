import numpy as np 
import torch 
import torchvision.models as models
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import ipdb 
import pickle
from QN3D_baselines.train import params, process_data
st = ipdb.set_trace

device = torch.device("cuda")

class nearest_neighbor():
    def __init__(self, dataset_name, checkpoint_file):
        self.param = params[dataset_name]
        self.dataset_name = dataset_name
        checkpoint = torch.load(checkpoint_file)
        model = models.resnet18()
        model.fc = nn.Linear(512, self.param['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = torch.nn.Sequential(*(list(model.children())[:-1]))

        
        model.eval()
        self.model = model.to(device)

    
    def find_nearest_neigbor(self):
        # st()
        features_list = []
        target_list = []
        data_list = []

        datafile = self.param['datafile']
        num_classes = self.param['num_classes']
        data = pickle.load(open(datafile, 'rb'))
        train_rgbs, train_labels, test_rgbs, test_labels = process_data(data, num_classes)


        with torch.no_grad():
            # st()
            test_features = self.model(test_rgbs) #torch.Size([100, 512, 1, 1]) (resnet)
            B, N, _, _ = test_features.shape
            test_features = test_features.reshape(B, N)
        
        cor_preds = 0
        # st()
        for i in range(B):
            mini = 1000000000000
            mini_idx = -1
            
            for j in range(B):
                if i==j:
                    continue

                dist = torch.norm(test_features[i] - test_features[j])
                if dist < mini:
                    mini = dist
                    mini_idx = j
            if test_labels[i] == test_labels[mini_idx]:
                cor_preds += 1
        accuracy = cor_preds/len(test_labels)
        print("Accuracy: ", accuracy)

            
                
        
        



if __name__ == "__main__":
    dataset_name = sys.argv[1]
    checkpoint_file = "baseline_checkpoints/dataset_clevr_expname_exp_2_model_000500.pth"
    nn = nearest_neighbor(dataset_name, checkpoint_file)
    nn.find_nearest_neigbor()

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class DMPWeightData(Dataset):
    def __init__(self, data_loc,num_dims,num_labels,MNIST=False,transform=None):
        self.MNIST=MNIST
        self.num_dims=num_dims
        self.num_labels=num_labels
        if self.MNIST==False:
            self.transform = transform
            DMPweights_df = pd.read_csv(data_loc,header=None)       
            #import ipdb;ipdb.set_trace() 
            if self.num_labels!=0:
                self.labels = DMPweights_df[DMPweights_df.columns[0]]        
                self.DMPweights = DMPweights_df.iloc[:, 1:].values.astype('float64').reshape(-1,1,1,self.num_dims)
            elif self.num_labels==0:
                self.DMPweights = DMPweights_df.iloc[:, 1:].values.astype('float64').reshape(-1,1,1,self.num_dims)
            

        elif self.MNIST==True:
            self.transform = transform
            fashion_df = pd.read_csv('/home/test2/Documents/Data/fashion-mnist_train.csv')   
            self.labels = fashion_df.label.values
            self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)


    def __len__(self):
        if self.MNIST==False:
            length=len(self.DMPweights)
             
        elif self.MNIST==True:
            length=len(self.images)
        
        return length 


    def __getitem__(self, idx):
        #import pdb; pdb.set_trace()
        if self.MNIST==False:
            if self.num_labels!=0:
                label = self.labels[idx]
                DMPwts = self.DMPweights[idx]
                return DMPwts, label

            elif self.num_labels==0:
                DMPwts = self.DMPweights[idx]
                return DMPwts


        elif self.MNIST==True:
            label = self.labels[idx]
            DMPwts = Image.fromarray(self.images[idx])        
            if self.transform:
                DMPwts = self.transform(DMPwts) 
                return DMPwts, label
    

def create_data_loader(dataset,batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    return data_loader

def visualize_colorplot_weights(data):
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256)) 
    newcmp = ListedColormap(newcolors)

    for i in range(5,6):      
        fig, axs = plt.subplots(1, 2, figsize=(12, 0.7), constrained_layout=True)
        for [ax, cmap] in zip(axs, [viridis]):
            psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-1, vmax=1)
            fig.colorbar(psm, ax=ax)
        #plt.show()
        
    return fig

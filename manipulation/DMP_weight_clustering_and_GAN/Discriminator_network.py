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

class Discriminator(nn.Module):
    def __init__(self,num_labels,num_dims,MNIST=False):
        super().__init__()       

        self.MNIST=MNIST       
        self.num_labels=num_labels
        self.num_dims=num_dims

        if self.MNIST==False:
            if self.num_labels!=0:
                self.label_emb = nn.Embedding(self.num_labels, self.num_labels) 
            self.model = nn.Sequential(            
           #DMP data -3 layers
            nn.Linear(self.num_dims+self.num_labels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()           
        )
    

        elif self.MNIST==True:
            self.label_emb = nn.Embedding(10, 10)       
            self.model = nn.Sequential(
                #MNIST data
                nn.Linear(794, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x, labels): 
        if self.MNIST==False:                   
            x = x.view(x.size(0), self.num_dims)            

        elif self.MNIST==True:   
            x = x.view(x.size(0), 784)    

        if self.num_labels!=0:
            c = self.label_emb(labels)       
            x = torch.cat([x, c], 1)            
            out = self.model(x)
        elif self.num_labels==0:
            out=self.model(x)        
        return out.squeeze()
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

class Generator(nn.Module):
    def __init__(self,num_labels,num_dims,z_size,MNIST=False):
        super().__init__()
       
        self.MNIST=MNIST    
        self.num_labels=num_labels
        self.num_dims=num_dims
        self.z_size=z_size

        if self.MNIST==False:  
            if self.num_labels!=0:
                self.label_emb = nn.Embedding(self.num_labels, self.num_labels) 
            self.model = nn.Sequential(           
            #DMP data - 3 layers
            nn.Linear(self.z_size+self.num_labels, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.num_dims),
            nn.Tanh()  
            )
        
        elif self.MNIST==True:
            self.label_emb = nn.Embedding(10, 10)       
            self.model = nn.Sequential(
            #MNIST data
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()                
        )
    
    def forward(self, z, labels):          
        if self.MNIST==False:    
            z = z.view(z.size(0), self.z_size)  
            if self.num_labels!=0:                   
                c = self.label_emb(labels)       
                x = torch.cat([z, c], 1)         
                out = self.model(x)     
            elif self.num_labels==0:
                out=self.model(z)       
            return out

        elif self.MNIST==True: 
            z = z.view(z.size(0), 100) 
            c = self.label_emb(labels)        
            x = torch.cat([z, c], 1)        
            out = self.model(x)
            return out.view(x.size(0), 28, 28)

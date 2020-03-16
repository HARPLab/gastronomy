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

def generator_train_step(batch_size, discriminator, generator, g_optimizer, loss, num_labels,z_size,MNIST=False):
    if MNIST==False:
        #Reset gradients
        g_optimizer.zero_grad()        
        z = Variable(torch.randn(batch_size, z_size)) #z is random noise        
        if num_labels!=0:     
            fake_labels = Variable(torch.LongTensor(np.random.randint(0, num_labels, batch_size)))        
            fake_DMPweights = generator(z, fake_labels)
        elif num_labels==0:
            labels=0
            fake_DMPweights = generator(z,labels)

        if num_labels!=0:
            validity = discriminator(fake_DMPweights, fake_labels)
        elif num_labels==0:
            validity = discriminator(fake_DMPweights,labels)

        g_loss = loss(validity, Variable(torch.ones(batch_size)))    
        g_loss.backward()        
        g_optimizer.step()

    elif MNIST==True:
        criterion=loss
        g_optimizer.zero_grad()
        z = Variable(torch.randn(batch_size, 100))        
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size)))
        fake_images = generator(z, fake_labels)
        validity = discriminator(fake_images, fake_labels)
        g_loss = criterion(validity, Variable(torch.ones(batch_size)))
        g_loss.backward()
        g_optimizer.step()
   
    return g_loss.data.item()


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, loss, real_DMPweights, labels,num_labels,z_size,MNIST=False):
    if MNIST==False:
        d_optimizer.zero_grad()        
        # train with real DMPweights   
        if num_labels!=0:
            real_validity = discriminator(real_DMPweights, labels) 
        elif num_labels==0:
            real_validity = discriminator(real_DMPweights,labels)

        #real_loss = loss(real_validity, Variable(torch.ones(batch_size)))
        real_loss = loss(real_validity, Variable(torch.FloatTensor(np.random.uniform(0.7, 1.2, batch_size))))
        #real_loss = loss(real_validity, Variable(torch.FloatTensor(np.random.uniform(0, 0.3, batch_size))))
        
        # train with fake DMPweights        
        z = Variable(torch.randn(batch_size, z_size)) #z is random noise  

        if num_labels!=0:
            fake_labels = Variable(torch.LongTensor(np.random.randint(0, num_labels, batch_size)))
            fake_DMPweights = generator(z, fake_labels)
            fake_validity = discriminator(fake_DMPweights, fake_labels)
        elif num_labels==0:
            fake_DMPweights=generator(z,labels)
            fake_validity=discriminator(fake_DMPweights,labels)

        #fake_loss = loss(fake_validity, Variable(torch.zeros(batch_size)))
        fake_loss = loss(fake_validity,Variable(torch.FloatTensor(np.random.uniform(0, 0.3, batch_size))))
        #fake_loss = loss(fake_validity,Variable(torch.FloatTensor(np.random.uniform(0.7, 1.2, batch_size))))
        
        d_loss = real_loss + fake_loss  
        d_loss.backward()  
        d_optimizer.step()

    elif MNIST==True:        
        real_images=real_DMPweights
        criterion=loss
        d_optimizer.zero_grad()
        # train with real images
        real_validity = discriminator(real_images, labels)
        real_loss = criterion(real_validity, Variable(torch.ones(batch_size)))        
        # train with fake images
        z = Variable(torch.randn(batch_size, 100))
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size)))
        fake_images = generator(z, fake_labels)
        fake_validity = discriminator(fake_images, fake_labels)
        fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)))        
        d_loss = real_loss + fake_loss
        d_loss.backward()  
        d_optimizer.step()

    return d_loss.data.item()
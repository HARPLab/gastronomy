#Train GAN or cGAN
    #set num_labels=0 if training regular GAN
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

from Discriminator_network import Discriminator
from Generator_network import Generator
from train_GAN import generator_train_step, discriminator_train_step
from data import DMPWeightData, create_data_loader, visualize_colorplot_weights

from torch.utils.tensorboard import SummaryWriter

num_dims=27
num_labels=0
labels=0
z_size=3

writer = SummaryWriter('/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/GAN/runs/exp35')

#Instantiate dataset
#DMP dataset
#data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/combinedWts_transformedRefFrame/XYZABG_weights_combined/combined_weights_all_6dims_with_initial_orientations_num_labels.txt'
#data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/combinedWts_transformedRefFrame/XYZABG_weights_combined/combined_wts_xyz_dims.txt'
#data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/combinedWts_transformedRefFrame/XYZABG_weights_combined/18dims_noScoring.txt'

#data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/013120_updated_DMPwts/XYZABG_weights_combined/combined_weights_all_XYZB_4dims.txt'
#data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/013120_updated_DMPwts/XYZABG_weights_combined/combined_weights_all_XYZB_4dims_w_IOs.txt'
# data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/013120_updated_DMPwts/XYZABG_weights_combined/combined_weights_all_XYZB_4dims_w_normalized_IOs.txt'
#data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/013120_updated_DMPwts/XYZABG_weights_combined/combined_weights_all_XYZB_4dims_w_normalized_IOs_no_scoring.txt'
#data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/013120_updated_DMPwts/XYZABG_weights_combined/combined_weights_all_XYZB_4dims_w_all_features_normalized_no_scoring.txt'
data_loc='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/013120_updated_DMPwts/XYZABG_weights_combined/combined_wts_all_XYZB_4dims_w_normalized_IOs_group1.txt'


dataset = DMPWeightData(data_loc,num_dims,num_labels,MNIST=False)

batch_size=39 #156
data_loader=create_data_loader(dataset,batch_size)

#Create instances of generator and discrim classes
#DMP dataset
generator = Generator(num_labels,num_dims,z_size)
discriminator = Discriminator(num_labels,num_dims)

#Train
num_epochs = 5000
n_critic = 5
display_step = 300
loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)

gen_loss=[]
discrim_loss=[]

scoring_wts_all=[]
pivotedChop_wts_all=[]
normalCut_wts_all=[]
movingPivotedChop_wts_all=[]
inHandCut_wts_all=[]
dice_wts_all=[]
angledSliceR_wts_all=[]
angledSliceL_wts_all=[]

for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))    
    for i, (DMPweights) in enumerate(data_loader):        
        real_DMPweights = Variable(DMPweights) #batchsizex1x39        
        generator.train()
        batch_size = real_DMPweights.size(0)                
        real_DMPweights=real_DMPweights.type(torch.FloatTensor)
        
        #Train discriminator pn real and fake data
        d_loss = discriminator_train_step(len(real_DMPweights), discriminator,
                                          generator, d_optimizer, loss,
                                          real_DMPweights,labels,num_labels,z_size)
        
        #Train generator by generating fake data, running it through the discriminator net, and calculating loss
        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, loss,num_labels,z_size)

    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
    gen_loss.append(g_loss)
    discrim_loss.append(d_loss)   
    
    if num_labels!=0:
        z = Variable(torch.randn(num_labels, z_size))     
        labels = Variable(torch.LongTensor(np.arange(num_labels)))
        sample_DMPweights = generator(z, labels).unsqueeze(1).data.cpu()   
    elif num_labels==0:
        z=Variable(torch.randn(batch_size, z_size)) 
        sample_DMPweights = generator(z,labels).unsqueeze(1).data.cpu()   

    #Save weights in an array    
    
    sample_DMPweights_arr=sample_DMPweights.numpy()
    # scoring_wts_all.append(sample_DMPweights_arr[0,:,:])
    # pivotedChop_wts_all.append(sample_DMPweights_arr[1,:,:])
    # normalCut_wts_all.append(sample_DMPweights_arr[2,:,:])
    # movingPivotedChop_wts_all.append(sample_DMPweights_arr[3,:,:])
    # inHandCut_wts_all.append(sample_DMPweights_arr[4,:,:])
    # dice_wts_all.append(sample_DMPweights_arr[5,:,:])
    # angledSliceR_wts_all.append(sample_DMPweights_arr[6,:,:])
    # angledSliceL_wts_all.append(sample_DMPweights_arr[7,:,:])  
    
    #Write results to tensorboard    
    #writer.add_image('cut 4_new',sample_DMPweights[5,:,:].view(1,1,39),global_step=epoch)
    #writer.add_image('cut 3_new',sample_DMPweights[4,:,:].view(1,1,39),global_step=epoch)    
    writer.add_scalar('g_loss',g_loss,global_step=epoch)
    writer.add_scalar('d_loss',d_loss,global_step=epoch)

    #Save gradients to tb
    writer.add_histogram('disc l1.weight.grad', discriminator.model[0].weight.grad,global_step=epoch)
    writer.add_histogram('disc l2.weight.grad', discriminator.model[3].weight.grad,global_step=epoch)
    #writer.add_histogram('disc l3.weight.grad', discriminator.model[6].weight.grad,global_step=epoch)
    #writer.add_histogram('disc outputLayer.weight.grad', discriminator.model[9].weight.grad,global_step=epoch)
    writer.add_histogram('gen l1.weight.grad', generator.model[0].weight.grad,global_step=epoch)
    writer.add_histogram('gen l2.weight.grad', generator.model[2].weight.grad,global_step=epoch)
    #writer.add_histogram('gen l3.weight.grad', generator.model[4].weight.grad,global_step=epoch)
    #writer.add_histogram('gen outputLayer.weight.grad', generator.model[6].weight.grad,global_step=epoch)

    writer.add_scalar('mean disc l1.weight.grad', discriminator.model[0].weight.grad.mean(),global_step=epoch)
    writer.add_scalar('mean disc l2.weight.grad', discriminator.model[3].weight.grad.mean(),global_step=epoch)
    #writer.add_scalar('mean disc l3.weight.grad', discriminator.model[6].weight.grad.mean(),global_step=epoch)
    #writer.add_scalar('mean disc outputLayer.weight.grad', discriminator.model[9].weight.grad.mean(),global_step=epoch)
    writer.add_scalar('mean gen l1.weight.grad', generator.model[0].weight.grad.mean(),global_step=epoch)
    writer.add_scalar('mean gen l2.weight.grad', generator.model[2].weight.grad.mean(),global_step=epoch)
    #writer.add_scalar('mean gen l3.weight.grad', generator.model[4].weight.grad.mean(),global_step=epoch)
    #writer.add_scalar('mean gen outputLayer.weight.grad', generator.model[6].weight.grad.mean(),global_step=epoch)   


plt.figure()
plt.plot(gen_loss,'r')
plt.plot(discrim_loss,'g')
plt.legend(('generator loss','discriminator loss'))
plt.title('generator loss vs discriminator loss')
plt.xlabel('num epochs')
plt.ylabel('loss')
plt.show()


#Save generator net weights to txt file
base_path='/home/test2/Documents/Data/post_processing/weights_sep_traj/120319/GAN/generator_net_weights/'
# np.savetxt(base_path+'scoring.txt',np.reshape(np.array(scoring_wts_all),(num_epochs,num_dims)), delimiter=',') 
# np.savetxt(base_path+'pivChop.txt', np.reshape(np.array(pivotedChop_wts_all),(num_epochs,num_dims)), delimiter=',') 
# np.savetxt(base_path+'normalCut.txt', np.reshape(np.array(normalCut_wts_all),(num_epochs,num_dims)), delimiter=',') 
# np.savetxt(base_path+'movingPivChop.txt', np.reshape(np.array(movingPivotedChop_wts_all),(num_epochs,num_dims)), delimiter=',') 
# np.savetxt(base_path+'inHandCut.txt', np.reshape(np.array(inHandCut_wts_all),(num_epochs,num_dims)), delimiter=',') 
# np.savetxt(base_path+'dice.txt', np.reshape(np.array(dice_wts_all),(num_epochs,num_dims)), delimiter=',') 
# np.savetxt(base_path+'angledR.txt', np.reshape(np.array(angledSliceR_wts_all),(num_epochs,num_dims)), delimiter=',') 
# np.savetxt(base_path+'angledL.txt', np.reshape(np.array(angledSliceL_wts_all),(num_epochs,num_dims)), delimiter=',')
np.savetxt(base_path+'last_iteration_generated_wts.txt', np.reshape(np.array(sample_DMPweights_arr),(batch_size,num_dims)), delimiter=',') 

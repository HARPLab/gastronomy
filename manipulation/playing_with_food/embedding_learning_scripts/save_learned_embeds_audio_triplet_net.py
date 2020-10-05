import numpy as np
import os
import time
import shutil
import argparse
import sys
# # Can use this to set the GPU(s) to use instead of the args
# os.environ["CUDA_DEVICE_ORDER"] = cuda_order
# os.environ["CUDA_VISIBLE_DEVICES"] = devices

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import autograd # only using for anomaly detectio11
sys.path.append(os.path.abspath("../playing_with_food"))

import food_embeddings.networks as networks
import food_embeddings.utils as utils
import food_embeddings.preprocess as preprocess
import food_embeddings.losses as losses

best_loss = np.inf 

def main():
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('-d', '--datapath', type=str, default='./data/processed_data',
                        help='The path to the directory containing the data files, do not include / if using a directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-ts', '--train_test_split', type=float, default=0.8,
                        help='Percentage of data to use for training, the rest is for validation. Must be < 1 and >0')
    # parser.add_argument('-sp', '--embed_save_path', type=str, default=None,
    #                     help='Where to save the embeds to')    
    parser.add_argument('-g', '--gpu_indexes', default=0, type=int,
                        help='The indexes of the VISIBLE GPU to use')   
    parser.add_argument('--thresh_filepath', type = str, default = '/data/audio_labels/all_sound_PCA_feat_thresh_10NNs.npy') 
    parser.add_argument('--audio_labels_filename', type=str, default = '/data/audio_labels/audio_PCA_feature_labels_9.pkl')
    parser.add_argument('--embed_save_filepath', type=str, help='location to save learned embeddings')
    parser.add_argument('--saved_checkpoint', type=str, help='location of saved checkpoint model)
    args = parser.parse_args()

    if args.gpu_indexes is None:
        devices = input("Please type the GPU ids you want to use\n")
        args.gpu_indexes = [int(devices)]
    else:
        if type(args.gpu_indexes) != list:
            args.gpu_indexes = [args.gpu_indexes]

    args.shuffle = False #TODO remember to save the labels of which data samples each embedding refers to later on!!

    # Define the model to use
    base_model = networks.ResNetEmbeddingsNet(out_size=16, #orig 256
                                              fc1_size=512,
                                              fc2_size=256,
                                              sequence_length=1,
                                              drop_prob=0.2)
    model = networks.TripletNet(base_model, multi_args=False) #TODO double check that this is ok to be outisde of the processes

    # load thresholds 
    thresh_filepath= args.thresh_filepath #'/home/test2/Documents/ISER-2020/data/audio_labels/all_sound_PCA_feat_thresh_10NNs.npy' 
    args.threshold = np.load(thresh_filepath)
    print('shape of threshold array', args.threshold.shape)

    # # Define transform to use on input data
    transform = None
    label_transform = None

    # run main worker
    train_embed_out, test_embed_out = get_embeds_from_learned_model(0, model, transform, label_transform, args)

    # save embedds to file 
    save_path = args.embed_save_filepath #'/home/test2/Documents/ISER-2020/playing_with_food/data_analysis/'
    train_embed_filename = save_path + 'train_embed_silhStart'  #'train_embed_run12_emb16_10NNs_cp4' 
    test_embed_filename = save_path + 'test_embed_silhStart' #'test_embed_run12_emb16_10NNs_cp4' 
    save_embed_to_file(train_embed_out, train_embed_filename)
    save_embed_to_file(test_embed_out, test_embed_filename)

    print('finished saving embedds to file')

def get_embeds_from_learned_model(process_id, model, transform, label_transform, args):
    """
    Single process for loading data, defining optimization, training and logging data
    Can be extended for multiple process
    """
    global best_loss
    gpu_id = args.gpu_indexes[process_id]
   
    # send model to GPU
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)
    
    # Load data #TODO these two should be loading the data in the same directory order
    print("Loading data...")
    data = preprocess.get_sample_file(data_path=args.datapath,
                                      filename='images/starting*', #some of the same images are named different
                                      image=True,
                                      img_size=(224, 224)
    )
    ''' Labels for gripper width:
    labels = preprocess.get_sample_file(data_path=args.datapath,
                                        filename='other/gripperWidth_deltaZpush_finalZpush.npy',
                                        key=None,
                                        image=False
    )
    labels = labels[:,0] # gripper width as labels
    '''
    # Labels for audio data - each "label" is a vector of PCA features - labels is array: num_samples x 6 PCs
    audio_labels_filename = args.audio_labels_filename #'/home/test2/Documents/ISER-2020/data/audio_labels/audio_PCA_feature_labels_9.pkl'
    all_audio_labels_dict = utils.get_pickle_file(audio_labels_filename)
    labels = all_audio_labels_dict['all_sound'] 
    #labels, mu, sigma = preprocess.scale_features(labels) # scale labels
    print('shape of audio labels', labels.shape)
            
    train_data, valid_data, train_labels, valid_labels, train_inds, test_inds = \
        preprocess.train_test_split_even_by_veg_type(
        data=[data], 
        labels=[labels],\
        shuffle=args.shuffle
    )
    train_test_inds = np.concatenate((train_inds, test_inds))
    # np.save('/home/test2/Documents/ISER-2020/playing_with_food/data_analysis/train_inds.npy',\
    #     train_inds)
    # np.save('/home/test2/Documents/ISER-2020/playing_with_food/data_analysis/test_inds.npy',\
    #     test_inds)

    #import pdb; pdb.set_trace()
    image_train = np.expand_dims(train_data[0], axis=1)
    image_valid = np.expand_dims(valid_data[0], axis=1)
    train_labels = train_labels[0]
    valid_labels = valid_labels[0]

    # permute train data 
    if torch.is_tensor(image_train):
        image_train = image_train.type(torch.float32)
    else:
        image_train = torch.from_numpy(image_train).type(torch.float32)
    image_train = image_train.permute(0,1,4,2,3) / 255.0 # convert to values between 1-0 if flag is set
    
    # permute test data  
    if torch.is_tensor(image_valid):
        image_valid = image_valid.type(torch.float32)
    else:
        image_valid = torch.from_numpy(image_valid).type(torch.float32)
   
    image_valid = image_valid.permute(0,1,4,2,3) / 255.0 # convert to values between 1-0 if flag is set

    # load model from saved checkpoint
    print('loading model from checkpoint')
    #args.saved_checkpoint  = '/home/test2/Documents/ISER-2020/playing_with_food/checkpts/run12_emb16_10NNs_moreSaving/checkpoint4.pth.tar'
    if args.saved_checkpoint is not None: 
        
        if os.path.isfile(args.saved_checkpoint):
            print("=> loading checkpoint '{}'".format(args.saved_checkpoint))
            if gpu_id is None:
                checkpoint = torch.load(args.saved_checkpoint)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu_id)
                checkpoint = torch.load(args.saved_checkpoint, map_location=loc)
            #args.start_epoch = checkpoint['epoch']
            #best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                  args.saved_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.saved_checkpoint))
    
    ######### test get embeddings from learned model
    train_data_embeds = []
    image_train = image_train.cuda(gpu_id, non_blocking=True)
    with torch.no_grad():
        train_embed_out, train_embed_out, train_embed_out = \
            model(image_train, image_train, image_train)
    # embed_out should be size n_samples x 16 
    
    test_data_embeds = []
    image_valid = image_valid.cuda(gpu_id, non_blocking=True)
    with torch.no_grad():
        test_embed_out, test_embed_out, test_embed_out = \
            model(image_valid, image_valid, image_valid)    
    #import pdb; pdb.set_trace()
    ##############

    train_embed_out = train_embed_out.cpu().numpy()
    test_embed_out = test_embed_out.cpu().numpy()
    return train_embed_out, test_embed_out

def save_embed_to_file(embed_nparray, filename):
    np.save(filename + '.npy', embed_nparray)

if __name__ == "__main__":
    main()

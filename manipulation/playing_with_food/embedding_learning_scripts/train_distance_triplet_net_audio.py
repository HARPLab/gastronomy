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

best_loss = np.inf #0

def main():
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('-d', '--datapath', type=str, default='./data/processed_data',
                        help='The path to the directory containing the data files, do not include / if using a directory')
    parser.add_argument('-sm', '--sim_mat_path', type=str, default='./data/similarity_matrs/gripWidth_finalZ_sim_matrs.pkl',
                        help='The path to the similarity matrix file')
    parser.add_argument('-e', '--epochs', default= 350, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-ts', '--train_test_split', type=float, default=0.8,
                        help='Percentage of data to use for training, the rest is for validation. Must be < 1 and >0')
    # saving and logging info
    parser.add_argument('--log_rate', type=int, default=1, help='How often to log the data, in training iterations(batches)')
    parser.add_argument('-pr', '--print_rate', type=int, default=1, help='How often to print training info, in training iterations(batches)')
    parser.add_argument('-tp', '--tensorboard_path', type=str, default='./runs/run1',
                        help='Path to save the tensorboard log data to')
    parser.add_argument('-sr', '--save_rate', type=int, default=0, help='How often to save the model, in epochs')
    parser.add_argument('-sp', '--save_path', type=str, default=None,
                        help='Where to save the checkpoints to')
    parser.add_argument('-sn', '--save_name', type=str, default=None,
                        help='Name to save the checkpoints as')
    # gpu/processing
    parser.add_argument('-g', '--gpu_indexes', default=0, type=int,
                        help='The indexes of the VISIBLE GPU to use')
    parser.add_argument('-nw', '--num_workers', type=int, default=0,
                        help='number of cpu processes to use for data loading')
    # resume from checkpoint
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='provide string to checkpoint if you want to resume from there')
    parser.add_argument('-se', '--start_epoch', default=0, type=int, 
                        help='epochs to start at')
    # Debugging
    parser.add_argument('-db', '--debug_mode', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='Set to True to run script with debug settings')
                        #TODO need to fix! change train_test_split to evenly split the
                        # data accroding to class. Right now if shuffle is false, the test only gives one label
                        # also should have it make sure that each contains samples of all the classes

    parser.add_argument('--thresh_filepath', type = str, default = '/data/audio_labels/all_sound_PCA_feat_thresh_10NNs.npy') 
    parser.add_argument('--audio_labels_filename', type=str, default = '/data/audio_labels/audio_PCA_feature_labels_9.pkl')
    args = parser.parse_args()

    if args.gpu_indexes is None:
        devices = input("Please type the GPU ids you want to use\n")
        args.gpu_indexes = [int(devices)]
    else:
        if type(args.gpu_indexes) != list:
            args.gpu_indexes = [args.gpu_indexes]

    print("Performing setup...")
    if args.save_path is None:
        args.save_path = args.tensorboard_path

    if args.debug_mode is True:
        autograd.set_detect_anomaly(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        torch.manual_seed(0)
        args.shuffle = False
        args.multi_gpu_mode = False
    else:
        args.shuffle = True #TODO remember to save the labels of which data samples each embedding refers to later on!!

    # Define the model to use
    base_model = networks.ResNetEmbeddingsNet(out_size=16, #orig 256
                                              fc1_size=512,
                                              fc2_size=256,
                                              sequence_length=1,
                                              drop_prob=0.2)
    model = networks.TripletNet(base_model, multi_args=False) #TODO double check that this is ok to be outisde of the processes

    #sim_matrix = utils.get_pickle_file(args.sim_mat_path)['gripWidth']

    #args.threshold = np.mean(sim_matrix).astype(np.float32) #need to change this!
    
    # load audio data thresholds saved in npy file - this is an array of thresholds (size = num samples)
    # Note : thresholds based on 'all_sound' PCA features
    thresh_filepath= args.thresh_filepath #'/home/test2/Documents/ISER-2020/data/audio_labels/all_sound_PCA_feat_thresh_10NNs.npy' #all_sound_PCA_feat_thresh_80NNs.npy
    
    args.threshold = np.load(thresh_filepath)
    print('shape of threshold array', args.threshold.shape)

    # Define transform to use on input data
    transform = None
    label_transform = None

    # run main worker
    run_training_process(0, model, transform, label_transform, args)

def run_training_process(process_id, model, transform, label_transform, args):
    """
    Single process for loading data, defining optimization, training and logging data
    Can be extended for multiple process
    """
    global best_loss
    gpu_id = args.gpu_indexes[process_id]

    # log values on main GPU only
    writer = SummaryWriter(f'{args.tensorboard_path}') # NOTE To use in terminal: $ tensorboard --logdir=<PATH> --host localhost
    # for remote: https://stackoverflow.com/a/42445070
    # to keep ssh from timing out: https://serverfault.com/questions/33283/how-to-setup-ssh-tunnel-to-forward-ssh
    
    # send model to GPU
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = losses.TripletLoss(margin=1) #TODO need to implement adaptive margin

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
   
    image_train = np.expand_dims(train_data[0], axis=1)
    image_valid = np.expand_dims(valid_data[0], axis=1)
    train_labels = train_labels[0]
    valid_labels = valid_labels[0]

    # Instantiate the datasets
    train_dataset = preprocess.RelativeSamplesDataset(
                    data=image_train,
                    labels=train_labels,
                    threshold=args.threshold[train_inds],
                    data_transform=transform,
                    label_transform=label_transform,
                    triplet=True,
                    image=True
    )
    valid_dataset = preprocess.RelativeSamplesDataset(
                    data=image_valid,
                    labels=valid_labels,
                    threshold=args.threshold[test_inds],
                    data_transform=transform,
                    label_transform=label_transform,
                    triplet=True,
                    image=True
    )
    
    # define the batch sampler to use
    if args.debug_mode:
        train_batch_sampler = None
        valid_batch_sampler = None
    else:
        train_batch_sampler = None
        valid_batch_sampler = None
    #NOTE: leaving this here incase we want to implement multi_gpu later on
    train_sampler=None
    valid_sampler=None
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              batch_sampler=train_batch_sampler,
                              shuffle=args.shuffle)
    valid_loader = DataLoader(dataset=valid_dataset,
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=True,
                             sampler=valid_sampler,
                             batch_sampler=valid_batch_sampler,
                             shuffle=False)

    
    # Load from checkpoint if provided, NOTE I dont think this works with multiple GPUs
    if args.resume is not None: 
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if gpu_id is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu_id)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if gpu_id is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_loss = best_loss.to(gpu_id)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                  args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set up clock for timing epochs
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)

    print('Beginning training')
    for epoch in range(args.start_epoch, args.epochs):
        #  record start time
        t_start = time.perf_counter()

        # go through a training validation epoch
        
        _ = train(train_loader=train_loader, 
                      model=model,
                      epoch=epoch,
                      optimizer=optimizer,
                      loss=loss,
                      gpu_id=gpu_id,
                      args=args,
                      writer=writer)
        valid_loss = validate(valid_loader=valid_loader,
                      model=model, 
                      epoch=epoch,
                      loss=loss,
                      gpu_id=gpu_id,
                      args=args,
                      writer=writer)

        t_end = time.perf_counter()
        print(f'Epoch {epoch+1} took {t_end-t_start:0.4f}s\n')
        #import pdb; pdb.set_trace()

        # keep track of best accuracy and save checkpoint
        is_best = valid_loss < best_loss # this is usually accuracy, but there is not acc value
        best_loss = min(valid_loss, best_loss)
        if args.save_rate == 0:
            # save checkpoint with best accruacy by default
            if is_best is True:
                print(f'Saving model at epoch {epoch+1} to {args.save_path}\n')
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.save_path, args.save_name)
            else:
                pass
        elif (epoch+1) % args.save_rate == 0:
            print(f'Saving model at epoch {epoch+1} to {args.save_path}\n')
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_path, args.save_name+str(epoch))
    
    t_final = time.perf_counter()
    print(f"Total runtime was {t_final-t_start:0.4f}")
    print(f'Remeber to delete or move "{args.init_method}"')
    
    print('finished training epochs')
    import pdb; pdb.set_trace()
    # cleanup the process
    writer.close()

def train(train_loader, model, epoch, optimizer, loss, gpu_id, args, writer=None):
    """
    A single epoch training loop

    Outputs:
        avg_loss - (torch.float): gives the average loss of the epoch
    """
    # set to train
    model.train()
    assert model.training == True

    avg_loss = 0
    for i, (anchor, positive, negative, label) in enumerate(train_loader):
        #import pdb; pdb.set_trace()
        # record start time
        t0 = time.perf_counter()

        # send values to gpu, NOTE: idk if setting non_blocking to true does anything for inputs, since models depends on it
        anchor = anchor.cuda(gpu_id, non_blocking=True)
        positive = positive.cuda(gpu_id, non_blocking=True)
        negative = negative.cuda(gpu_id, non_blocking=True)
        label = label.cuda(gpu_id, non_blocking=True) #NOTE: I don't think we are using this
        
        # zero the optimizer
        optimizer.zero_grad()

        # pass through network
        anchor_out, pos_out, neg_out = model(anchor, positive, negative)

        train_loss = loss(anchor_out, pos_out, neg_out, size_average=True)

        avg_loss += train_loss.item()

        # backprop
        train_loss.backward()
        optimizer.step()

        # log info
        if (i) % args.log_rate == 0:
            writer.add_scalar('Average_Loss/Training',
                            avg_loss / (i + 1),
                            epoch * len(train_loader) + i + 1)
        t1 = time.perf_counter()
        if (i) % args.print_rate == 0:
            print('Currently on training epoch {} and batch {} of {}'.format(
                epoch+1, i+1, len(train_loader)))
            print('Average loss = {:0.4f}, Loop Time = {:0.4f}s\n'.format(
                avg_loss / (i + 1), (t1 - t0)))

    avg_loss /= len(train_loader)

    return avg_loss

def validate(valid_loader, model, epoch, loss, gpu_id, args, writer=None):
    """
    A single epoch of the validation loop

    Outputs:
        avg_loss - (torch.float): gives the average loss of the epoch
    """
    # set to eval
    model.eval()
    assert model.training == False

    avg_loss = 0
    with torch.no_grad():
        for i, (anchor, positive, negative, label) in enumerate(valid_loader):
            #import pdb; pdb.set_trace()
            # record start time
            t0 = time.perf_counter()

            # send values to gpu, NOTE: idk if setting non_blocking to true does anything for inputs, since models depends on it
            anchor = anchor.cuda(gpu_id, non_blocking=True)
            positive = positive.cuda(gpu_id, non_blocking=True)
            negative = negative.cuda(gpu_id, non_blocking=True)
            label = label.cuda(gpu_id, non_blocking=True) #NOTE: I don't think we are using this
            
            # pass through network
            anchor_out, pos_out, neg_out = model(anchor, positive, negative)

            valid_loss = loss(anchor_out, pos_out, neg_out, size_average=True)

            avg_loss += valid_loss.item()

            # log info
            writer.add_scalar('Average_Loss/Validation', 
                                avg_loss / (i + 1), epoch + 1)
            t1 = time.perf_counter()
            if (i) % args.print_rate == 0:
                print('Currently on validation epoch {} and iteration {} of {}'.format(
                    epoch+1, i+1, len(valid_loader)))
                print('Avergae loss = {:0.4f}, Loop Time = {:0.4f}s\n'.format(
                    avg_loss / (i + 1), (t1 - t0)))
    
    avg_loss /= len(valid_loader)

    return avg_loss


if __name__ == "__main__":
    main()

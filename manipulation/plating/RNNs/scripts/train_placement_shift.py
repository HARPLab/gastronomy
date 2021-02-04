import numpy as np
import os
import sys
import time
import tqdm
import shutil
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

# # Set the GPU(s) to use, need to set the GPUs before you import torch
# os.environ["CUDA_DEVICE_ORDER"] = cuda_order
# os.environ["CUDA_VISIBLE_DEVICES"] = devices

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch import autograd # only using for anomaly detection

import autolab_core
import perception
import rnns.losses as losses
import rnns.accuracy as accuracy
import rnns.image_utils as image_utils
from rnns.utils import plot_grad_flow
from rnns.preprocess import PlacementShiftFileDataset, train_test_split_filenames, FastDataLoader
from rnns.networks import PlacementShiftNet, PlacementShiftDistNet
from rnns.initialization import general_uniform, general_normal

#NOTE:  This script assumes you are training on a GPU
# Some possible things to try, send each depth image and the  ee_pose
# through a network seperatly and then concatenate (idk if ee pose will benefit though)
# consider using other than the diff between obj center and end effector as label(something with the whole bbox or IOU)

#TODO make this work with multiple gpus, and put in save checkpoints
    # - copy and check image net ex
    # - change all the cuda/device calls

#TODO might want to try different intializations
"""
Helpful links:
    - tips on when to push data to GPU: https://discuss.pytorch.org/t/pin-memory-vs-sending-direct-to-gpu-from-dataset/33891
        - use cpu to load data and gpu to train, 
    - explain how to use multiple GPUs: https://discuss.pytorch.org/t/multi-gpu-dataloader-and-multi-gpu-batch/66310/4
    - For using GPU vs CPU, usually best to use CPU tensors to get the data, and GPU for training.
      So you would use CPU for the dataloader w/ your dataset, and then send you model/network to
      GPU and in the training loop send the training data gathered from dataloader to the GPU
        - main info: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        - extra info: https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
        - extra info: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        - more examples: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
        - extensive example: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    - Tips on handling data loading: https://stackoverflow.com/questions/59836100/how-to-store-and-load-training-data-comprised-50-millions-25x25-numpy-arrays-whi
    - Notes on normalizing data:
        - how to normalize: https://docs.google.com/document/d/1x0A1nUz1WWtMCZb5oVzF0SVMY7a_58KQulqQVT8LaVA/edit#heading=h.bt7jdccuynnx
        - forum discussion: https://www.researchgate.net/post/Which_data_normalization_method_should_be_used_in_this_artificial_neural_network
        - normal dist and denormalizing it: http://www.nkd-group.com/sta308/notes/normaldistribution.pdf
        - sklearn power transformer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html

Example:
CUDA_VISIBLE_DEVICES=0 python scripts/train_placement_shift.py -d ../carbongym-utils/data/run_no_max_no_1st_l2/final_no_max_no_1st_l2 -pr 5 -lg 5 -bs 512 -nw 4 -p 14320 -lr 1e-5 -sr 500 -od False -ob True -om False -e 10000 -op RMSProp -mb 100000 -c 500 -ci 0.005 -cm 0.03 -tp runs/run66
CUDA_VISIBLE_DEVICES=7 python scripts/train_placement_shift.py -d ../carbongym-utils/data/p1_less_rand/final_no_max_no_1st_l2 -pr 10 -lg 10 -le False -bs 512 -nw 7 -p 12369 -lr 1e-5 -sr 500 -od False -ob True -om False -e 10000 -op RMSProp -c 500 -ci 0.005 -cm 0.05 -tp runs/run84

Hyperparameters:
RMSProp
-lr 1e-6
"""
def main():
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('-d', '--datapath', type=str, default='../Documents/carbongym-utils/data/run',
                        help='The path to data file or the direcotry containing the data files, do not include / if using a directory')
    parser.add_argument('-i', '--info_file_name', type=str, default='info.npy',
                        help='Name of the file containing information on the dataset')
    parser.add_argument('-e', '--epochs', default=1000, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('-op', '--optimizer', default='RMSProp', type=str,
                        help="Can be 'RMSProp', 'Adam', or 'SGD'")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-m', '--momentum', type=float, default=0.0)
    parser.add_argument('-w', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-dp', '--drop_prob', type=float, default=0.8,
                        help='The dropout probability to use. Should be between 0.0 and 1.0')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-mb', '--mini_batch_size', default=None, type=float,
                        help='Size of the subset of the dataset to use for training instead of the entire dataset. ' +\
                             'Use a value > 1 to specify a specific mini batch size (rounds value to int) or a float ' +\
                             'from 0-1 to use a percentage. Default is None and will use the entire dataset. ' +\
                             'The subset of data used will always be shuffled. This DOES NOT work with multiple GPUs')
    parser.add_argument('-ts', '--train_test_split', type=float, default=0.8,
                        help='Percentage of data to use for training, the rest is for validation. Must be < 1 and >0')
    parser.add_argument('-wx', '--x_dim_weighting', type=float, default=1.0, 
                        help='Amount to weight the x dimension predictions for the loss')
    parser.add_argument('-wz', '--z_dim_weighting', type=float, default=1.0, 
                        help='Amount to weight the z dimension predictions for the loss')
    parser.add_argument('-in', '--initialization', default=None, type=str,
                        help='None for default intialization or "general_uniform" or "general_normal"')
    parser.add_argument('-o', '--mean_offset', default=0.0, type=float,
                        help='Value to offset the mean of the predictions, this value is subtracted')
    parser.add_argument('-ep', '--epsilon', default=0.0, type=float,
                        help='Value to add to the values in network to avoid zero division')
    parser.add_argument('-od', '--output_dist', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='set to true to have the network output a distribution, this overides classifiers and is the default')
    parser.add_argument('-ob', '--output_binary', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='set to true to have the network output boolean predictions')
    parser.add_argument('-om', '--output_multi', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='''set to true to have the network output multi-class predictions. The network will classify
                            predictions based on the distance interval they land within.''')
    parser.add_argument('-c', '--curriculum', default=None, type=int,
                        help='How many epochs to wait until adding the next set of data in the curriculum,' \
                             'None turns off curriculum learning. Too large of a value could lead to not all of the data being used')
    #TODO should let this be a list too, so you can set how many epochs per interval (just assert than len(list)==num_class intervals)
    parser.add_argument('-ci', '--curriculum_class_interval', type=float, default=0.001,
                        help='The distance interval to use between classes for the curriculum')
    parser.add_argument('-cm', '--curriculum_max_class', type=float, default=0.01,
                        help='The max interval value, all samples that have a distance value higher than this are grouped into one class')
    parser.add_argument('-cn', '--curriculum_min_class', type=float, default=0.0,
                        help='The min interval value, all samples that have a distance value smaller than this are grouped into one class')
    # parser.add_argument() # add a minimum
    parser.add_argument('-rm', '--remove_classes', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='For binary curriculum classification, set to true to remove values from dataset when it is updated')
    # resume from checkpoint
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='provide string to checkpoint if you want to resume from there')
    parser.add_argument('-se', '--start_epoch', default=0, type=int, 
                        help='epochs to start at')
    parser.add_argument('-cc', '--curriculum_counter_value', type=int, default=1,
                        help='The curriculum interval to start from when resuming from checkpoint, if checkpoint has a value it will be used instead')
    # saving and logging info
    parser.add_argument('-lg', '--log_rate', type=int, default=1,
                        help='How often to log the data, in training iterations(batches)')
    parser.add_argument('-pr', '--print_rate', type=int, default=1,
                        help='How often to print training info, in training iterations(batches)')
    parser.add_argument('-tp', '--tensorboard_path', type=str, default='/home/stevenl3/Attention/runs/placement/placement1',
                        help='Path to save the tensorboard log data to')
    parser.add_argument('-sr', '--save_rate', type=int, default=100,
                        help='How often to save the model, in epochs')
    parser.add_argument('-sp', '--save_path', type=str, default=None,
                        help='Where to save the checkpoints to')
    parser.add_argument('-sn', '--save_name', type=str, default=None,
                        help='Name to save the checkpoints as')
    parser.add_argument('-le', '--log_extra_info', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='set to log confusion matrices and images of the predictions vs ground truth to tensorboard')
    # multi process and gpu
    parser.add_argument('-mp', '--multi_gpu_mode', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='Set to True to run script with multi-process, NOTE: This has not been tested thouroghly')
    parser.add_argument('-nw', '--num_workers', type=int, default=0,
                        help='The number of cpu processes to use for the data loader')
    parser.add_argument('-g', '--gpu_indexes', nargs='+', default=None, type=int,
                        help='The indexes of the VISIBLE GPUs to use, these will be used in the order provided, so first provided is main')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', 
                        help='number of nodes(devices) to use')
    parser.add_argument('-ng', '--n_gpus', default=1, type=int,
                        help='number of gpus per node, this needs to be the same value among all the nodes')
    parser.add_argument('-nr', '--node_ranking', default=0, type=int,
                        help='ranking of this node among all the nodes')
    # initialization
    parser.add_argument('-b', '--backend', default='nccl', type=str,
                        help='Backend to use for processes')
    parser.add_argument('-im', '--init_method', default="env://",
                        type=str, help="Initialization method or string to shared file to use between processes, it shouldn't exist, but the directory should")
                        # default="file:///home/stevenl3/temp/sharedfile", https://discuss.pytorch.org/t/shared-file-system-is-a-file/51151
    parser.add_argument('-a', '--address', default='localhost', type=str)
    parser.add_argument('-p', '--port', default='12355', type=str)
    # Debugging
    parser.add_argument('-db', '--debug_mode', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='Set to True to run script with debug settings')# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/36031646
    parser.add_argument('-gf', '--plot_grad_flow', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        help='Set to True to output gradient flow graphs')
    args = parser.parse_args()

    # Set GPU indexe(s) to use
    if args.gpu_indexes is None and args.multi_gpu_mode:
        devices = input("Please type the GPU ids you want to use, seperated by a comma, e.g. 1,3,4\n")
        args.gpu_indexes = [int(i) for i in devices.split(',')]
    elif not args.multi_gpu_mode:
        args.gpu_indexes = [0]
    else:
        if type(args.gpu_indexes) != list:
            args.gpu_indexes = [args.gpu_indexes]

    if args.save_path is None:
        args.save_path = args.tensorboard_path

    # check the formatting of the mini_batch_size argument
    if args.mini_batch_size is not None:
        if args.mini_batch_size >= 1:
            args.mini_batch_size = int(args.mini_batch_size)
        else:
            assert args.mini_batch_size > 0

    setup_env(addrs=args.address, port=args.port) 
    args.world_size = args.n_gpus * args.nodes
    # How to weight the dimensions during training #TODO should change this, no need to combine them
    args.xyz_weighting = [args.x_dim_weighting, args.z_dim_weighting]

    if args.debug_mode is True:
        autograd.set_detect_anomaly(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        torch.manual_seed(0)
        args.shuffle = True #False
        args.multi_gpu_mode = False
        args.num_workers = 0
        args.rand_state = 1
    else:
        args.shuffle = True
        args.rand_state = None

    # Define model
    if args.output_dist:
        assert not args.output_multi and not args.output_binary
        model = PlacementShiftDistNet(in_channels=2,
                                      num_out_dims=2,
                                      fc1_size=1024,
                                      fc1_input2_size=64,
                                      fc2_size=512,
                                      input2_size=1,
                                      drop_prob=args.drop_prob,
        )
    elif args.output_multi:
        assert not args.output_binary
        num = args.curriculum_max_class - args.curriculum_min_class
        assert num > 0
        num_classes = int(np.ceil(num/args.curriculum_class_interval)) + 1
        model = PlacementShiftNet(in_channels=2,
                                    num_out_dims=num_classes,
                                    fc1_size=1024,
                                    fc1_input2_size=64,
                                    fc2_size=512,
                                    input2_size=1,
                                    drop_prob=args.drop_prob,
        )
    elif args.output_binary:
        model = PlacementShiftNet(in_channels=2,
                                    num_out_dims=1,
                                    fc1_size=1024,
                                    fc1_input2_size=64,
                                    fc2_size=512,
                                    input2_size=1,
                                    drop_prob=args.drop_prob,
        )
    else:
        raise ValueError('Pick an output type')
    # Define transform to use on the data
    transform = None

    if args.multi_gpu_mode:
        print("Spawning processes...")
        mp.spawn(run_train_processes, nprocs=args.n_gpus, args=(model, transform, args))
    else:
        run_train_processes(0, model, transform, args)


def run_train_processes(process_id, model, transform, args):
    """
    Function to be passed to each process that handles training and validation 

    Inputs:
        gpu_id(int): ID of the GPU to run this process on (on current node)
        model(nn.Module): the neural network model to use
        transform: torchvision transform to use on data set
    """
    gpu_id = args.gpu_indexes[process_id]
    rank = (args.node_ranking * args.n_gpus) + gpu_id #TODO should this be gpu id or process id?
    
    # log values on main GPU only
    if rank == 0:
        writer = SummaryWriter(f'{args.tensorboard_path}') # NOTE To use in terminal: $ tensorboard --logdir=<PATH> --host localhost
        # for remote: https://stackoverflow.com/a/42445070
    else:
        writer = None
    
    # initialize this single process group
    dist.init_process_group(backend=args.backend,
                            init_method=args.init_method,
                            world_size=args.world_size,
                            rank=rank
    )

    if args.curriculum is not None:
        assert args.curriculum != 1 and args.curriculum != 0
        use_curriculum = True
    else:
        use_curriculum = False

    # normalization values
    info = np.load(f'{args.datapath}/{args.info_file_name}', allow_pickle=True)[()]
    args.num_samples = info['num_samples']
    args.labels_mu = info['labels_mu']
    args.labels_std = info['labels_std']
    args.labels_mu = np.delete(args.labels_mu, 1) # remove the y values
    args.labels_std = np.delete(args.labels_std, 1)
    if args.labels_mu.shape[0] == 3: # if there was an extra dim for the binary labels
        args.labels_mu = np.delete(args.labels_mu, 2)
        args.labels_std = np.delete(args.labels_std, 2)
    assert args.labels_mu.shape[0] == 2
    args.max_rot = info['max_ee_rot']
    args.max_image_depth = info['max_image_depth']
    if args.max_image_depth is None:
        args.max_image_depth = 0.17
    if args.max_rot is None:
        args.max_rot = 0.7071
    # Values for the cameras/images
    args.intrinsics = info['intrinsics']
    args.extrinsics = info['extrinsics']
    args.image_mu = info['image_mu']
    args.image_std = info['image_std']
    args.initial_obj_pose = info['initial_obj_pose']

    modes = info['labels_mode']
    modes_count = info['labels_mode_count']
    # import ipdb; ipdb.set_trace()
    # info.close()

    if args.mini_batch_size is not None:
        if type(args.mini_batch_size) == float:
            args.mini_batch_size = int(args.mini_batch_size * args.num_samples)
        assert type(args.mini_batch_size) == int
        assert args.mini_batch_size <= args.num_samples * args.train_test_split

    # for normalizing images channel-wise, insterad of pixel-wise 
    if args.image_mu.ndim == 3:
        args.norm_image_mu = np.mean(args.image_mu, axis=(0,1))
        args.norm_image_std = np.std(args.image_std, axis=(0,1))
    else:
        args.norm_image_mu = None
        args.norm_image_std = None
    
    if torch.is_tensor(args.initial_obj_pose):
        args.initial_obj_pose = args.initial_obj_pose.numpy() # this is just to check that its the right format
    args.initial_obj_pose[1] -= 0.025 #this is the end_effector, lower the values towards table

    # send model to GPU
    model.cuda(gpu_id)
    torch.cuda.set_device(gpu_id)

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay
        )
    elif args.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"{args.optimizer} isn't defined")
    
    # wrap model in data parallelization module, assigning one process per GPU
    model = DDP(model, device_ids=[rank])
    if args.initialization is not None:
        if args.initialization == 'general_uniform':
            model.apply(general_uniform)
        elif args.initialization == 'general_normal':
            model.apply(general_normal)

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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                curriculum_counter = checkpoint['curriculum_counter_state']
            except:
                curriculum_counter = args.curriculum_counter_value
            print("=> loaded checkpoint '{}' (epoch {})".format(
                  args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError
    else:
        curriculum_counter = 1

    print('Getting dataset ready...') 
    # split the data into train and valid
    train_test_data = train_test_split_filenames(
                      root_dir=args.datapath, 
                      file_prefixes=['image_data','ee_data', 'label'],
                      percentage=args.train_test_split,
                      shuffle=args.shuffle,
                      file_suffix='.npy',
                      sss=use_curriculum,
                      class_interval=args.curriculum_class_interval,
                      max_class_value=args.curriculum_max_class,
                      min_class_value=args.curriculum_min_class,
                      random_state=args.rand_state
    )
    image_train = train_test_data['image_data']['train']
    image_test = train_test_data['image_data']['test']
    ee_train = train_test_data['ee_data']['train']
    ee_test = train_test_data['ee_data']['test']
    label_train = train_test_data['label']['train']
    label_test = train_test_data['label']['test']

    # create data loaders
    train_data = PlacementShiftFileDataset(image_files=image_train,
                                        ee_files=ee_train,
                                        label_files=label_train,
                                        root_dir=args.datapath,
                                        labels_mu=args.labels_mu,
                                        labels_std=args.labels_std,
                                        max_rot=args.max_rot,
                                        image_mu=args.image_mu,
                                        image_std=args.image_std,
                                        image_channel_mu=args.norm_image_mu,
                                        image_channel_std=args.norm_image_std,
                                        transform=transform,
                                        multi_classifier=args.output_multi,
                                        binary_classifier=args.output_binary,
                                        use_curriculum=use_curriculum,
                                        class_interval=args.curriculum_class_interval,
                                        max_class_value=args.curriculum_max_class,
                                        min_class_value=args.curriculum_min_class
    )
    test_data = PlacementShiftFileDataset(image_files=image_test,
                                        ee_files=ee_test,
                                        label_files=label_test,
                                        root_dir=args.datapath,
                                        labels_mu=args.labels_mu,
                                        labels_std=args.labels_std,
                                        max_rot=args.max_rot,
                                        image_mu=args.image_mu,
                                        image_std=args.image_std,
                                        image_channel_mu=args.norm_image_mu,
                                        image_channel_std=args.norm_image_std,
                                        transform=transform,
                                        multi_classifier=args.output_multi,
                                        binary_classifier=args.output_binary,
                                        use_curriculum= use_curriculum,
                                        class_interval=args.curriculum_class_interval,
                                        max_class_value=args.curriculum_max_class,
                                        min_class_value=args.curriculum_min_class
    )

    # update the data set if resuming from a checkpoint
    if args.curriculum is not None:
        args.curriculum_classes = train_data.class_intervals
        # counter starts at 1 and skips last class since those 2 are always included
        if curriculum_counter != 1 and curriculum_counter < (len(train_data.class_labels) - 1):
            if args.output_multi:
                #NOTE: This hasn't been tested yet
                train_data.update_dataset(curriculum_counter)
                test_data.update_dataset(curriculum_counter)
            else:
                train_data.update_dataset_binary(curriculum_counter)
                test_data.update_dataset_binary(curriculum_counter)
        else:
            pass

    # set up the dataloaders
    if args.multi_gpu_mode is True:
        train_sampler = DistributedSampler(dataset=train_data,
                                           num_replicas=args.world_size,
                                           rank=rank,
                                           shuffle=args.shuffle
        )
        test_sampler = DistributedSampler(dataset=test_data,
                                          num_replicas=args.world_size,
                                          rank=rank,
                                          shuffle=args.shuffle
        )
        shuffle = False
    elif args.mini_batch_size is not None:
        # NOTE: This ALWAYS shuffles the subset of the dataset that will be used
        train_sampler = RandomSampler(data_source=train_data,
                                      replacement=True,
                                      num_samples=args.mini_batch_size,
        ) 
        #TODO might want to use WeightedRandomSample instead
        test_sampler = None
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = args.shuffle
    train_loader = FastDataLoader(dataset=train_data,
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  sampler=train_sampler,
                                  shuffle=shuffle
    )
    test_loader = FastDataLoader(dataset=test_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=True,
                                 sampler=test_sampler,
                                 shuffle=shuffle #Don't really need to shuffle here
    )

    # Define the loss functions
    if args.output_dist:
        # for regression, predict distance directly
        loss = losses.NLL_Loss
    elif args.output_multi:
        assert args.output_binary != True
        # if use_curriculum:
        n_samples = train_data.class_weights
        weights = [1 - (x / sum(n_samples)) for x in n_samples]
        weights = torch.FloatTensor(weights).cuda(gpu_id, non_blocking=True)
        loss = nn.CrossEntropyLoss(weight=weights)
        # NOTE: might want to get the num samples per class from the train_test split since it is over thet entire dataset
        # how to use crossentropy weight: https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731
        #https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514
    elif args.output_binary: 
        # for binary classification
        try:
            pos_weight = 1 - (sum(train_data.near_labels) / len(train_data.near_labels))
        except KeyError:
            print(f"Couldn't find the positive samples weight for the BCE loss")
            pos_weight = None
        #NOTE: a pos_weight < 1 helps with precision, > 1 helps with recall
        loss = losses.WeightedBCEWithLogitsLoss(PosWeightIsDynamic=True)
    else:
        raise ValueError('Pick an output mode')


    # make text file with network hyperparameter info
    i = 1
    model_desc = f'{args.tensorboard_path}/network_info.txt'
    # make sure not to overwrite old files
    while os.path.isfile(model_desc):
        i += 1
        model_desc = f'{args.tensorboard_path}/network_info_trial{i}.txt'

    f = open(model_desc, 'w')
    f.write(f'Model architecture info: \
            \n epochs: {args.epochs}, \
            \n learning rate: {args.learning_rate}, \
            \n momentum: {args.momentum}, \
            \n dropout: {args.drop_prob}, \
            \n batch size: {args.batch_size}, \
            \n mini-batch size: {args.mini_batch_size}, \
            \n train_test_split: {args.train_test_split}, \
            \n loss: {loss}, \
            \n loss weighting: {args.xyz_weighting}, \
            \n initialization: {args.initialization}, \
            \n optimizer: {optimizer}, \
            \n model: {model}, \
            \n run: {args.tensorboard_path}, \
            \n dataset_path: {args.datapath}, \
            \n curriculum: {args.curriculum}, \
            \n curriculum class interval: {args.curriculum_class_interval}, \
            \n curriculum max class value: {args.curriculum_max_class}, \
            \n curriculum min class value: {args.curriculum_min_class}'
    )
            # \n train_data_info: {train_data.get_data_info()}, \
    f.close()


    # set up clock for timing epochs
    t_begin = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    t_begin = time.perf_counter()

    print('Beginning training')
    for epoch in range(args.start_epoch, args.epochs):
        #  record start time
        t_start = time.perf_counter()

        if args.multi_gpu_mode is True:
            # increment the sampler 
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        # go through a training validation epoch
        print(f'Logging data to {args.tensorboard_path}')
        _, __ = train(train_loader=train_loader, 
                      model=model,
                      epoch=epoch,
                      optimizer=optimizer,
                      loss=loss,
                      gpu_id=gpu_id,
                      args=args,
                      writer=writer,
                      curriculum_counter=curriculum_counter
        )
        _, __ = validate(valid_loader=test_loader,
                      model=model, 
                      epoch=epoch,
                      loss=loss,
                      gpu_id=gpu_id,
                      args=args,
                      writer=writer,
                      curriculum_counter=curriculum_counter
        )

        t_end = time.perf_counter()
        print(f'Epoch {epoch+1} took {t_end-t_start:0.4f}s\n')
        
        # save checkpoints
        if rank % args.n_gpus == 0:  # save on each node, not each process
            if (epoch+1) % args.save_rate == 0:
                print(f'Saving model at epoch {epoch+1} to {args.save_path}\n')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'curriculum_counter_state' : curriculum_counter
                    # 'data_info': train_data.get_data_info() # train and test use same info file
                    }, args.save_path, args.save_name
                )

        if args.curriculum is not None and (epoch+1) % args.curriculum == 0:
            # counter starts at 1 and skips last class since those 2 are always included
            if curriculum_counter < (len(train_data.class_labels) - 1):
                if args.output_multi:
                    train_data.update_dataset(curriculum_counter)
                    test_data.update_dataset(curriculum_counter)
                else:
                    train_data.update_dataset_binary(curriculum_counter)
                    test_data.update_dataset_binary(curriculum_counter)
                    if args.remove_classes:
                        #TODO there is probably a cleaner way to do this
                        train_data.return_indices = True
                        test_data.return_indices = True
                        temp_train_loader =  FastDataLoader(dataset=train_data,
                                            batch_size=args.batch_size, 
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            drop_last=False,
                                            sampler=None,
                                            shuffle=False
                        )
                        temp_test_loader = FastDataLoader(dataset=test_data,
                                                    batch_size=args.batch_size, 
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    sampler=None,
                                                    shuffle=False
                        )
                        rm_train_indexes = eval_model(model, temp_train_loader, gpu_id, args)
                        rm_test_indexes = eval_model(model, temp_test_loader, gpu_id, args)
                        rm_train_indexes = rm_train_indexes.cpu().numpy().astype(int)
                        rm_test_indexes = rm_test_indexes.cpu().numpy().astype(int)
                        train_data.remove_dataset_class(class_key=None, indexes=rm_train_indexes)
                        test_data.remove_dataset_class(class_key=None, indexes=rm_test_indexes)
                        train_data.return_indices = False
                        test_data.return_indices = False
                train_loader =  FastDataLoader(dataset=train_data,
                                            batch_size=args.batch_size, 
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            drop_last=True,
                                            sampler=train_sampler,
                                            shuffle=shuffle
                )
                test_loader = FastDataLoader(dataset=test_data,
                                            batch_size=args.batch_size, 
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            drop_last=True,
                                            sampler=test_sampler,
                                            shuffle=shuffle
                )
                curriculum_counter += 1
            else:
                pass

    t_final = time.perf_counter()
    print(f"Total runtime was {t_final-t_begin:0.4f}")
    print(f'Remeber to delete or move "{args.init_method}"')

    # cleanup the process
    writer.close()
    cleanup()

def train(train_loader, model, epoch, optimizer, loss, gpu_id, args, writer=None, curriculum_counter=None):
    """
    A single epoch training loop

    Inputs:
        train_loader - a torch data loader
        model - neural network to pass input through (assuming a nn.Module)
        epoch - (int): current epoch (for logging purposes)
        optimizer - optimizer object to use for training
        loss - loss object to use for training
        gpu_id - ID (index) of the GPU to use
        args - args from argsparse 
        writer - tensorboard writer to write to
    Outputs:
        avg_loss - (torch.float): gives the average loss of the epoch
        avg_acc - (torch.float): gives the average acc of the epoch
    """
    # set to train
    model.train()
    assert model.training == True

    # convert dimension weighting for loss to floats
    xyz_weight = torch.FloatTensor(args.xyz_weighting).cuda(gpu_id, non_blocking=True)
    labels_mu = torch.FloatTensor(args.labels_mu).cuda(gpu_id, non_blocking=True)
    labels_std = torch.FloatTensor(args.labels_std).cuda(gpu_id, non_blocking=True)
    mean_offset = torch.FloatTensor(np.array([args.mean_offset])).cuda(gpu_id, non_blocking=True)
    epsilon = torch.FloatTensor(np.array([args.epsilon])).cuda(gpu_id, non_blocking=True)

    avg_loss = 0
    avg_acc = 0
    avg_f1 = 0
    avg_bal_acc = 0
    avg_prec = 0 
    avg_recall = 0
    avg_fp_error = 0
    avg_x_std = 0
    avg_z_std = 0
    for i, sample in enumerate(train_loader):
        # record start time
        t0 = time.perf_counter()
        
        # send values to gpu, NOTE: idk if setting non_blocking to true does anything for inputs, since models depends on it
        x1 = sample[0].cuda(gpu_id, non_blocking=True).type(torch.float32)
        x2 = sample[1].cuda(gpu_id, non_blocking=True).type(torch.float32) #TODO i think you need to add the pose(spec. the height) since the depth is at a fixed location there isn't anything encoding the placement height
        if args.output_dist:
            y = sample[2].cuda(gpu_id, non_blocking=True).type(torch.float32)
        elif args.output_multi:
            y = sample[3].cuda(gpu_id, non_blocking=True).type(torch.long)
        else:
            shift = sample[2].cuda(gpu_id, non_blocking=True).type(torch.float32)
            y = sample[3].cuda(gpu_id, non_blocking=True).type(torch.float32)
        # zero the optimizer
        optimizer.zero_grad()

        # send value through network 
        output = model(input1=x1, input2=x2, epsilon=epsilon, mean_offset=mean_offset)
        # calculate loss and accuracy
        if args.output_dist:
            loss1 = loss(output[0], y[:,0])*xyz_weight[0]
            loss2 = loss(output[1], y[:,1])*xyz_weight[1]
            train_loss = (loss1+loss2)/2
            train_acc = mse_accuracy(output, y, labels_mu, labels_std)
        elif args.output_multi:
            train_loss = loss(output, y)
            train_acc = accuracy.multi_class_acc(output, y, softmax=True)
            #TODO haven't tested this yet
            train_bal_acc, train_precision, train_recall, train_f1, _ = accuracy.get_balanced_metrics(output, y, softmax=True)
        else:
            #binary classifier
            output = output.view(-1)
            y = y.view(-1)
            train_loss = loss(output, y)
            train_acc = accuracy.binary_acc(output, y, sigmoid=True)
            train_bal_acc, train_precision, train_recall, train_f1, _ = accuracy.get_balanced_metrics(output, y, sigmoid=True)
            false_pos_error = accuracy.false_pos_error(output, y, shift, args.curriculum_classes[curriculum_counter-1],
                labels_mu, labels_std, sigmoid=True)
        
        avg_loss += train_loss.item()
        avg_acc += train_acc.item()
        if args.output_dist:
            avg_x_std += output[0].stddev.mean().item()
            avg_z_std += output[1].stddev.mean().item()
        elif args.output_binary or args.output_multi:
            avg_f1 += train_f1.item()
            avg_bal_acc += train_bal_acc.item()
            avg_prec += train_precision.item()
            avg_recall += train_recall.item()
            avg_fp_error += false_pos_error

        # backprop
        train_loss.backward()
        optimizer.step()

        # log info on main process
        if gpu_id == args.gpu_indexes[0]:
            if (i) % args.log_rate == 0:
                writer.add_scalar('Average_Loss/Training',
                                avg_loss / (i + 1),
                                epoch * len(train_loader) + i + 1)
                writer.add_scalar('Average_Accuracy/Training',
                                avg_acc / (i + 1),
                                epoch * len(train_loader) + i + 1)
                writer.add_scalar('Average_X_Standard_Deviation/Training',
                                avg_x_std / (i + 1),
                                epoch * len(train_loader) + i + 1)
                writer.add_scalar('Average_Z_Standard_Deviation/Training',
                                avg_z_std / (i + 1),
                                epoch * len(train_loader) + i + 1)
                if args.output_binary or args.output_multi:
                    writer.add_scalar('Average_Balanced_F1_Score/Training',
                                    avg_f1 / (i + 1),
                                    epoch * len(train_loader) + i + 1)
                    writer.add_scalar('Average_Balanced_Accuracy/Training',
                                    avg_bal_acc / (i + 1),
                                    epoch * len(train_loader) + i + 1)
                    writer.add_scalar('Average_Balanced_Precision/Training',
                                    avg_prec / (i + 1),
                                    epoch * len(train_loader) + i + 1)
                    writer.add_scalar('Average_Balanced_Recall/Training',
                                    avg_recall / (i + 1),
                                    epoch * len(train_loader) + i + 1)
                    writer.add_scalar('Average_False_Positive_Error(m)/Training',
                                    avg_fp_error / (i + 1),
                                    epoch * len(train_loader) + i + 1)
                if args.plot_grad_flow:
                    writer.add_figure('Gradient_Flow/Training',
                                    plot_grad_flow(model.named_parameters()),
                                    epoch * len(train_loader) + i + 1)
        t1 = time.perf_counter()
        if (i) % args.print_rate == 0:
            print('Currently on training epoch {} and batch {} of {}'.format(
                epoch+1, i+1, len(train_loader)))
            if args.output_dist:
                print('Average loss = {:0.4f}, Average accuracy = {:0.4f}, Loop Time = {:0.4f}s'.format(
                    avg_loss / (i + 1), avg_acc / (1 + i), (t1 - t0)))
                print('Average x std = {:0.4f}, Average z std = {:0.4f}\n'.format(
                    avg_x_std / (i + 1), avg_z_std / (i + 1), (t1 - t0)))
            else:
                print('Average loss = {:0.4f}, Average accuracy = {:0.4f}, Loop Time = {:0.4f}s'.format(
                    avg_loss / (i + 1), avg_acc / (1 + i), (t1 - t0)))
                print(('Average Balanced Precision = {:0.4f}, Average Balanced Accuracy = {:0.4f}, ' \
                    'Average Balanced F1 score = {:0.4f}, Average Balanced Recall = {:0.4f}, Average False Positive Error = {:0.4f}\n').format(
                    avg_prec / (i + 1), avg_bal_acc / (1 + i), avg_f1 / (i + 1), avg_recall / (i + 1), avg_fp_error / (i + 1)))

    avg_loss /= len(train_loader) 
    avg_acc /= len(train_loader)

    return avg_loss, avg_acc

def validate(valid_loader, model, epoch, loss, gpu_id, args, writer=None, curriculum_counter=None):
    """
    Inputs:
        valid_loader - a torch data loader
        model - neural network to pass input through
        epoch - (int): current epoch (for logging purposes)
        loss - loss object to use
        gpu_id - ID (index) of the GPU to use
        writer - tensorboard writer to write to 
        args - args from argsparse 
    Outputs:
        avg_loss - (torch.float): gives the average loss of the epoch
        avg_acc - (torch.float): gives the average acc of the epoch
    """
    # set to eval
    model.eval()
    assert model.training == False

    # convert dimension weighting for loss to floats
    xyz_weight = torch.FloatTensor(args.xyz_weighting).cuda(gpu_id, non_blocking=True)
    labels_mu = torch.FloatTensor(args.labels_mu).cuda(gpu_id, non_blocking=True)
    labels_std = torch.FloatTensor(args.labels_std).cuda(gpu_id, non_blocking=True)
    mean_offset = torch.FloatTensor(np.array([args.mean_offset])).cuda(gpu_id, non_blocking=True)
    epsilon = torch.FloatTensor(np.array([args.epsilon])).cuda(gpu_id, non_blocking=True)

    avg_loss = 0
    avg_acc = 0
    avg_f1 = 0
    avg_bal_acc = 0
    avg_prec = 0 
    avg_recall = 0
    avg_fp_error = 0
    avg_x_std = 0
    avg_z_std = 0
    with torch.no_grad():
        for i, sample in enumerate(valid_loader):
            # record start time
            t0 = time.perf_counter()

            # send values to gpu
            x1 = sample[0].cuda(gpu_id, non_blocking=True).type(torch.float32)
            x2 = sample[1].cuda(gpu_id, non_blocking=True).type(torch.float32)
            if args.output_dist:
                y = sample[2].cuda(gpu_id, non_blocking=True).type(torch.float32)
            elif args.output_multi:
                y = sample[3].cuda(gpu_id, non_blocking=True).type(torch.long)
            else:
                shift = sample[2].cuda(gpu_id, non_blocking=True).type(torch.float32)
                y = sample[3].cuda(gpu_id, non_blocking=True).type(torch.float32)
            # send value through network 
            output = model(input1=x1, input2=x2, epsilon=epsilon, mean_offset=mean_offset)

            # calculate loss and accuracy
            if args.output_dist:
                loss1 = loss(output[0], y[:,0])*xyz_weight[0]
                loss2 = loss(output[1], y[:,1])*xyz_weight[1]
                valid_loss = (loss1+loss2)/2
                valid_acc = mse_accuracy(output, y, labels_mu, labels_std)
            elif args.output_multi:
                valid_loss = loss(output, y)
                valid_acc = accuracy.multi_class_acc(output, y, softmax=True)
                #TODO haven't tested this yet
                valid_bal_acc, valid_precision, valid_recall, valid_f1, _ = accuracy.get_balanced_metrics(output, y, softmax=True)
            else:
                output = output.view(-1)
                y = y.view(-1)
                valid_loss = loss(output, y)
                valid_acc = accuracy.binary_acc(output, y, sigmoid=True)
                valid_bal_acc, valid_precision, valid_recall, valid_f1, _ = accuracy.get_balanced_metrics(output, y, sigmoid=True)
                false_pos_error = accuracy.false_pos_error(output, y, shift, args.curriculum_classes[curriculum_counter-1],
                    labels_mu, labels_std, sigmoid=True)


            avg_loss += valid_loss.item()
            avg_acc += valid_acc.item()
            if args.output_dist:
                avg_x_std += output[0].stddev.mean().item()
                avg_z_std += output[1].stddev.mean().item()
            elif args.output_binary or args.output_multi:
                avg_f1 += valid_f1.item()
                avg_bal_acc += valid_bal_acc.item()
                avg_prec += valid_precision.item()
                avg_recall += valid_recall.item()
                avg_fp_error += false_pos_error

            # log info on main process (logs info every iteration)
            if gpu_id == args.gpu_indexes[0]:
                if (i) % args.log_rate == 0:
                    writer.add_scalar('Average_Loss/Validation', 
                                    avg_loss / (i + 1),
                                    epoch * len(valid_loader) + i + 1)
                    writer.add_scalar('Average_Accuracy/Validation', 
                                    avg_acc / (i + 1),
                                    epoch * len(valid_loader) + i + 1)
                    writer.add_scalar('Average_X_Standard_Deviation/Validation',
                                    avg_x_std / (i + 1),
                                    epoch * len(valid_loader) + i + 1)
                    writer.add_scalar('Average_Z_Standard_Deviation/Validation',
                                    avg_z_std / (i + 1),
                                    epoch * len(valid_loader) + i + 1)
                    if args.output_binary or args.output_multi:
                        writer.add_scalar('Average_Balanced_F1_Score/Validation',
                                        avg_f1 / (i + 1),
                                        epoch * len(valid_loader) + i + 1)
                        writer.add_scalar('Average_Balanced_Accuracy/Validation',
                                        avg_bal_acc / (i + 1),
                                        epoch * len(valid_loader) + i + 1)
                        writer.add_scalar('Average_Balanced_Precision/Validation',
                                        avg_prec / (i + 1),
                                        epoch * len(valid_loader) + i + 1)
                        writer.add_scalar('Average_Balanced_Recall/Validation',
                                        avg_recall / (i + 1),
                                        epoch * len(valid_loader) + i + 1)
                        writer.add_scalar('Average_False_Positive_Error(m)/Validation',
                                        avg_fp_error / (i + 1),
                                        epoch * len(valid_loader) + i + 1)
                        if args.log_extra_info:
                            writer.add_figure('Confusion_Matrix/Validation',
                                            accuracy.plot_confusion_matrix(
                                                predictions=output,
                                                labels=y,
                                                sigmoid=args.output_binary,
                                                softmax=args.output_multi),
                                            epoch * len(valid_loader) + i + 1)
                    if args.plot_grad_flow:
                        writer.add_figure('Gradient_Flow/Validation',
                                        plot_grad_flow(model.named_parameters()),
                                        epoch * len(valid_loader) + i + 1)
            t1 = time.perf_counter()
            if (i) % args.print_rate == 0:
                print('Currently on validation epoch {} and iteration {} of {}'.format(
                    epoch+1, i+1, len(valid_loader)))
                if args.output_dist:
                    print('Average loss = {:0.4f}, Average accuracy = {:0.4f}, Loop Time = {:0.4f}s'.format(
                        avg_loss / (i + 1), avg_acc / (i + 1), (t1 - t0)))
                    print('Average x std = {:0.4f}, Average z std = {:0.4f}\n'.format(
                        avg_x_std / (i + 1), avg_z_std / (i + 1), (t1 - t0)))
                else:
                    print('Average loss = {:0.4f}, Average accuracy = {:0.4f}, Loop Time = {:0.4f}s'.format(
                        avg_loss / (i + 1), avg_acc / (1 + i), (t1 - t0)))
                    print(('Average Balanced Precision = {:0.4f}, Average Balanced Accuracy = {:0.4f}, ' \
                        'Average Balanced F1 score = {:0.4f}, Average Balanced Recall = {:0.4f}, Average False Positive Error = {:0.4f}\n').format(
                        avg_prec / (i + 1), avg_bal_acc / (1 + i), avg_f1 / (i + 1), avg_recall / (i + 1), avg_fp_error / (i + 1)))

    # plot some of the predictions #NOTE need to change the below code if you predict things other than x and z
    intrinsics = args.intrinsics
    extrinsics = args.extrinsics
    sample_number = 1
    if args.output_dist:
        prediction = torch.stack((output[0].loc, output[1].loc), axis=-1)
        std = torch.stack((output[0].scale, output[1].scale), axis=-1)
        label = torch.clone(y)
        binary = False
    else:
        prediction = torch.zeros((output.shape[0],3))
        label = sample[2]
        binary = True #TODO make sure these figures work and that the labels are correct
        std = None
    if args.log_extra_info:
        pred_fig = image_utils.plot_placement_pred(image=x1,
                                                prediction=prediction,
                                                std=std,
                                                label=label,
                                                intrinsics=args.intrinsics,
                                                extrinsics=args.extrinsics,
                                                image_mu=args.image_mu,
                                                image_std=args.image_std,
                                                label_mu=args.labels_mu,
                                                label_std=args.labels_std,
                                                initial_obj_pose=args.initial_obj_pose,
                                                num_samples=1,
                                                convert=True,
                                                title=f'Prediction on epoch_{epoch+1}',
                                                binary_classifier=binary,
                                                binary_result=y,
                                                n_stds=100
        )
        writer.add_figure(f'Predictions/Validation',
                            pred_fig,
                            epoch+1
        )
    # #TODO might want to paste the image onto the preditcion (crop the obj image and paste onto scene)

    avg_loss /= len(valid_loader)
    avg_acc /= len(valid_loader)

    return avg_loss, avg_acc

def eval_model(model, dataloader, gpu_id, args):
    """
    Evaluate model on dataset, this is different from validate() because
    it returns the predictions of the model on the given dataset
    Only implemented for binary classification
    Inputs:
        model (nn.Module): The model to evaluate the data on
        dataset (torch Dataset): The dataset to evaluate model with
    """
    # set to eval
    model.eval()
    assert model.training == False

    # convert dimension weighting for loss to floats
    xyz_weight = torch.FloatTensor(args.xyz_weighting).cuda(gpu_id, non_blocking=True)
    epsilon = torch.FloatTensor(np.array([args.epsilon])).cuda(gpu_id, non_blocking=True)

    prog = tqdm.tqdm(initial=0, total=len(dataloader), file=sys.stdout, desc='Batches left to evaluate')
    with torch.no_grad():
        rm_indexes = torch.tensor([]).cuda(gpu_id, non_blocking=True)
        for sample in dataloader:
            # send values to gpu
            x1 = sample[0].cuda(gpu_id, non_blocking=True).type(torch.float32)
            x2 = sample[1].cuda(gpu_id, non_blocking=True).type(torch.float32)
            indexes = sample[4].cuda(gpu_id, non_blocking=True).type(torch.float32)
            # send value through network 
            output = model(input1=x1, input2=x2, epsilon=epsilon)
            output = output.view(-1)
            # add indexes of data that is labeled as over threshold
            predictions = torch.sigmoid(output)
            predictions = torch.round(predictions)
            rm_idx = indexes[predictions == 0.0]
            rm_indexes = torch.cat((rm_indexes, rm_idx), axis=0)
            prog.update(1)
            prog.refresh()
    return rm_indexes


def setup_env(addrs='localhost', port='12355'):
    """
    Setup environmental vairables
    Inputs:
        addrs(str): the master address to use for the nodes and processes to communicate with one another
        port(int): the master port to use for communcation
    """
    os.environ['MASTER_ADDR'] = addrs
    os.environ['MASTER_PORT'] = port 

def cleanup():
    """Cleanup one of the processes"""
    dist.destroy_process_group()

def save_checkpoint(state, filepath=None, filename=None):
    if filepath is None:
        filepath = '.'
    if filename is None:
        filename = 'checkpoint.pth.tar'
    torch.save(state, f'{filepath}/{filename}')

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mse_accuracy(predicted, ground_truth, mu=0, std=1):
    """
    Mean squared error as the acuracy
    Args:
        predicted (): tuple of torch.distributions
    """
    with torch.no_grad():
        #TODO need to decide whether to use mean or sample from the distribution
        predicted = torch.stack((predicted[0].mean, predicted[1].mean),axis=-1)

        assert predicted.ndim == 2 and ground_truth.ndim == 2 # assuming axis 0 is num_samples
        assert predicted.shape[1] == 2
        assert ground_truth.shape[1] == 2
        # de-normalize #TODO do you unnormalize and then get value or perform caluclation and then unormalize
        # Also might want to give accuracy for each dimension
        predicted = (predicted * std) + mu
        ground_truth = (ground_truth * std) + mu
        acc = F.mse_loss(predicted, ground_truth)
        # squared part
        acc = torch.sqrt(acc)
        #acc = (acc * std) + mu

        return acc

if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(f'Exception:\n{e}\nRemember to delete the init_method file for file transfer between GPUs')
    #     # https://discuss.pytorch.org/t/shared-file-system-is-a-file/51151
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

import rnns.attention as attention
import rnns.networks as networks
import rnns.utils as utils
import rnns.preprocess as preprocess
"""
Note:
This script was written for use in a sequence ot sequence neural network.
Some of the functions aren't written to optimize a normal feed forward 
network well.
Config:
python 3.6.8
torch 1.3.1
"""
def main():
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('-d', '--datapath', type=str, default='/home/stevenl3/Darknet/images/rnn_training',
                        help='The path to data file or the direcotry containing the data files, do not include / if using a directory')
    parser.add_argument('-e', '--epochs', default=2000, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('-op', '--optimizer', default='RMSProp', type=str,
                        help="Can be 'RMSProp', 'Adam', or 'SGD'")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-m', '--momentum', type=float, default=0.1)
    parser.add_argument('-w', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-dp', '--drop_prob', type=float, default=0.5,
                        help='''The dropout probability to use. Should be between 0.0 and 1.0. Dropout is on
                             resnet before last fc, and LSTM on first two layers''')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-mb', '--mini_batch_size', default=None, type=float,
                        help='Size of the subset of the dataset to use for training instead of the entire dataset. ' +\
                             'Use a value > 1 to specify a specific mini batch size (rounds value to int) or a float ' +\
                             'from 0-1 to use a percentage. Default is None and will use the entire dataset. ' +\
                             'The subset of data used will always be shuffled. This DOES NOT work with multiple GPUs')
    parser.add_argument('-ts', '--train_test_split', type=float, default=0.8,
                        help='Percentage of data to use for training, the rest is for validation. Must be < 1 and >0')
    parser.add_argument('-in', '--initialization', default=None, type=str,
                        help='None for default intialization or "general_uniform" or "general_normal"')
    parser.add_argument('-ep', '--epsilon', default=0.0, type=float,
                        help='Value to add to the values in network to avoid zero division')
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
    parser.add_argument('-tp', '--tensorboard_path', type=str, default='/home/stevenl3/Attention/runs/rnns/run1',
                        help='Path to save the tensorboard log data to')
    parser.add_argument('-sr', '--save_rate', type=int, default=250,
                        help='How often to save the model, in epochs')
    parser.add_argument('-sp', '--save_path', type=str, default=None,
                        help='Where to save the checkpoints to')
    parser.add_argument('-sn', '--save_name', type=str, default=None,
                        help='Name to save the checkpoints as')
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
 
    encoder_lr = None
    decoder_lr = None 
    img_size = 224 #img_size = 416
    seq_length = 8
    train_length = 7
    remove_length = seq_length - train_length
    loss1_weight = 1 # for encoder/embeddings
    loss2_weight = 1 # for decoder/rnn
    norm_weight = 1 # weight to scale the normalized loss input by #TODO these aren't normalized though?

    # hyperparameters for embeddings CNN
    res_net_size = 224
    embed_fc1_size = 1000
    embed_fc2_size = 512
    embed_size = 256

    # hyperparameters for RNN
    rnn_in_size = embed_size
    rnn_out_size = 2 # set to img_size*img_size for masks and 2 for obj_centers
    rnn_layers = 4
    hidden_dim = 512

    # data path to load from
    data_name = 'train_data.npz'

    if args.save_path is None:
        args.save_path = args.tensorboard_path

    if args.debug_mode is True:
        autograd.set_detect_anomaly(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        torch.manual_seed(0)
        args.shuffle = True #False
        args.multi_gpu_mode = False
        args.num_workers = 0
    else:
        args.shuffle = True

    # Start loading data
    t_begin = time.perf_counter()
    print('Loading Data...')
    stuff = np.load(f'{args.datapath}/{data_name}')
    data = stuff['train_data']
    labels = stuff['train_labels']

    # for tensorboard
    # NOTE  use $ tensorboard --logdir=<PATH> --host localhost in terminal
    writer = SummaryWriter(f'{args.tensorboard_path}')

    # define transform NOTE not using right now
    # transform = tf.Compose([tf.Resize([res_net_size, res_net_size]),
    #                         tf.Normalize(mean=[0.485, 0.456, 0.406], 
    #                                      std=[0.229, 0.224, 0.225])])
    transform = tf.Resize([res_net_size,res_net_size])

    print('Splitting data into train and validation...')
    train_test_data = preprocess.train_test_split(data,
                                                  labels,
                                                  args.train_test_split,
                                                  mix=True)
    train_x, train_y, test_x, test_y = train_test_data
    # convert to tensors, split into batches and shuffle
    # NOTE: use len(train_data) for dataset size
    # train_data[n] gives sample number n as a tuple (x, y)
    train_data = preprocess.SequenceDataset(train_x,
                                            train_y,
                                            rm_amount=remove_length
    )
    test_data = preprocess.SequenceDataset(test_x,
                                           test_y,
                                           rm_amount=remove_length
    )
    train_loader = DataLoader(train_data,
                              shuffle=args.shuffle,
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers,
                              drop_last=True
    )
    test_loader = DataLoader(test_data,
                             shuffle=args.shuffle,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             drop_last=True
    )

    print("Getting ready...")
    # use GPU if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # clear cache
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # define network architecture
    embedder = networks.ResNetEmbeddings(embed_size=embed_size, 
                                         fc1_size=embed_fc1_size,
                                         fc2_size=embed_fc2_size,
                                         drop_prob=args.drop_prob,
                                         momentum=args.momentum
    )
    rnn = networks.LSTMNet(input_dim=rnn_in_size,
                           hidden_dim=hidden_dim,
                           output_dim=rnn_out_size,
                           num_layers=rnn_layers,
                           drop_prob=args.drop_prob
    )
    network = networks.Seq2SeqNet(encoder=embedder,
                                  decoder=rnn,
                                  device=device, 
                                  embed_in=embed_size,
                                  embed_out=rnn_out_size,
                                  fc_size=hidden_dim
    )
    network.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    # define loss function
    loss1 = nn.MSELoss()
    loss2 = nn.L1Loss() #use 1 over mse if loss values are too small

    # make text with network architecture
    i = 1 
    model_desc = f'{agrs.tensorboard_path}/network_info_trial{i}.txt'
    # make sure not to overwrite old files
    while os.path.isfile(model_desc):
        model_desc = f'{args.tensorboard_path}/network_info_trial{i}.txt'
        i += 1

    f = open(model_desc, 'w')
    f.write(f'Model architecture info: \n epochs: {args.epochs}, \
            \n learning rate: {args.learning_rate}, \
            \n encoder learning rate: {encoder_lr}, \
            \n decoder learning rate: {decoder_lr}, \
            \n momentum: {args.momentum}, \
            \n drop_prob: {args.drop_prob}, \
            \n batch size: {args.batch_size}, \
            \n train/test split percent: {args.train_test_split}, \
            \n train data: {args.datapath}/{data_name}, \
            \n enocder loss: {loss1}, \
            \n decoder loss: {loss2} \
            \n optimizer: {optimizer}, \
            \n model: {network}')
    f.close()

    # set up clock for epochs
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)

    # instantiate loss and scores for recording progress
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    print('Beginning training')
    for epoch in range(args.epochs):
        # record start time
        t_start = time.perf_counter() # I think you need to add torch.cuda.synchronize() before statrting and stopping
        train_loss, train_accuracy = train(train_loader=train_loader, 
                            network=network, epoch=epoch, optimizer=optimizer,
                            loss1=loss1, loss2=loss2, device=device, writer=writer, 
                            log_rate=args.log_rate, norm_weight=norm_weight)
        valid_loss, valid_accuracy = validate(valid_loader=test_loader,
                            network=network, epoch=epoch, optimizer=optimizer,
                            loss1=loss1, loss2=loss2, device=device, writer=writer, 
                            log_rate=args.log_rate, norm_weight=norm_weight)

        # reference for how to set up the loss, this is a sequence to sequence model
        # https://discuss.pytorch.org/t/why-lstm-models-do-not-require-labels-for-each-step/45895
        # train_losses.append(train_loss)
        # train_accuracies.append(train_accuracy)
        # valid_losses.append(valid_loss)
        # valid_accuracies.append(valid_accuracy)
        #NOTE not doing anything with these values right now

        t_end = time.perf_counter()

        print(f'Epoch {epoch+1} took {t_end-t_start:0.4f}s\n')
        if (epoch+1) % args.save_rate == 0:
            # input("Press enter to continue saving, otherwise Ctrl+C to stop script...")
            print(f'Saving model at epoch {epoch+1} to {save_path}\n')
            torch.save(network, f'{save_path}/seq2seq_model_{epoch+1}')
            # should load and save whole path, if you want to retrain something
            # you used model.parameters() to get parameters and this only
            # works with nn.Modules, not list or state_dicts
            # so for training make sure to load all the networks and subnetworks
            # as the entire models
    t_final = time.perf_counter()
    print(f"Total runtime was {t_final-t_begin:0.4f}")


def train(train_loader, network, epoch, optimizer, loss1, 
          loss2, device, writer=None, log_rate=10, norm_weight=1):
    """
    A single epoch training loop

    Inputs:
        train_loader - a torch data loader
        network - neural network to pass input through (assuming a nn.Module)
        epoch - (int): current epoch (for logging purposes)
        optimizer - optimizer object to use for training
        loss - loss object to use for training
        device - "cuda" or "cpu" 
        writer - tensorboard writer to write to 
        log_rate - (int): the interval to log the training status
        norm_weight - hyperparameter to multiply values output and label by
    Outputs:
        avg_loss - (torch.float): gives the average loss of the epoch
        avg_acc - (torch.float): gives the average acc of the epoch
    """
    # set to train
    network.train()

    assert network.training == True

    avg_loss = 0
    avg_acc = 0
    for i, (x, y, obj_shape, y0, obj_shape0) in enumerate(train_loader):
        # record start time
        t0 = time.perf_counter()
        x = x.to(device)
        y = y.to(device)
        y0 = y0.to(device)
        # zero the optimizer
        optimizer.zero_grad()

        # pass through network
        output, _, encode_out = network(x)
        # normalize predictions to between 0-1 -> (out/416,y/416)
        #NOTE do you realy need to norm output? why not just label
        y = y.permute(1,0,2)
        y0 = y0.permute(1,0,2)
        norm_output =  torch.mul(output, norm_weight)
        norm_y = torch.mul(y, norm_weight)
        norm_y0 = torch.mul(y0, norm_weight)
        norm_encode_out = torch.mul(encode_out, norm_weight)
        # calculate loss (default is average loss)
        train_loss = torch.mul(loss1(norm_encode_out, norm_y0), loss1_weight) + \
                     torch.mul(loss2(norm_output, norm_y), loss2_weight)
        avg_loss += train_loss.item()
        #NOTE might also want to use cdist as loss
        #NOTE also might want to consider instance or layer normilization

        # record accuracy (num correct predictions/num images)
        train_acc = utils.get_accuracy(output, y, 0.01) # threshold of 1% of 416
        avg_acc += train_acc.item() #NOTE this might be wrong

        # backprop
        train_loss.backward()
        optimizer.step()

        # log info, if loss is not set to sum, it averages, ie divides by total_num elements, so just divide by i
        if (i) % log_rate == 0:
            writer.add_scalar('Average_Loss/Training',
                              avg_loss / (i + 1),
                              epoch * len(train_loader) + i + 1)
            writer.add_scalar('Average_Accuracy/Training',
                              avg_acc / (i + 1),
                              epoch * len(train_loader) + i + 1)
        # record loop finish time
        t1 = time.perf_counter()
 
        print('Currently on training epoch {} and iteration {} of {}'.format(
               epoch+1, i+1, len(train_loader)))
        print('Loss = {:0.4f}, Accuracy = {:0.4f}, Loop Time = {:0.4f}s\n'.format(
               avg_loss / (i + 1), train_acc.item(), (t1-t0)))
    
    avg_loss /= len(train_loader) # divide by num images/predictions
    avg_acc /= len(train_loader) # divide by num images/predictions

    return avg_loss, avg_acc

def validate(valid_loader, network, epoch, optimizer, loss1, 
             loss2, device, writer, log_rate=10, norm_weight=1):
    """
    Inputs:
        valid_loader - a torch data loader
        network - neural network to pass input through
        optimizer - optimizer object that was used, so save state
        loss - loss object to use
        device - cuda or cpu
    """
    # set to eval
    network.eval()
    
    assert network.training == False

    avg_loss = 0
    avg_acc = 0
    with torch.no_grad():
        for i, (x, y, obj_shape, y0, obj_shape0) in enumerate(valid_loader):
            # record start time
            t0 = time.perf_counter()
            # set values to cpu/gpu
            x = x.to(device)
            y = y.to(device)
            y0 = y0.to(device)
            # send value through network 
            output, _, encode_out = network(x)
            # weigh the outputs and labels
            y = y.permute(1,0,2)
            y0 = y0.permute(1,0,2)
            norm_output = torch.mul(output, norm_weight)
            norm_y = torch.mul(y, norm_weight)
            norm_y0 = torch.mul(y0, norm_weight)
            norm_encode_out = torch.mul(encode_out, norm_weight)
            # calculate loss (default is average loss)
            valid_loss = torch.mul(loss1(norm_encode_out, norm_y0), loss1_weight) + \
                         torch.mul(loss2(norm_output, norm_y), loss2_weight)            
            avg_loss += valid_loss.item()
            #NOTE might also want to use cdist as loss
            
            # record accuracy (num correct predictions/num images)
            valid_acc = utils.get_accuracy(output, y, 0.01) # threshold of 1% of 416
            avg_acc += valid_acc.item()

            writer.add_scalar('Average_Loss/Validation', 
                               avg_loss / (i+1), epoch+1)

            writer.add_scalar('Average_Accuracy/Validation', 
                               avg_acc / (i+1), epoch+1)

            # record loop finish time
            t1 = time.perf_counter()

            print('Currently on validation epoch {} and iteration {} of {}'.format(
                epoch+1, i+1, len(valid_loader)))
            print('Loss = {:0.4f}, Accuracy = {:0.4f}, Loop Time = {:0.4f}s\n'.format(
                avg_loss / (i+1), valid_acc.item(), (t1-t0)))
    
    # plot some of the predictions
    sample_number = 3
    subplot = (1, 1)
    seq_length = x.shape[1]
    # unnormalize image location predictions
    plt_output = torch.round(torch.mul(output, img_size))
    plt_output = plt_output.permute(1,0,2)
    plt_obj_shape = torch.round(torch.mul(obj_shape, img_size))
    img = torch.round(torch.mul(x, 255))

    for j in range(sample_number):
        # get random indices
        idx = np.random.randint(0, x.shape[0])
        i = np.random.randint(0, seq_length)
        writer.add_figure(f'Predictions/Sample{j+1}',
                    utils.plot_box(img[idx][i], plt_output[idx][i],
                    plt_obj_shape[idx][i], 1, subplot,
                    f'Prediction on epoch_{epoch+1}'),
                    epoch+1)

    avg_loss /= len(valid_loader) # divide by num images/predictions
    avg_acc /= len(valid_loader) # divide by num images/predictions

    return avg_loss, avg_acc

if __name__ == "__main__":
    main()

"""
References:
https://github.com/HHTseng/video-classification
"""

"""
Comments:
- learning rate of 1e-3 seems good as long as the epochs are relatively higher
- L1 loss with no sum is not learning, actually L1 in general 
    seems to just go towards the center and stay there
    - might be better to use L1 though, MSE is given false Loss?
- There might be a bug with the validation or training loss
    - why are the loss values so low if the predictions are so bad?
    - try to step through code and check for bugs
- Also maybe try image differences and pass 6 channels instead?
    - so first 3 channels are just the images. the last 6 is the
        difference between the current image and the previous
    - another thing to try is to add a mask as the 4th layer, 
        where the mask is ones aorund the object and zeros else where
- Also, try multiple losses with different learning rates
    - have the output of the cnn compared to the correct label
        of where the current object is. this should be one loss
    - the other loss is the loss of the rnn. should weigh these two
        differently. if the weights for the cnn are correct make
        itds weights really low so it doesn't change
- the loss values should be out of one, like error/percentage
- the element wise division using /, doesn't seem to work properly?
    or maybe just isnt accurate? use torch.div, try using only their functions
- Pytorch loss function for mse and l1 are element wise loss and sum
    You need the 2d version of this since you are using cartesian coordinates
    for mse in particular, idk if its necesarry for l1
    https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/2
    https://pytorch.org/docs/master/notes/extending.html
- might want to set up the loss of the cnn like yolo, loss for position,
    another for height/width, and another for object class
    (ie, for width/height make the output of the network 4 instead of 2)
- RESNET is taking in the 416x416 inputs and its working?
- Might be lossing accuracy/decimal places because you aren't using
    big enough dtypes (maybe double)
- in  preprocessing  webuild a dictionary mapping the length of a 
    sentence to thecorresponding subset of captions. Then, during
    training werandomly sample a length and retrieve a mini-batch
    of size64 of that length. (see show, attend and tell paper)
- for adam optimizer beta2 should be set close to 1.0 on problems with
    a sparse gradient (e.g. NLP and computer vision problems
- maybe set optimizer up to have decaying learning rate? adam already does, but you can increase it
- might want to get rid of the batch norm in CNN
- batch normalization causes training loss/acc to be a lot higher than validation
"""

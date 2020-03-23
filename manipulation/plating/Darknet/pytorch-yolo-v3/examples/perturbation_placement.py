import numpy as np
import matplotlib.pyplot as plt
import argparse

from rnns import utils
from prediction_utils import collect_data
from prediction_utils import gaussian
from prediction_utils import predict
from prediction_utils import kernel_density

if __name__ == '__main__':
    """
    Example script of training a model to predict the best location to
    place the next item for a caprese salad, with a perturbation to one
    of the placements. Uses meanshift clustering and gaussian mixture models.
    
    Notes:
        - If the sequence length of the shortest sequence being trained on is S
          then sequence_length argument must be < (S-(n+1)). Subtract another 1
          from seq_length if the training sequence contains a blank cutting 
          board image at the beginning
        - Works much nicer if the reference sequence you are using is longer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', '-d', type=str,
                        default='../../images/test04/images',
                        help='''Directory where all of the subdirectories contain 
                        a .npy file containing the bounding box information for an 
                        image sequence.''')
    parser.add_argument('--sequence_length', '-l', type=int, default=10,
                        help='Length of sequence to produce')
    parser.add_argument('--suffix_list', '-u', type=list, default=['06', '06'],
                        help='''A list of strings that represent the suffixes
                        of the sub-directories to gather the data from''')
    parser.add_argument('--bandwidths', '-b', default=[0.01, 0.001, 0.02, 0.04, 0.01, 0.005],
                        help='Bandwidth to use for meanshift clustering')
    parser.add_argument('--iterations', '-i', type=int, default=8000)
    parser.add_argument('--placement_area', '-a', type=list,
                        default=[173, 138, 256, 204],
                        help='Area in image coordinates to restrict predictions to')
    parser.add_argument('--perturbation_index', '-p', type=int, default=7,
                        help='The sequence index to apply a perturbation')
    parser.add_argument('--n', '-n', type=int, default=1, help='''Number of previous
                        time steps to consider for training''')
    parser.add_argument('--save_path', '-s', type=str,
                        default='../results',
                        help='''Directory to save images of placements to. Leave as
                        None to just display the results''')
    args = parser.parse_args()

    workin_dir = args.train_data_path
    save_path = args.save_path
    box_list = args.suffix_list
    n = args.n
    seq_length = args.sequence_length
    perterb_idx = args.perturbation_index
    iterations = args.iterations
    bandwidths = args.bandwidths
    # if bandwidths were only given for the 6 feature dimensions for one n=1, duplicate n times
    if len(bandwidths) == 6:
        bandwidths = bandwidths * n

    work_area = args.placement_area #location of cutting board
    # need to remove last alpha layer of PIL images since these images are 4D
    tomato = plt.imread('../../images/test04/resized_images/foreground0.png')[:,:,:3] *255
    cheese = plt.imread('../../images/test04/resized_images/foreground10.png')[:,:,:3]*255
    
    tomato, tomato_mask = utils.crop_image(tomato, mode='Circle')
    cheese, cheese_mask = utils.crop_image(cheese, mode='Circle')

    #just for height and width measurements
    tomato_label = np.load('../../images/test04/images/set01/predictions.npy')[2]
    cheese_label = np.load('../../images/test04/images/set02/predictions.npy')[2]
    
    background = plt.imread('../../images/test04/resized_images/background0.png')[:,:,:3]*255
    plate_label = np.load('../../images/test04/images/set01/predictions.npy')[0]
    plate_dims = 0 # use preset dimensions

    # first placement location
    last_obj = np.array([1., 223.92022705, 145.13595581, 240.45913696, 160.01464844]) # first object location
    count = 1 # for switching between cheese and tomato
    #placing cheese first
    foreground = cheese 
    mask = cheese_mask
    
    # #format label so it can be used with Prediction class
    h = last_obj[4] - last_obj[2]
    w = last_obj[3] - last_obj[1]
    last_location = [last_obj[2]+h//2, last_obj[1]+w//2] #H,W
    
    #make first placement
    place = np.copy(last_location)
    place[1]  = place[1] - 3
    last_image = utils.paste_image(background, foreground, last_location, 
                                   alpha_mask=mask)

    n_objs = None
    for i in range(seq_length):
        #collect training data, only for first item in sequence
        data = collect_data.gather_data(workin_dir, n=n, suffix_list=box_list,
                           remove_first=False, return_n=i+1)
        #train gaussians
        _, centers = gaussian.mean_shift(data, bandwidths)
        score = gaussian.MultiGaussian(data, centers, n)

        #switch between cheese and tomato
        if count % 2 == 1:
            foreground = np.copy(tomato)
            mask = np.copy(tomato_mask)
            new_obj = np.copy(tomato_label) #just for height and width
        else:
            foreground = np.copy(cheese)
            mask = np.copy(cheese_mask)
            new_obj = np.copy(cheese_label) #just for height and width
        count +=1

        # perform a perturbed placement
        if i == perterb_idx-1:
            new_location = [last_location[0]+7, last_location[1]+8]
            # plot results
            new_image = utils.paste_image(last_image, foreground, new_location, 
                                       alpha_mask=mask)
            plt.imshow(new_image)
            if save_path is not None:
                plt.imsave(f'{save_path}/figure{i+1}_perturbation.png', new_image)
            else:
                plt.imshow(new_image)
                plt.show()
            #updates variables
            n_objs = np.vstack((n_objs, np.array(new_location).reshape(1,2)))
            last_location = new_location
            last_image = new_image
            #format label so it can be used with Prediction class
            last_obj = utils.mins_max(last_location, foreground.shape[0], foreground.shape[1])
            last_obj = np.insert(last_obj, 0, 0) #Nx5 now instead of Nx8
            continue

        #get prediction
        pred = predict.Prediction(iterations, last_obj, plate_label, 
            plate_dims, last_image, new_obj, score, n_objs)
        mode = 'all'
        # mode = ['dp'] # uncomment to see how only taking distance to plate effects predictions
        winning, n_objs = pred.winner2D(mode=mode, max_n=n, n=i+1) #Doesnt work
        if i+1 > n:
            m = n
        else:
            m = i+1
        pred.img = pred.img[:,:,::-1] #convert BGR to RGB
        winning, n_objs = pred.plot_2D_gaussian(mode=mode, n=m, i=i+1, save_path=save_path)

        new_location = [winning[0], winning[1]] #switch xy to yx
        # paste results into image
        new_image = utils.paste_image(last_image, foreground, new_location, 
                                       alpha_mask=mask)
        
        if save_path is not None:
            plt.imsave(f'{save_path}/figure{i+1}_placement.png', new_image)
        else:
            plt.imshow(new_image)
            plt.show()

        # plots bounding box of object at predicted location
        # pred.plot_prediction(winning, pred.new_obj_width, pred.new_obj_height)

        # update values
        last_image = new_image
        last_location = new_location
        #format label so it can be used with Prediction class
        last_obj = utils.mins_max(last_location, foreground.shape[0], foreground.shape[1])
        last_obj = np.insert(last_obj, 0, 0) #Nx5 now instead of Nx8

    if save_path is not None:
        print(f"Saved files to {save_path}")
    print('Finished')

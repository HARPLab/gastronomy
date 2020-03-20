import numpy as np


from prediction_utils import gaussian
from prediction_utils import predict
from prediction_utils import collect_data
from prediction_utils import heatmap

if __name__ == "__main__":
############################ Parameters #################################
    workin_dir = '../../images/test04/images'
    testing_dir = '../../images/test04/images/set05'
    # testing_dir = '../../images/test03/labels/data/set03'

    # box_list = ['05', '07', '08']
    # box_list = ['05', '06', '07']
    # box_list = ['09', '10', '11']
    box_list = ['05', '06']
    n = 1

    iterations = 8000
    bandwidth = [0.01, 0.01, 0.1, 0.03, 0.01, 0.008]
    bandwidth2 = [0.01, 0.001, 0.02, 0.04, 0.01, 0.005]
    bandwidth = bandwidth * n
    bandwidth2 = bandwidth2 * n

    titles = ['Horizontal Distance\nBetween Object Centers', 
    'Veritcal Distance\nBetween Object Centers',
    'Horizontal Distance Between\nObject and Plate Centers',
    'Vertical Distance Between\nObject and Plate Centers',
    'Horizontal Distance\nBetween Object Edges',
    'Veritcal Distance\nBetween Object Edges']
    titles = titles * n

    xlabels = ['dcx', 'dcy', 'dpx', 'dpy', 'dex', 'dey']
    xlabels = xlabels * n

    # Load the data
    data = collect_data.gather_data(workin_dir, n=n, suffix_list=box_list)

##################### Mean Shift Predictions ##########################
    # prep a sequence of images for predictions
    prep_predict = collect_data.Preperation("{}".format(testing_dir), (416,416)) 
    images = prep_predict.images()
    bboxes = prep_predict.boxes()
    cutting_board_idx = prep_predict.board_index()
    data2 = collect_data.ImageData(images, bboxes, cutting_board_idx, 0)

    # define scoring function
    labels, centers = gaussian.mean_shift(data, bandwidth2)
    score = gaussian.MultiGaussian(data, centers, n)
    # # Another option
    # score = kernel_density.KDE(data)
    # logprob = False
    # score = gaussian.GaussianMix(centers, bandwidth, labels, weighting=True)

    predictions = [] # might want to use this to show all the predictions
    n_objs = None
    for i in range(len(data2.images)-1):
        # Define the image to show, previous object, next object, and cutting board
        img = data2.images[i+1]
        last_obj = data2.food_boxes[i]
        plate = data2.boxes[data2.board_index[i], :]
        plate_dims = 0
        new_obj = data2.food_boxes[i+1]
        # instantiate prediction class
        pred = predict.Prediction(iterations, last_obj, plate, plate_dims, 
                                  img, new_obj, score, n_objs)
        # get coordinate with highest score
        mode = ['dc']
        # mode = 'all'
        # mode = ['dp']
        winning, n_objs = pred.winner2D(mode=mode, max_n=n, n=i+1)
        predictions.append(winning)
        # plot results
        pred.plot_prediction(winning, pred.new_obj_width, pred.new_obj_height)
        pred.plot_2D_gaussian(mode=mode, n=n)

###################### Unimodal Predictions ###########################
    #Load data
    workin_dir = '../../images/test03/labels/data'   
    stuff = np.load('../../images/test03/labels/data/L2RData_caution.npz', allow_pickle=True)
    data = stuff['data']
    xlabels =stuff['titles']
    box_list = stuff['box_list']
    info = stuff['info']

    prep_predict = collect_data.Preperation("{}".format(testing_dir), (416,416))
    images = prep_predict.images()
    bboxes = prep_predict.boxes()
    cutting_board_idx = prep_predict.board_index()
    data2 = collect_data.ImageData(images, bboxes, cutting_board_idx, 0)
    
    # define scoring function
    score = gaussian.GaussianDistribution(scipy=True, data=data, log=False)
    
    predictions = []
    for i in range(len(data2.images)-1):
        # Define the image to show, previous object, next object, and cutting board
        img = data2.images[i]
        last_obj = data2.food_boxes[i]
        plate = data2.boxes[data2.board_index[i], :]
        plate_dims = 0
        new_obj = data2.food_boxes[i+1]
        # instantiate prediction class
        pred = predict.Prediction(iterations, last_obj, plate, plate_dims,
                                  img, new_obj, score, None)
        # get coordinate with highest score
        mode = ['all']
        winning = pred.winner(mode=mode, n=n, logprob=True)
        predictions.append(winning)
        # plot results
        pred.plot_prediction(winning, pred.new_obj_width, pred.new_obj_height)

        # tranlate the mean values to absolute image location
        # and convert from meters to pixels
        mu_x = centers[0][0]*(1/pred.ratio) + pred.last_obj_centerx
        mu_y = centers[1][0]*(1/pred.ratio) + pred.last_obj_centery
        #convert from meters to pixels and add width of object to be placed
        # so we can see the entire object's preded position
        sigma_x = centers[0][0]*(1/pred.ratio) + pred.new_obj_width/2
        sigma_y = centers[1][0]*(1/pred.ratio) + pred.new_obj_height/2 + 100
        # heat map of the gaussians (only doing for dcx and dcy)
        heatmap.plot_heatmap(img, mu_x, mu_y, sigma_x, sigma_y)
        
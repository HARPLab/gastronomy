import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Reference: https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image

def twoD_Gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y):
    """
    Get an  equation for a 2-D Gaussian
    """
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-mu_x)**2) + c*((y-mu_y)**2)))

    return g.ravel()

def transparent_cmap(cmap, N=255):
    """
    Copy colormap and set alpha values (makes transparent color map)

    Note: decrease the 2nd number in linspace to increase transparency
    """
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.9, N+4)

    return mycmap

def plot_heatmap(image, mu_x, mu_y, sigma_x, sigma_y):
    """
    image (np.array): image to show
    mu_x: the x-mean/x position of prediction center
    mu_y: the y-mean/y position of prediction center
    simga_x: standard deviation is x direction
        (make bigger for better visuals)
    sigma_y: standard deviation is y direction
        (make bigger for better visuals)
    """
    #Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.Reds)

    # Import image and get x and y extents
    img = image
    img = img[:,:,::-1] #convert BGR to RGB
    # p = np.asarray(img).astype('float')
    w, h = img.shape[1], img.shape[0]
    y, x = np.mgrid[0:h, 0:w]

    #Plot image and overlay colormap
    plt.close()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    Gauss = twoD_Gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y)
    cb = ax.contourf(x, y, Gauss.reshape(x.shape[0], y.shape[1]), 15, cmap=mycmap)
    plt.colorbar(cb)
    plt.show()

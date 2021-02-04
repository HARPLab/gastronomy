import numpy as np
import torch
import torch.nn.init as init

"""
This script contains functions that you can apply to custom nn.Module
or nn.Sequential instances for weight initialization of a neural network.

To apply these initializations to your model insert the following in
your code after you have instantiated your model/
    model.apply(initialization_function)

Some initialization techniques:
https://pytorch.org/docs/stable/nn.init.html
"""

def general_uniform(model, layers=['Linear'], bounds=None, b=0.0):
    """
    Initializes weights to the range of 
    Inputs:
        model (nn.Module): your torch neural network model
        layers (list): list of strings of the name of the layers you want
            to apply initalization to
        bounds (list): list containing floats of the high and low values to 
            use for the uniform distribution. Uses [-y, y] by default, where
            y=1/sqrt(n) (n is the number of inputs to a given neuron)
        bias (float): value to initialize the bias weights to
    Ref: https://stackoverflow.com/a/55546528
    """
    if bounds is not None:
        assert len(bounds) == 2
    classname = model.__class__.__name__
    for layer in layers:
        if classname.find(layer) != -1:
            if bounds is None:
                # get the number of the inputs
                n = model.in_features
                bounds = [-1.0/np.sqrt(n), 1.0/np.sqrt(n)]
            model.weight.data.uniform_(bounds[0], bounds[1])
            model.bias.data.fill_(b)

def general_normal(model, layers=['Linear'], std=None, mean=0.0, b=0.0):
    """
    Initializes weights with values sampled from a normal distribution
    Inputs:
        model (nn.Module): your torch neural network model
        layers (list): list of strings of the name of the layers you want
            to apply initalization to
        std (float): standard deviation to use for the normal distribution.
            Uses std=1/sqrt(n) by default (n is the number of inputs to a
            given neuron)
        mean (float): value to set as the mean of the normal distribution
        b (float): value to initialize the bias weights to
    Ref: https://stackoverflow.com/a/55546528
    """
    classname = model.__class__.__name__
    for layer in layers:
        if classname.find(layer) != -1:
            if std is None:
                y = model.in_features
                std = 1/np.sqrt(y)
            model.weight.data.normal_(mean, std)
            model.bias.data.fill_(b)

import numpy as np
import re
import itertools
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from textwrap import wrap
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

def MSE_accuracy(predicted, ground_truth, mu=0, std=1):
    """
    Mean squared error as the acuracy
    """
    with torch.no_grad():
        acc = F.mse_loss(predicted.detach(), ground_truth)
        acc = torch.sqrt(acc)

    return acc

def binary_acc(pred, labels, sigmoid=True):
    """"
    Calculate the average accuracy over the batch of predictions given.
    """
    with torch.no_grad():
        if sigmoid:
            pred = torch.sigmoid(pred)
        acc = (torch.round(pred) == labels).sum()
        acc = acc.type(torch.float32) / labels.size(0)  

    return acc

def multi_class_acc(output, labels, softmax=True):
    """
    Calculate the average accuracy over the batch of predictions given
    """
    with torch.no_grad():
        if softmax:
            output = torch.softmax(output, dim=1)
        predicted = output.max(dim=1)[1] # get the index of the max per sample
        accuracy = ((predicted == labels).sum()).float() / labels.size(0)

    return accuracy

def rel_accuracy(predicted, ground_truth, abs_tolerance=np.array([1e-3]*2), rel_tolerance=np.array([0.0]*2)):
    """Calculate the average accuracy of predictions given, considered correct if prediction is within tolerance of ground truth"""
    with torch.no_grad():
        assert predicted.ndim == 2 and ground_truth.ndim == 2 # assuming axis 0 is num_samples
        assert predicted.shape[1] == 2
        assert ground_truth.shape[1] == 2
        if not torch.is_tensor(abs_tolerance):
            abs_tolerance = torch.from_numpy(abs_tolerance)
        if not torch.is_tensor(rel_tolerance):
            rel_tolerance = torch.from_numpy(rel_tolerance)
        num_correct = torch.zeros(predicted.shape)
        num_correct[:,0] = torch.isclose(predicted[:,0], ground_truth[:,0], atol=abs_tolerance[0], rtol=rel_tolerance[0])
        num_correct[:,1] = torch.isclose(predicted[:,1], ground_truth[:,1], atol=abs_tolerance[1], rtol=rel_tolerance[1])
        num_correct = torch.prod(num_correct, dim=1) # only correct if all dims are correct
    
        return torch.sum(num_correct) / (predicted.shape[0])

def f1_acc(predictions, labels, sigmoid=True, mode='weighted'):
    """
    Calculate the F1 score of the predictions
    Ref:
     - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    with torch.no_grad():
        if sigmoid:
            predictions = torch.sigmoid(predictions)
        predictions = torch.round(predictions).cpu().numpy()
        labels = labels.cpu().numpy()
        acc = f1_score(labels, predictions, mode)
        return torch.from_numpy(np.array([acc]))

def f1_acc_multi_class(predictions, labels, softmax=True, mode='weighted'):
    """
    Gets the weighted average f1 score across classes
    Set mode to None to get the accuracy for each individual class
    """
    with torch.no_grad():
        if softmax:
            predictions = torch.softmax(predictions, dim=1)
        predictions = predictions.max(dim=1)[1].cpu().numpy()
        labels = labels.cpu().numpy()
        acc = f1_score(labels, predictions, average=mode)
        return torch.from_numpy(np.array([acc]))

def get_balanced_metrics(predictions, labels, beta=1.0, sigmoid=False, softmax=False):
    """
    Get the metrics for accuracy, precision, recall, f_score(f1 by default), and support
    These values are weighted by the number of occurences for evaluating imbalanced datasets
    """
    with torch.no_grad():
        if sigmoid:
            predictions = torch.sigmoid(predictions)
            predictions = torch.round(predictions).cpu().numpy()
        elif softmax:
            predictions = torch.softmax(predictions, dim=1)
            predictions = predictions.max(dim=1)[1].cpu().numpy()
        else:
            predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        bal_acc = balanced_accuracy_score(labels, predictions)
        precision, recall, f_score, support = precision_recall_fscore_support(labels, predictions, beta=beta, average='weighted')
        return bal_acc, precision, recall, f_score, support

def false_pos_error(predictions, labels, error, reference, error_mean=None,
                    error_std=None, sigmoid=False):
    """
    Calculate the average error (distance between pose 1 and pose 2) on
        predictions that are incorrectly labeled as positive
    Args:
        predictions (torch.tensor): The ouput of the neural network
        labels (torch.tensor): The ground truth class/binary labels for predictions
        error (torch.tensor): The error values to use when calculating the output error.
            Should have shape NxM, where N is the number of samples and M is the number
            of dimensions, so you'll have one error value per dimension per sample.
        reference (torch.tensor): The reference value to compare the error values to.
            So the error is calulated as (L2_norm(error)-reference)
        error_mean (torch.FloatTensor): the mean of the normalized error value
            Leave as None if the error values aren't normalized
        error_std (torch.FloatTensor): the standard deviation of the normalized error value
        sigmoid (bool): whether to apply sigmoid activation to predictions
    """
    with torch.no_grad():
        if sigmoid:
            predictions = torch.sigmoid(predictions)
            predictions = torch.round(predictions).cpu().numpy()
        else:
            predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        false_pos = (predictions == 1) & (labels == 0)
        indexes = np.nonzero(false_pos)[0]
        if indexes.shape[0] == 0:
            return 0.0
        else:
            err = error[indexes].cpu().numpy()
            if error_mean is not None:
                err = (err * error_std.cpu().numpy()) + error_mean.cpu().numpy()
            dist = (np.linalg.norm(err, axis=1) - reference)
            avg_error = dist.sum() / dist.shape[0]
            assert avg_error >= 0
            return avg_error


def plot_confusion_matrix(predictions, labels, class_labels=None, sigmoid=False, softmax=False, normalize=False):
    """
    Arg:
        predictions (np.array): These are you predicted classifications 
        labels (np.array): ground truth labels for each sample
        class_labels (list): list of labels which will be used to display the axis labels

    Returns:
        matplotlib fig

    Other itema to note:
        - Currently, some of the ticks dont line up due to rotations.

    Ref:
        - https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
    """
    with torch.no_grad():
        if sigmoid:
            predictions = torch.sigmoid(predictions)
            predictions = torch.round(predictions).cpu().numpy()
        elif softmax:
            predictions = torch.softmax(predictions, dim=1)
            predictions = predictions.max(dim=1)[1].cpu().numpy()
        else:
            predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        if class_labels is None:
            class_labels = list(set(labels))
            classes = class_labels
        else:
            classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in class_labels]
            classes = ['\n'.join(wrap(l, 40)) for l in classes]

        cm = confusion_matrix(labels, predictions, labels=class_labels)
        if normalize:
            cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')
        num_classes = len(class_labels)

        fig = plt.figure(figsize=(num_classes, num_classes), dpi=100, facecolor='w', edgecolor='k')
        ax = plt.subplot(1, 1, 1)
        plt.imshow(cm, cmap='Oranges')

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", verticalalignment='center', color= "black")
        fig.tight_layout()

        return fig

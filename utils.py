#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:33:21 2020

@author: calmac

Adapted from Jupyter notebook style (Colab). 

"""

import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import matplotlib.pyplot as plt
import torch

 
##########################################################################
#               Performance functions 
##########################################################################
def compute_perform_stats(probs, label, n_classes):
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(label, preds)
    precisions = precision_score(label, preds, average=None,
                                 labels=range(n_classes), zero_division=0.)
    recalls = recall_score(label, preds, average=None, labels=range(n_classes),
                           zero_division=0.)
    f1 = f1_score(label, preds, average=None, labels=range(n_classes),
                           zero_division=0.)
    perform_stats = {'accuracy': accuracy, 'precision': precisions,
                     'recall': recalls, 'f1': f1}
    return perform_stats  

def print_progress(epoch=None, n_epoch=None, n_iter=None, iters_one_batch=None,
                   mean_loss=None, cur_lr=None, metric_collects=None,
                   prefix=None):
    """
    Print the training progress.
    :epoch: epoch number
    :n_epoch: total number of epochs
    :n_iter: current iteration number
    :mean_loss: mean loss of current batch
    :iters_one_batch: number of iterations per batch
    :cur_lr: current learning rate
    :metric_collects: dictionary returned by function calc_multi_cls_measures
    :returns: None
    """
    accuracy = metric_collects['accuracy']
    precisions = metric_collects['precisions']
    recalls = metric_collects['recalls']

    log_str = ''
    if epoch is not None:
        log_str += 'Ep: {0}/{1}|'.format(epoch, n_epoch)

    if n_iter is not None:
        log_str += 'It: {0}/{1}|'.format(n_iter, iters_one_batch)

    if mean_loss is not None:
        log_str += 'Loss: {0:.4f}|'.format(mean_loss)

    log_str += 'Acc: {:.4f}|'.format(accuracy)
    templ = 'Pr: ' + ', '.join(['{:.4f}'] * 2) + '|'
    log_str += templ.format(*(precisions[1:].tolist()))
    templ = 'Re: ' + ', '.join(['{:.4f}'] * 2) + '|'
    log_str += templ.format(*(recalls[1:].tolist()))

    if cur_lr is not None:
        log_str += 'lr: {0}'.format(cur_lr)
    log_str = log_str if prefix is None else prefix + log_str
    print(log_str)


##########################################################################
#            my functions 
##########################################################################
   
def split_dataset(dataset, val_percent, test_percent):
    dataset = list(dataset)
    length = len(dataset)
    n_val = int(length * val_percent)
    n_test = int(length * test_percent)
    n_train = length - n_val - n_test
#    random.shuffle(dataset) # not anymore: done it in matlab to make it easier! # need to shuffle so that the training, val, and test contain examples of all difficulties of image 
    return {'train': dataset[:-(n_val+n_test)], 'val': dataset[(n_train) : (length-n_test)], 'test': dataset[-n_test:]} 
    # That bit does the actual splitting.
#         Train_set: get all from dataset [:] except (-) end bits (n_val+n_test)
#         Val_set: get the examples after training [n_train:] but not the last bit [length-n_test]
#         Test_set: get only the n examples from the end of the dataset [-n_test:].
    
def split_train_val(dataset, val_percent):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
#    random.shuffle(dataset)  
    return {'train': dataset[:-n], 'val': dataset[-n:]} # train = all data minus number of validation examples
                                                        # val   = the remaining number of examples

# Custom function for displaying images 
def imshow(img, title=None):
    npimg = img.numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * npimg + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


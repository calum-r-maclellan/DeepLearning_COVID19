#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to guide the dataloading process, and to clean up dataset.py.

Created on Wed 1 Oct, 2020.
Adapted from Jupyter notebook style (Colab). 

@author: calmac
"""

# Standard packages for reading/processing data 
import numpy as np
from numpy.random import default_rng
import os
import cv2

# PyTorch
import torch

# My stuff
import settings
args = settings.parse_arguments()

##########################################################################
#                           Helper functions  
##########################################################################    
        
""" Convert string labels to OHE versions """ 
# Take in csv/txt file, split, and extract 3rd column (which contains the str labels of pathologies)
# Next, encode these strings as integers using pathology_mapping. 
# Send to _one_hot_encode(). 
# Return OHE labels to __init__().
# choose to return either int_labels for in-built loss (here: nn.CrossEntropyLoss), or ohe_labels for custom loss.
def _convert_labels(csvfile):
    int_labels, ids = _get_int_labels(csvfile)
    ohe_labels = _one_hot_encode(int_labels)
    return int_labels, ids # return chosen label format along with patient ids for counting later

""" Take in .txt file, split apart, and return 3rd column as integer labels """
def _get_int_labels(csvfile):
    pathology_mapping = { 'normal': 0, 'pneumonia': 1, 'COVID-19': 2 } # assign integers to pathology labels
    int_labels = np.empty(len(csvfile)).astype(np.int)  # initialise storage array
    ids = [] 
    for i, patient_i in enumerate(csvfile):             # for each patient in our file list, get index i and string of info for that patient
        patient_list = patient_i.split()                # extract ith patient info as a list using split()
        pathology = patient_list[2]                     # get pathology name (in 3rd column)
        int_labels[i] = pathology_mapping[pathology]    # now map that name to its corresponding int_label
        # Count patients: use id since some patients have multiple images
        ids.append(patient_list[0]) 
      
    return int_labels, ids  # return int_labels for all pathologies

""" Return one hot encoded equivalents of integer labels from _get_int_labels(). """
def _one_hot_encode(int_labels):
    labels = torch.from_numpy(int_labels)
    return F.one_hot(labels, n_classes)

""" Stack pixels over RGB channels and resize """
def _config_images(image):
    if len(image.shape) != 3:                             # if image is grayscale (H,W)
      image = np.stack((image, image, image), axis=2)     # stack to give new shape: (H, W, 3)
    elif image.shape[2] == 4: 
      imgray = image[:, :, 0]                             # remove colour channel
      image = np.stack((imgray, imgray, imgray), axis=2)  # new shape: (H, W, 3)
    return image

""" Read .txt file and store as list """
def _process_txt_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

"""Rescales images to be [0, 255]."""
def denormalize(image, maxval):
    return image.astype(np.float32) * maxval

""" For resizing the image without needing to convert to PIL image (as with torchvision.transforms.Resize()) """
class Xray_resize(object):
    def __init__(self):
      self.size = args.input_resize
    def __call__(self, img):
      return cv2.resize(img, (self.size, self.size))

""" Scales image pixels from [0,255] to [0,1] """      
class rescale(object):
    def __init__(self):
      self.maxval = 255
    def __call__(self, img):
      return torch.div(img,self.maxval)

##########################################################################
#      Functions for balancing the dataset relative to COVID sample size
########################################################################## 
class balanceToCovid(data.Dataset):
    def __init__(self, csv_path):
        
        self.csvfile = _process_txt_file(csv_path) 
        self.labels, self.ids = _convert_labels(self.csvfile)
        self.n_covids = count_covids(self.labels)
        self.all_files, self.n_files = self._getfiles()

    def _getfiles(self): 
        y = self.labels 
        normal_files = []
        pneum_files = []
        covid_files = []
        for j in range(len(y)):
          if y[j]==0: # if the label at row j is 'normal'
            normal_files.append(self.csvfile[j].split()[1])  # append normal_files with filename at that row 
          elif y[j]==1: # if the label at row j is 'pneum'
            pneum_files.append(self.csvfile[j].split()[1]) 
          elif y[j]==2: # if the label at row j is 'covid'
            covid_files.append(self.csvfile[j].split()[1]) 
        
        # Store files as separate keys in a dict()
        all_files = {'normal': normal_files, 'pneum': pneum_files, 'covid': covid_files}
        n_files = [len(all_files[x]) for x in ['normal', 'pneum', 'covid']]

        return all_files, n_files
        
    def __call__(self):  
        # clunky way: 
        # Create lists for storing randomly selected 265 files of each class, 
        # and add all lists together at the end to form our balanced dataset.
        # TODO: there will be a slicker way to do this, but get it working first.
        x_n = [] 
        x_p = []
        y_list = []
        rng = default_rng()
        for i in range(n_classes):
          rand_idx = rng.choice(range(self.n_files[i]), self.n_covids, replace=False)  # choose random instances
          if i==0:
            [x_n.append(self.all_files['normal'][j]) for j in rand_idx]
          elif i==1:
            [x_p.append(self.all_files['pneum'][j]) for j in rand_idx]
          X_list = x_n + x_p + self.all_files['covid']   # all filenames
          y_list = np.concatenate((np.zeros(self.n_covids),np.ones(self.n_covids),np.full(self.n_covids,2)), axis=None)
        
        return X_list, y_list  

#  Count patients by checking for duplicate ids
def countPatients(ids):
    patientCount = 0
    for i in range(0, len(ids)):
        if i==0:
            patientCount += 1
            continue
        if ids[i]==ids[i-1]:
            continue
        else: 
            patientCount += 1
    return patientCount

#  Count covid patients using class label
def count_covids(labels):
    n_covids = 0
    for (i, y) in enumerate(labels):
        if y == 2:
            n_covids += 1 # only count covid images
        else:
            continue
    return n_covids

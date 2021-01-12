#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 1 Oct, 2020

@author: calmac
"""

# Standard packages for reading/processing data 
import numpy as np
from numpy.random import default_rng
import os
import cv2
from PIL import Image
import skimage
from skimage.io import imread
from imgaug import augmenters as iaa

# PyTorch libraries
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as tfs

# My libraries
import settings
import dataset_helpers
args = settings.parse_arguments()

##########################################################################
#                           X-ray data reader  
##########################################################################

class COVIDx_Dataset(data.Dataset):
    def __init__(
            self,
            image_dir,
            txt_file,
            transform, 
            augment,
            n_imgs=None
            ):
        
        """ Initialise some things """
        self.imgpath = image_dir
        self.csv_file = dataset_helpers._process_txt_file(txt_file)       # read .txt file containing image info.
        self.labels, self.ids = dataset_helpers._convert_labels(self.csv_file)  # convert labels from str to int
        self.transform = transform
        self.augment = augment
        self.n_classes = args.n_classes
        self.n_channels = args.in_channels

    def __len__(self):
        return len(self.csv_file)
    
    def _affineImage(self,img):
        
        # Rotate, translate, and resize
        affine = tfs.Compose([
                tfs.ToPILImage(),
                tfs.RandomAffine(degrees=(-15,15), 
                                 translate=(0.05,0.05), 
                                 scale=(0.95,1.05)),
        ])
        return np.array(affine(img))
    
    def _intensityShift(self,img):
        img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
        img = cv2.GaussianBlur(img, (args.gaussian_blur,
                                     args.gaussian_blur), 0)
        return img

    def _transformImage(self,img):
        transform = tfs.Compose([
                  dataset_helpers.Xray_resize(),
                  tfs.ToTensor(), 
                  tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) 
        return transform(img)

    def __getitem__(self, idx):

        """ Get image using info from spreadsheet """
        samples = self.csv_file[idx].split()                # get full line of data at row idx 
        filename = samples[1]                               # get filename from 2nd column
        img_path = os.path.join(self.imgpath, filename)     # get image with that filename
        imgs = imread(img_path)
        imgs = dataset_helpers._config_images(imgs)

        # Apply data augmentation if requested
        if self.augment:
          imgs = self._affineImage(imgs)
          imgs = self._intensityShift(imgs)
        
        # Apply transforms
        if self.transform:  
          imgs = self._transformImage(imgs)

        """ Get labels and patient ids """
        labels = self.labels[idx]
        ids = self.ids[idx]
        
        return imgs, labels, ids, filename # return filename for GradCam step 
    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
settings.py: contains the settings needed for setting up the model (parameters etc)
             path to datasets, save directory locations

Created on Thu May 21 11:58:15 2020

@author: calmac
"""

import argparse

def parse_arguments():
    """
    Argument Parser for the commandline argments
    :returns: command line arguments

    """
    ##########################################################################
    #                            Training settings                           #
    ##########################################################################
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--momentum',           type=float,     default=0.999,          help='momentum in optimizer: ')
    parser.add_argument('--gamma',              type=float,     default=0.1,            help='factor to reduce lr by')
    parser.add_argument("--beta1",              type=float,     default=0.9,            help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2",              type=float,     default=0.999,          help="adam: decay of 2nd order momentum of gradient")
    parser.add_argument('--lr',                 type=float,     default=1e-3,           help='factor to reduce the loss gradients by. The larger this is, the less influence gradients have on weight updates per iteration.')
    parser.add_argument('--lr_scheduler',       type=str,       default='step',         help='')
    parser.add_argument('--num_epochs',         type=int,       default=30,             help='max number of training epochs')
    parser.add_argument('--batch_size',         type=int,       default=64,             help='')
    parser.add_argument('--num_workers',        type=int,       default=1,              help='')    
    parser.add_argument('--input_resize',       type=int,       default=224,            help='(H,W) size we want all images to be')
    parser.add_argument('--in_channels',        type=int,       default=3,              help='factor to reduce lr by')
    parser.add_argument('--n_classes',          type=int,       default=3,              help='number of pathologies we want to classify. For us, we want 3: healthy, other pneumonia (Viral and Bact.), and COVID-19') 
    
    # Data Directories
    parser.add_argument('--dataset',            type=str,       default="COVIDx",             help='option for selecting specific dataset')
    parser.add_argument('--root_path',          type=str,       default="./Datasets/ChestXray_Datasets/COVIDxv3/data",             help='path to where data lives')
    
    # Output directories
    #parser.add_argument('--output_path',        type=str,     default="./Experiments/COVIDxv3",      help='file location for storing results and model weights')

    # Others
    parser.add_argument('--patience',           type=int,       default=9,              help='number of epochs we allow validation loss to increase before introducing early stopping mechanism.')
    parser.add_argument('--log-every',          type=int,       default=2,              help='')
    parser.add_argument('--save-model',         type=bool,      default=True,           help='')
    parser.add_argument('--data_aug',           type=bool,      default=False,          help='')
    parser.add_argument('--covid_percent',      type=float,     default=0.3,            help='Percentage of covid samples in batch. Useful for testing semi-supervised model on varying amounts of training data.')
    parser.add_argument('--model_type',         type=str,       default='resnet18’,     help='choice of baseline classifier.')

    args = parser.parse_args()

    return args

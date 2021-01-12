#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:33:34 2020

@author: Calum

Adapted from Jupyter notebook style (Colab). 

Description: 
train.py: main script containing functions needed to carry out training steps
          of the model. 
                  
Code structure:
    - def main(): create loop for calling train and validate functions for each epoch
    - def train_loop(): 
        -> for loading each image/target pair using traindataloader 
                        and train dataset, and updating parameters in the model (gradients ON)
    - def validate_loop(): 
        -> for checking the models performance under parameter changes 
                           made by train_loop(), early stopping, etc (gradients OFF)
    - if name == main: 
        args = settings.get_settings() 
        main(args)
        -> for loading the settings from the settings.py script into main()
        
Updates:
    - 22/5/2020: 
        - comment out all validation-related code: focus on getting training loop
          working first before including validation tests.

@author: calmac
"""

##########################################################################
#                             Import packages                              
##########################################################################
 
""" System libaries """       
from __future__ import print_function
import os
import time

""" Main PyTorch libraries """
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

""" Torchvision"""
import torchvision
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

""" My stuff """
import dataset
import model
import settings
import utils
args = settings.parse_arguments()

""" Others """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


##########################################################################
#                             MAIN SCRIPT
# TODO:
# - assign validation data (80/20 train/test, with 10% val)
# - get validation loop working 
##########################################################################


# establish device 
device = torch.device(‘cuda’ if torch.cuda.is_available() else ‘cpu’)

def main(args):
    
    """
    Main function for the training pipeline

    INPUTS:
        - args: settings/arguments for model/training etc

    OUTPUTS:
        - 
    """
    ##########################################################################
    #                             Create directories                             
    ##########################################################################
        
    # Dataset paths
    global train_img_path, train_csv_path
    global test_img_path, test_csv_path
    root_path  = args.root_path   # root to where data is stored on computer
    train_img_path   = os.path.join(root_path, 'train')                  # image folder
    train_csv_path   = os.path.join(root_path, 'train_COVIDx_v3.txt')    # spreadsheet of image/patient meta datatest_image_dir = 'test'                
    test_img_path    = os.path.join(root_path, 'test')  
    test_csv_path    = os.path.join(root_path, 'test_COVIDx_v3.txt')    
    
    # Create folder to store model paths
    model_path = os.path.join(root_path, 'experiments')  
    os.makedirs(model_path, exist_ok=True)
    
    # Specific model paths 
    resnet18_dir = os.path.join(model_path, 'resnet18')  # create folder for storing ResNet18 models
    resnet50_dir = os.path.join(model_path, 'resnet50')  # create folder for storing ResNet18 models
    densenet161_dir = os.path.join(model_path, 'densenet161')  # create folder for storing ResNet18 models
    os.makedirs(resnet18_dir, exist_ok=True)
    os.makedirs(resnet50_dir, exist_ok=True)
    os.makedirs(densenet161_dir, exist_ok=True)
    
    # Create folder for model trained on the test data (smaller, lower performance)
    global train_test_model_dir
    train_test_model_dir  = os.path.join(resnet18_dir, 'train_on_test')   
    os.makedirs(train_test_model_dir, exist_ok=True)
    
    # Create folder for model trained on the training data (larger)
    global train_train_model_dir
    train_train_model_dir = os.path.join(resnet18_dir, 'train_on_train')  
    os.makedirs(train_train_model_dir, exist_ok=True)
    
    # Choose to train on either small (test) or large (training) datasets 
    train_log_root = os.path.join(train_test_model_dir, 'train_log')          # path to folder
    if not os.path.exists(train_log_root): os.mkdir(train_log_root)
    LOG_loss = open(os.path.join(train_log_root, 'lossPerEpoch.txt'), 'w')    # Text files for storing loss/acc results
    LOG_acc = open(os.path.join(train_log_root, 'accPerEpoch.txt'), 'w')
    
    # Helper functions for writing results to .txt files during train/testing 
    def log_string_loss(out_str):  
        LOG_loss.write(out_str+'\n')
        LOG_loss.flush()
    
    def log_string_acc(out_str):  
        LOG_acc.write(out_str+'\n')
        LOG_acc.flush()

    ##########################################################################
    #                               Datasets
    ##########################################################################
    
    """ Data transforms/augmentation """ 
    cxr_transform = transforms.Compose([
        Xray_resize(),                            
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    
   
    """ Training data """
    covidx_data = dataset.COVIDx_Dataset(train_img_path, train_csv_path, transforms=transform, is_training=True)
    train_dataloader = torch.utils.data.DataLoader(covidx_data, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)

    """ TODO: Validation data """
#    val_dataloader = torch.utils.data.DataLoader(covidx_data, batch_size=args.batch_size,
#                                             shuffle=True, num_workers=args.num_workers)

    # Sanity check
    print('Training model on train data ({} images).'.format(len(covidx_data)))
    
    
    ##########################################################################
    #                            Model settings  
    ##########################################################################

    """ Load model and send to available device """
    classifier = models.covidClassifier(args.in_channels, args.n_classes).cuda()
    
    """ Optimiser and learning rate decay function """
    optimiser = optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, len(dataloader))

    """ Loss function """
    loss_function = nn.CrossEntropyLoss()
    
    """ Other """
    best_val_loss = float('inf')
    best_val_accu = float(0)
    iteration_change_loss = 0
    
    ##########################################################################
    #                           Main training loop                           
    ##########################################################################
    
    """ Start epochs  """
    for epoch in range(args.num_epochs):
        
        print('Epoch {}/{} stats:'.format(epoch+1, args.num_epochs))

        #######################################################################
        #                       Training and validation steps                             
        #######################################################################
        
        """ Training loop """
        train_loop(classifier, train_dataloader, loss_function, optimiser, scheduler, perf_stats=True, device, args)
            
        
        """ TODO: Validation loop """
        # turn off gradients, and run a validation loop across all images 
        # in the validation dataset.
#        with torch.no_grad():
#            validation_loop(classifier, val_dataloader, loss_function, args)
#
        print('-' * 20)

        ############################################################
        #      TODO: Examine performance and save model checkpoints
        ############################################################

        """ Track validation results """
#        # Save model weights every time val_acc exceeds previous epoch acc.
#        if val_acc > best_val_accu:
#            best_val_accu = val_acc
#            torch.save(classifier, os.path.join(model_dir, 'best.pth'))
#       
#        # Check if current loss is better than previous epoch. If so, reset counter. 
#        if val_loss < best_val_loss:
#            best_val_loss = val_loss
#            iteration_change_loss = 0
#        
#        # If val_loss continues to increase for certain number of epochs, model overfitting -> stop training.
#        if iteration_change_loss == args.patience:
#            print(('Early stopping after {0} iterations without the decrease ' +
#                  'of the val loss').format(iteration_change_loss))
#            break
    
    """ end of training """
    time_elapsed = time.time() - t_start
    print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

    
    

##########################################################################
#                     Training loop for supervised model                             
##########################################################################
def train_loop(model, dataloader, loss_function, optimiser, scheduler, perf_stats, args):
            
    model.train()
    loss_list, acc_list, f1_list = [], [], []                                 
    
    # Iterate over data (suppress 'filename' return)
    for i, (inputs, labels, _) in enumerate(dataloader):

        # Assign as autograd variables and send to CPU/GPU
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        # Training functions
        optimiser.zero_grad()   # zero the parameter gradients
        outputs = model(inputs) # send images through model
        loss = loss_function(outputs, labels) # compute the loss for batch i
        loss.backward()    # backpropagate gradients 
        optimiser.step()   # update parameters 
        loss_list.append(loss.detach().cpu().numpy()) # append list with loss for image i

        # Performance stats: add results to lists for averaging over epoch
        y_probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
        y_trues = labels.detach().cpu().numpy()
        perform_stats = utils.compute_perform_stats(y_probs, y_trues)
        acc_list.append(perform_stats['accuracy'])
        f1_list.append(perform_stats['f1'])

    # Update learning rate scheduler and avg results over epoch
    scheduler.step()
    loss_epoch = np.mean(loss_list) # average the loss over all images in train dataset
    acc_epoch  = np.mean(acc_list)  # average accuracy over all images
    f1_epoch = np.mean(f1_list)
    
    print('Training Stats:')
    print('Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(loss_epoch, acc_epoch, f1_epoch))
    print('-' * 20)
    print()

    # Save model every epoch
    file_name = ('{}_epoch_{}_loss_{:.4f}_acc_{:.4f}_f1_{:.4f}.pth'.format(
                args.model_type, epoch+1, loss_epoch, acc_epoch, f1_epoch)
    )

    
    if args.save_model:
	torch.save(model.state_dict(), os.path.join(args.train_train_model_dir, file_name)) # save model weights to train_on_train folder
   
    # Write results to training logs
    log_string_loss( ('%f') % (loss_epoch) )
    log_string_acc(  ('%f') % (acc_epoch) )


#    return model.load_state_dict(best_model_wts)

##########################################################################
#                       Validation loop
# TODO:
#- fix write function to send results to validation txt file
                           
##########################################################################
def validation_loop(model, dataloader, perf_stats, device):
            
    model.train()
    loss_list, acc_list, f1_list = [], [], []                                 
    use_trainData=False
    
    # Iterate over data (suppress 'filename' return)
    for i, (inputs, labels, _) in enumerate(dataloader):

        # Assign as autograd variables and send to CPU/GPU
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs) # send images through model
        loss = loss_function(outputs, labels) # compute the loss for batch i 
        loss_list.append(loss.detach().cpu().numpy()) # append list with loss for image i

        # Performance stats: add results to lists for averaging over epoch
        y_probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
        y_trues = labels.detach().cpu().numpy()
        perform_stats = utils.compute_perform_stats(y_probs, y_trues)
        acc_list.append(perform_stats['accuracy'])
        f1_list.append(perform_stats['f1'])

    # Average results over epoch
    loss_epoch = np.mean(loss_list) # average the loss over all images in train dataset
    acc_epoch  = np.mean(acc_list)  # average accuracy over all images
    f1_epoch = np.mean(f1_list)
    
    print('Validation Stats:')
    print('Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(loss_epoch, acc_epoch, f1_epoch))
    print('-' * 20)
    print()

    # Save model every epoch
    file_name = ('{}_epoch_{}_loss_{:.4f}_acc_{:.4f}_f1_{:.4f}.pth'.format(
                args.model_type, epoch+1, loss_epoch, acc_epoch, f1_epoch)
    )

    if use_trainData:
       torch.save(model.state_dict(), os.path.join(args.train_train_model_dir, file_name)) # save model weights to train_on_train folder
    else:
       torch.save(model.state_dict(), os.path.join(args.train_test_model_dir, file_name))  # save model weights to train_on_test folder

    # Write results to training logs
    log_string_loss( ('%f') % (loss_epoch) )
    log_string_acc( ('%f') % (acc_epoch) )


#    return model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    args = settings.parse_arguments()
    main(args)
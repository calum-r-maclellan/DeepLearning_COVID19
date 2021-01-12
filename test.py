#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:33:40 2020

@author: calmac

Adapted from Jupyter notebook style (Colab). 

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
import torch.nn as nn
import torch.nn.functional as F

""" Torchvision """
import torchvision.transforms as transforms
import torchvision.utils as vutils

""" My packages """
import dataset
import models
import settings
import utils
import gradcams
args = settings.parse_arguments()

""" Others """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
        
    # Set up data
    test_covidx = dataset.COVIDx_Dataset(test_img_path, test_csv_path, transforms=True, augment=False)
    test_dataloader = torch.utils.data.DataLoader(test_covidx, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # Load pre-trained model
    model_weights =  'resnet50_epoch_30_loss_0.0453_acc_0.9966_f1_0.8613.pth.tar’
    path = os.path.join(train_test_model_dir, model_weights) # complete path to model weights
    model = models.covidClassifier(args.in_channels, args.n_classes)
    model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
        
    # Create f1 results .txt 
    file = open(os.path.join(train_test_model_dir, 'results.txt'),'w') 

    # Run a pass of images 
    test(model, test_dataloader, file)
    
    # Get gradcam saliency map results
    gradcam_result()

def test(model, test_dataloader, file):
    
    model.eval()
    y_preds, y_trues = [], []
         
    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(test_dataloader):
          inputs, labels = Variable(inputs), Variable(labels)
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
          pred = np.argmax(probs, axis=1)        # get predicted label
          label = labels.detach().cpu().numpy()  # get true label
          y_preds.append(pred)  
          y_trues.append(label)  
        
        # Now we have the probs and trues list for all test images, we can compute performance metrics
        results = utils.compute_perform_stats(y_preds, y_trues, args.n_classes)
        acc = np.array(results['accuracy'])
        f1 = np.array(results['f1'])
        recall = np.array(results['recall'])
        precision = np.array(results['precision'])
        
        # Print both average results (across each class) and inter-class results
        print('Mean Accuracy: {:.4f}, Mean F1-score: {:.4f}'.format(acc, np.mean(f1)))
        print()
        print('Normal stats:\n')
        print('\t Accuracy: {:.4f}, Recall: {:.4f}', 'Precision: {:.4f}'.format(precision[0], recall[0], f1[0]))
        print('Pneumonia stats:\n')
        print('\t Accuracy: {:.4f}, Recall: {:.4f}', 'Precision: {:.4f}'.format(precision[1], recall[1], f1[1]))
        print('COVID-19 stats:\n')
        print('\t Accuracy: {:.4f}, Recall: {:.4f}', 'Precision: {:.4f}'.format(precision[2], recall[2], f1[2]))
        
        # Save results to .txt files 
        file.write('Testing results on {} images:\n'.format(len(test_dataloader)))
        file.write('\tmean accuracy:  {} \n'.format(acc))
        file.write('\tmean precision: {} \n'.format(precision))
        file.write('\tmean recall:    {} \n'.format(recall))
        file.write('\tmean f1 score:  {} \n'.format(f1))
        file.write('Classes: normal, pneumonia, COVID-19')
        file.close()

    return y_trues, y_preds

def gradcam_result(dataset, classifier, n):
    
    # Make new dataloader with custom batch size for retreiving test images for CAM
    cam_dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers=args.num_workers)
    imgs, lbls, ids, filenames = next(iter(cam_dataloader))
    outputs, reals = models.classifier(imgs.to(device))
    gt_labels = lbls.detach().cpu().numpy()
    pred_labels = np.argmax(F.softmax(outputs,dim=1).detach().cpu().numpy(), axis=1)
    print(reals)
    # Compare predictions with ground truth labels
    print('True labels: '+str(gt_labels))
    print('Pred labels: '+str(pred_labels))
    
    # Get class instances 
    print('Normal: '+str(np.where(gt_labels==0)))
    print('Pneumonia: '+str(np.where(gt_labels==1)))
    print('COVID: '+str(np.where(gt_labels==2)))
    
    # Select image n from the dataloader, and send through network to activate neurons in penultimate conv layer
    img = imgs[n,:,:,:].unsqueeze(0).to(device) # get rescaled, ImageNet standardised image from dataloader
    lbl, pred_label = gt_labels[n], pred_labels[n] # get nth labels
    id, filename = ids[n], filenames[n]            # get nth patients info
    print('PatientID: '+str(id))
    print('True class= '+str(lbl))
    print('Pred class= '+str(pred_label))
    model_dict = dict(model_type='resnet', arch=classifier, layer_name='layer4_bottleneck2_conv3', input_size=(args.GAN_imgresize, args.GAN_imgresize))
    gradcam = gradcams.GradCAM(model_dict)
    gradcampp = gradcams.GradCAMpp(model_dict)
    
    # get a GradCAM saliency map on the class index of interest
    logit = classifier(img)
    mask1, logit1, gradients, activations, weights = gradcam(img, class_idx=pred_label) # get map of predicted class 
    mask2, logit2,_ = gradcampp(img, class_idx=pred_label)
    prob, index = torch.max(F.softmax(logit2,dim=1), 1)
    print(logit)
    print(F.softmax(logit2,dim=1))
    
    # Manually get the original images loaded by dataloader for using in gradcam 
    orig_img_path = os.path.join(test_img_path, str(filename)) # use img_path from above, and get image with that filename
    input_img = skimage.io.imread(orig_img_path)
    # Extract original image and resize 
    orig_img = cv2.resize(input_img,(args.GAN_imgresize,args.GAN_imgresize))
    # Scale pixels between [0,1], and send to np.stacking (see Loading the data)
    orig_img = dataset_helpers._config_images(orig_img/255) 
    
    # make heatmap from mask and synthesize saliency map using heatmap and the original image 
    heatmap1, cam_result1 = gradcams.visualize_cam(mask1, torch.from_numpy(orig_img).permute(2,0,1).unsqueeze(0))
    heatmap2, cam_result2 = gradcams.visualize_cam(mask2, torch.from_numpy(orig_img).permute(2,0,1).unsqueeze(0))
    gradcam_np, gradcampp_np = cam_result1.permute(1,2,0).numpy(), cam_result2.permute(1,2,0).numpy()
    
    # resize images to double input_resize
    orig_img = cv2.resize(orig_img,(args.GAN_imgresize*2,args.GAN_imgresize*2))
    gradcam_np = cv2.resize(gradcam_np,(args.GAN_imgresize*2,args.GAN_imgresize*2))
    gradcampp_np = cv2.resize(gradcampp_np,(args.GAN_imgresize*2,args.GAN_imgresize*2))
    
    # Show cam results
    plt.subplot(1, 3, 1).set_title('Input Chest X-ray')
    plt.imshow(orig_img, vmin=0, vmax=1, cmap='jet', aspect='auto')
    plt.subplot(1, 3, 2).set_title('Grad-CAM')
    plt.imshow(gradcam_np, vmin=0, vmax=1, cmap='jet', aspect='auto')
    plt.subplot(1, 3, 3).set_title('Grad-CAM++')
    plt.imshow(gradcampp_np, vmin=0, vmax=1, cmap='jet', aspect='auto')
    plt.colorbar()
    
    # Save images to .pngs
    # orig_img_result  = Image.fromarray((orig_img*255).astype(np.uint8)).convert('RGB')
    # gradcam_result   = Image.fromarray((gradcam_np*255).astype(np.uint8)).convert('RGB')
    # gradcampp_result = Image.fromarray((gradcampp_np*255).astype(np.uint8)).convert('RGB')
    # orig_img_result.save(os.path.join(train_train_model_dir,  'bs=64/epoch30_results/pneum',  ('{}_input_gt={}_pred={}.png'.format(id,lbl,pred_label))))
    # gradcam_result.save(os.path.join(train_train_model_dir,   'bs=64/epoch30_results/pneum',  ('{}_gradcam_gt={}_pred={}.png'.format(id,lbl,pred_label))))
    # gradcampp_result.save(os.path.join(train_train_model_dir, 'bs=64/epoch30_results/pneum',  ('{}_gradcam++_gt={}_pred={}.png'.format(id,lbl,pred_label))))




if __name__ == '__main__':
    args = settings.parse_arguments()
    main(args)
    
    
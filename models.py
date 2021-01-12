
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 1 Oct, 2020

@author: calmac
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import settings
args = settings.parse_arguments()

##########################################################################
#                            Supervised models: 
#               copy and paste the commented code for the model we want.
# 1. ResNet-18 
#  -> model = models.resnet18(pretrained=True)
# and
# 2. ResNet-50  
#  -> model = models.resnet50(pretrained=True)
# ------------
# Last layers:
# AdaptiveAvgPool2d: take all 512 FMs for each image in the batch (e.g. [4, 512] for bs=4), 
#                    which are (7x7), and average pool those into 1x1 array. This reduces
#                    our FM size from 7x7 to 1x1 for all 512 FMs for each image.
#                    Thus: input = [4, 512, 7, 7] -> avgpool -> output = [4, 512, 1, 1]
#                    Adaptive since we might not always know input size. Use PyTorch adaptive inferencing to
#                    determine correct kernel and stride sizes for pooling features. 
#
# Linear (Fully connected): after pooling conv features (7x7) to (1x1) score for all 512 Fms, map the 512Fms to 
#                           n_classes to give raw prediction for each class (pre-softmax)

# 3. CheXNet: Fill standard DenseNet-121 with pretrained CheXNet weights in main()
# ------------
#   -> model = torchvision.models.densenet121(pretrained=True)
#   where model weights are at: URL
# initialize and load the model
    # model = DenseNet121(N_CLASSES).cuda()
    # model = torch.nn.DataParallel(model).cuda()

    # if os.path.isfile(CKPT_PATH):
    #     print("=> loading checkpoint")
    #     checkpoint = torch.load(CKPT_PATH)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> loaded checkpoint")
    # else:
    #     print("=> no checkpoint found")
# 4. DenseNet 121 (Imagenet pretrained weights)
# ------------
#   ->  model = torchvision.models.densenet121(pretrained=True)

# 5. VGG-16/19: 
# ------------
#   ->  model = torchvision.models.vgg16(pretrained=False)
        # num_ftrs = model.classifier[6].in_features
        # [self.backbone.add_module(name, child) for name, child in model.named_children() if name!= 'classifier'] 

# Can either Fine-tune (fixed=False): update ImageNet weights for all layers, including new FC layer (requires_grad=True)
# or
# Fixed feature extractor (fixed=True): freeze entire network weights except our new FC layer,
# which will be the only weights updated during training (requires_grad=False).
# Note: params of newly constructed layers will be set to required_grad=True by default.    

##########################################################################

class covidClassifier(nn.Module):
    def __init__(self, in_channels, n_classes, model_type, fixed=False):
        super(covidClassifier, self).__init__()

        if model_type == 'densenet121':
          model = models.densenet121(pretrained=True)
          if fixed:
            for param in model.parameters():
              param.requires_grad = False
          num_ftrs = model.classifier.in_features
          # weights = input# user input: ask if chexnet or imagenet
    
          self.backbone = nn.Sequential()   
          [self.backbone.add_module(name, child) for name, child in model.named_children() if name!= 'classifier'] 
          self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) # add Global average pooling (missing for some reason)
          self.classifier = nn.Linear(num_ftrs, n_classes)    

        elif model_type == 'vgg16':
          model = models.vgg16_bn(pretrained=True)
          if fixed:
            for param in model.parameters():
              param.requires_grad = False
          vggFC_in = model.classifier[0].in_features    # flat: 512*7*7=25088
          vggFC_last = model.classifier[6].in_features  # 4096
          self.backbone = nn.Sequential()   
          # create custom classifier block for n_classes
          self.VggClassifier = nn.Sequential(
                nn.Linear(vggFC_in, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5,inplace=False),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5,inplace=False),
              # My FC layer for 3 classes
                nn.Linear(vggFC_last, n_classes),
          )

        elif model_type == 'resnet18':
          model = models.resnet18(pretrained=True)
          if fixed:
            for param in model.parameters():
              param.requires_grad = False
          num_ftrs = model.fc.in_features                 
          self.backbone = nn.Sequential()   
          [self.backbone.add_module(name, child) for name, child in model.named_children() if name!= 'fc'] 
          self.fc = nn.Linear(num_ftrs, n_classes)

        elif model_type == 'resnet50':
          model = models.resnet50(pretrained=True)
          if fixed:
            for param in model.parameters():
              param.requires_grad = False
          num_ftrs = model.fc.in_features                 
          self.backbone = nn.Sequential()             
          [self.backbone.add_module(name, child) for name, child in model.named_children() if name!= 'fc'] 
          self.fc = nn.Linear(num_ftrs, n_classes)
       
    def forward(self, batch):        
      # Send through main body of network
        x = self.backbone(batch) 
  
      # Output:
        if model_type == 'densenet121':
          x = self.avgpool(x)
          flat = x.view(x.size(0), -1)
          output = self.classifier(flat)

        elif model_type == 'vgg16':
          flat = x.view(x.size(0), -1)
          output = self.VggClassifier(flat)  

        elif (model_type == 'resnet18') or (model_type == 'resnet50'):
          flat = x.view(x.size(0), -1)
          output = self.fc(flat)   

        return output

# Look under bonnet to make sure layer names preserved and new fc layer properly assigned
model = covidClassifier(in_channels, n_classes, model_type=model_type)
print(model)
# If CheXNet, then load Densenet121 with ChexNet weights 
# checkpoints = torch.load(cpt_path)
# model.load_state_dict(checkpoints['state_dict'])

# Count number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Total number of parameters: {:4.2f} M'.format(pytorch_total_params / 1e6))

   

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:10:27 2022

@author: sazzouzi
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import *
import os
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import torchvision

data_dir= '/home/cyrilhannier/Bureau/ts228-15691'
batch_size= 25
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize([int(480/2),int(640/2)]),
    #transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'imgs_train'), transform= TRANSFORM_IMG), batch_size, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_iter))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img.permute(1,2,0),cmap="gray") #permute: to put the channels as the last of dimensions
print(f"Label: {label}")

#%% CNN with RESNET18

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
outputs = resnet(train_features)

outputs_lab= torch.argmax(outputs,dim=1)
print (outputs.size())     

#%% Loss function
epochs= 100
Loss= np.zeros(epochs) 
criterion= nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr= 0.001)
#retain_graph=True
for k in range(epochs):
    outputs = resnet(train_features)
    loss= criterion(outputs, train_labels) 
    optimizer.zero_grad()
    loss.backward(retain_graph= True)
    Loss[k]= loss
    print('loss:', loss.item())
    optimizer.step()
    

#%%
# Example of target with class indices
#loss = nn.CrossEntropyLoss()
#input = torch.randn(3, 5, requires_grad=True)
#target = torch.empty(3, dtype=torch.long).random_(5)
#output = loss(input, target)
#output.backward()
# Example of target with class probabilities
#input = torch.randn(3, 5, requires_grad=True)
#target = torch.randn(3, 5).softmax(dim=1)
#output = loss(input, target)
#output.backward()
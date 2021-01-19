# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
# https://discuss.pytorch.org/t/how-to-group-bunch-of-layers-together/51188

import os
from PIL import Image
import cv2
import csv
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# skimage.metrics.structural_similarity
#from skimage.measure import compare_ssim

class MySubmodule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MySubmodule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1) # again...
        self.pool = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class Net(nn.Module):
    """ The model expects a source of 256x256 pixels
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)  # notice the padding
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1) # again...
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1) # again...
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1) # again...
     
        self.feat0 = MySubmodule(1, 256)
        self.feat1 = MySubmodule(256, 128)
        self.feat2 = MySubmodule(128, 64)
        self.feat3 = MySubmodule(64, 32)
        
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.pool4 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(2304 * 32, 1024) # it is 64....
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.relu1 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.relu2 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p=0.5)
        self.relu3 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(p=0.5)
        self.relu4 = torch.nn.ReLU()
    def forward(self, x):
        #print('x0:{}'.format(x.shape))
        #x = self.conv1(x)
        #print('x0:{}'.format(x.shape))
        #x = self.relu1(x)
        #print('x0:{}'.format(x.shape))
        #x = self.pool1(x)
        x = self.feat0(x)
        #print('x0:{}'.format(x.shape))
        #x2 = self.conv2(x) 
        #print('x0:{}'.format(x.shape))
        #x2 = self.relu2(x2)
        #print('x0:{}'.format(x.shape))
        #x2 = self.pool2(x2) 
        #print('x0:{}'.format(x.shape))
        x2 = self.feat1(x)
        #print('x:{}'.format(x.shape))
        #print('x2:{}'.format(x2.shape))

        y = x.view(x.size(0), -1)
        z = x2.view(x2.size(0), -1)
        out = torch.cat((y, z), 1)
        out = torch.reshape(out, (32,128,32,-1))
        #print('out:{}'.format(out.shape))


        #x = self.conv3(out) 
        #print('x0:{}'.format(x.shape))
        #x = self.relu3(x)
        #print('x0:{}'.format(x.shape))
        #x = self.pool3(x) 
        #print('x0:{}'.format(x.shape))
        x = self.feat2(out)

        #x = self.conv4(x) 
        #print('x0:{}'.format(x.shape))
        #x = self.relu4(x)
        #print('x0:{}'.format(x.shape))
        #x = self.pool4(x) 
        #print('x4:{}'.format(x.shape))
        x = self.feat3(x)

        #x = x.view(-1, 8192)#16 * 64*64) 
        #x = self.bn1(x)
        x = x.view(-1, 2304 * 32)#16 * 64*64) 
        #print('x0:{}'.format(x.shape))
        #x = self.fc1(x) 
        x = F.sigmoid(self.fc1(x))
        #print('x0:{}'.format(x.shape))
        x = F.dropout(x)
        #print('x0:{}'.format(x.shape))
        x= F.sigmoid(self.fc2(x))
        #x = F.dropout(x)
        x= F.sigmoid(self.fc3(x))
        #x = F.dropout(x)
        #print('x0:{}'.format(x.shape))
        prediction = self.fc4(x)
        #print('x0:{}'.format(prediction.shape))
        return prediction




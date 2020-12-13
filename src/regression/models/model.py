# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

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
from skimage.measure import compare_ssim



class NetColor(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)  # notice the padding
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1) # again...
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1) # again...
        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) # again...
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc1 = nn.Linear(8192 * 4, 512) # it is 64....
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        #print('x0:{}'.format(x.shape))
        x = self.conv1(x)
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x)
        #print('x0:{}'.format(x.shape))
        x = self.conv2(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x) 
        #print('x0:{}'.format(x.shape))

        x = self.conv3(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x) 
        #print('x0:{}'.format(x.shape))


        #x = x.view(-1, 1024)#16 * 64*64) 
        #x = self.bn1(x)
        x = x.view(-1, 8192 * 4)#16 * 64*64) 
        #print('x0:{}'.format(x.shape))
        #x = self.fc1(x) 
        x = F.relu(self.fc1(x))
        #print('x0:{}'.format(x.shape))
        x = self.dropout(x)
        #print('x0:{}'.format(x.shape))
        x= F.relu(self.fc2(x))
        #print('x0:{}'.format(x.shape))
        prediction = self.fc3(x) 
        #print('x0:{}'.format(prediction.shape))
        return prediction



class Net(nn.Module):
    """ The model expects a source of 256x256 pixels
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)  # notice the padding
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1) # again...
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1) # again...
        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) # again...
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm1d(num_features=8192)
        self.fc1 = nn.Linear(8192 * 8, 1024) # it is 64....
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.relu1 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.relu2 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p=0.5)
        self.relu3 = torch.nn.ReLU()
    def forward(self, x):
        #print('x0:{}'.format(x.shape))
        x = self.conv1(x)
        #print('x0:{}'.format(x.shape))
        x = self.relu1(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool1(x)
        #print('x0:{}'.format(x.shape))
        x = self.conv2(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu2(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool2(x) 
        #print('x0:{}'.format(x.shape))

        x = self.conv3(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu3(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool3(x) 
        #print('x0:{}'.format(x.shape))


        #x = x.view(-1, 8192)#16 * 64*64) 
        #x = self.bn1(x)
        x = x.view(-1, 8192 * 8)#16 * 64*64) 
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




class Net1(nn.Module):
    """ The model expects a source of 256x256 pixels
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)  # notice the padding
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # notice the padding
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # notice the padding
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # notice the padding
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # notice the padding
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # notice the padding
        self.pool3 = nn.MaxPool2d(2,2)

        self.avgpool1 = nn.AdaptiveAvgPool2d((32,32))

        self.fc1 = nn.Linear(256 * 32 * 32, 512) # it is 64....
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        #print('x0:{}'.format(x.shape))
        x = F.relu(self.conv11(x))
        #print('x0:{}'.format(x.shape))
        x = F.relu(self.conv12(x))
        #print('x0:{}'.format(x.shape))
        x = self.pool1(x)
        #print('x0:{}'.format(x.shape))
        x = F.relu(self.conv21(x))
        #print('x0:{}'.format(x.shape))
        x = F.relu(self.conv22(x))
        #print('x0:{}'.format(x.shape))
        x = self.pool2(x) 
        #print('x0:{}'.format(x.shape))
        x = F.relu(self.conv31(x))
        #print('x0:{}'.format(x.shape))
        x = F.relu(self.conv32(x))
        #print('x0:{}'.format(x.shape))
        x = self.pool3(x) 
        #print('x0:{}'.format(x.shape))

        x = self.avgpool1(x) 
        #print('x0:{}'.format(x.shape))

        #x = x.view(-1, 8192)#16 * 64*64) 
        #x = self.bn1(x)
        x = x.view(-1, 256 * 32 * 32)#16 * 64*64) 
        #print('x0:{}'.format(x.shape))
        #x = self.fc1(x) 
        x = F.sigmoid(self.fc1(x))
        #x = F.dropout(x)
        #print('x0:{}'.format(x.shape))
        x= F.sigmoid(self.fc2(x))
        #x = F.dropout(x)
        #print('x0:{}'.format(x.shape))
        prediction = self.fc3(x)
        #print('x0:{}'.format(prediction.shape))
        return prediction

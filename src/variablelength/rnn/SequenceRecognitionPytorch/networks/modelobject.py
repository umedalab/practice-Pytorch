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


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        print('td0:{}'.format(x.shape))
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        print('xx0:{}'.format(x_reshape.shape))
        print('yy0:{}'.format(y.shape))
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
        
class MySubmodule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MySubmodule, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=0) # again...
        self.batch = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #print('mx0:{}'.format(x.shape))
        x = self.conv(x)
        #print('mx1:{}'.format(x.shape))
        x = self.batch(x)
        #print('mx2:{}'.format(x.shape))
        return self.relu(x)


class NetBoh(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    """ link: https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier """
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        #print('x0a:{}'.format(x.shape))
        x = x.view(x.shape[0], 1, -1)
        #print('x0b:{}'.format(x.shape))
        #h0, c0 = self.init_hidden(x)
        #print('x0b:{}'.format(x.shape))
        #out, (hn, cn) = self.rnn(x, (h0, c0))
        out, _ = self.rnn(x)
        #print('x0c:{}'.format(out.shape))
        out = self.fc(out[:, -1, :])
        #print('x0d:{}'.format(out.shape))
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

class Net(nn.Module):
    """ The model expects a source of 256x256 pixels
    """
    def __init__(self):
        super(Net, self).__init__()
     
        # 1D CovNet for learning the Spectral features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=(3,))
        self.bn1 = nn.BatchNorm1d(128)
        self.maxpool1 = nn.MaxPool1d(kernel_size=1, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        # 1D LSTM for learning the temporal aggregation
        # if the total number of imput dimensions is 18 * 20
        #self.lstm = nn.LSTM(input_size=155, hidden_size=128, num_layers=2, dropout=0.3, batch_first=True)
        # if the total number of imput dimensions is 14 * 20
        self.lstm = nn.LSTM(input_size=5, hidden_size=128, num_layers=2, dropout=0.3, batch_first=True)

        # Fully Connected layer
        #self.fc3 = nn.Linear(128, 128)
        #self.bn3 = nn.BatchNorm1d(128)
        # Get posterior probability for target event class
        self.fc4 = nn.Linear(128, 64)#128, 3 (original 3 actions class)
        self.fc5 = nn.Linear(64, 2)#128, 3 (original 3 actions class)
        #self.timedist = TimeDistributed(self.fc4)
     
        #self.feat0 = TimeDistributed(MySubmodule(1, 64))
        #self.fc = nn.Linear(14160, 1)

    def forward(self, x):
        #print('x0a:{}'.format(x.shape))
        x = x.view(x.shape[0], 1, -1)
        #print('x0b:{}'.format(x.shape))
        x = self.conv1(x)
        #print('x0c:{}'.format(x.shape))
        x = self.bn1(x)
        #print('x0d:{}'.format(x.shape))
        x = self.maxpool1(x)
        #print('x0e:{}'.format(x.shape))
        x = self.dropout1(x)
        #print('x0f:{}'.format(x.shape))
        x, states = self.lstm(x)
        #print('x0g:{}'.format(x.shape))
        prediction = self.fc4(x[:, -1, :])
        prediction = self.fc5(prediction)
        #print('x0h:{}'.format(prediction.shape))
        return prediction



class NetOk(nn.Module):
    """ The model expects a source of 256x256 pixels
    """
    def __init__(self):
        super(Net, self).__init__()
     
        self.feat0 = MySubmodule(1, 64)
        self.fc = nn.Linear(30592, 3)

    def forward(self, x):
        #print('x0:{}'.format(x.shape))
        x = x.view(x.shape[0], 1, -1)
        #print('x0b:{}'.format(x.shape))
        x = self.feat0(x)
        #print('x1:{}'.format(x.shape))
        x = x.view(1, -1)
        #print('x2:{}'.format(x.shape))
        prediction = self.fc(x)
        return prediction

class NetBoh2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=40, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(40, 3)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 40)
        x = F.relu(self.fc1(x))
        print('x2:{}'.format(x.shape))
        return F.log_softmax(x)



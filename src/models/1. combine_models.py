# check https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383/2

import os
from PIL import Image
import cv2
import csv
import numpy as np
# https://docs.python.org/3/library/statistics.html
import statistics

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


class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        return x
    

class MyModelB(nn.Module):
    def __init__(self):
        super(MyModelB, self).__init__()
        self.fc1 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        return x


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x

def main():
    # Create models and load state_dicts    
    modelA = MyModelA()
    modelB = MyModelB()
    # Load state dicts (if any)
    #modelA.load_state_dict(torch.load(PATH))
    #modelB.load_state_dict(torch.load(PATH))

    model = MyEnsemble(modelA, modelB)
    x1, x2 = torch.randn(1, 10), torch.randn(1, 20)
    output = model(x1, x2)

    print('x:{}'.format((x1, x2)))
    print('o:{}'.format(output))

if __name__ == "__main__":
    main()

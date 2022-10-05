# check https://stackoverflow.com/questions/65216411/how-to-concatenate-2-pytorch-models-and-make-the-first-one-non-trainable-in-pyto

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
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2

# How to concatenate 2 pytorch models and make the first one non-trainable in PyTorch
def main():
    # Create models and load state_dicts    
    modelA = MyModelA()
    modelB = MyModelB()
    # Load state dicts
    #modelA.load_state_dict(torch.load(PATH))

    #modelA = MyModelA()
    #modelB = MyModelB()

    criterionB = nn.MSELoss()
    optimizerB = torch.optim.Adam(modelB.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        for samples, targets in dataloader:
            optimizerB.zero_grad()

            x = modelA.train()(samples)
            predictions = modelB.train()(samples)
    
            loss = criterionB(predictions, targets)
            loss.backward()
            optimizerB.step()

if __name__ == "__main__":
    main()

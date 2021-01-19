# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

# Export model as ONNX
# https://pytorch.org/docs/stable/onnx.html

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

from models.model import Net

def TransposeModel(Weight_Path, Model, height=448, width=448, ch=3):
    device = torch.device("cuda")
    Model.to(device)
    Model.load_state_dict(torch.load(Weight_Path))
    Model.eval()
    # jit へ変換
    traced_net = torch.jit.trace(Model, torch.rand(1, ch, height, width).to(device))
    # 後の保存(Save the transposed Model)
    traced_net.save('model_h{}_w{}_mode{}_cuda.pt'.format(height, width, ch))
    print('model_h{}_w{}_mode{}_cuda.pt is exported.'.format(height, width, ch))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
custom_model = Net().to(device)
print(custom_model)
# Set the model path
Weight_Path = 'my_model.pth'
# Set the model
TransposeModel(Weight_Path, custom_model, 256, 256, 1)
# Serialize the module
sm = torch.jit.script(custom_model)
sm.save("traced_model.pt")

# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
# https://github.com/pytorch/examples/blob/master/snli/train.py

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
import torch.onnx

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../../..')

from modelshardcoded.model import Net
from loader.CustomDataLoader import CustomImageThresholdDataset
from utils.utilsModel import UtilsModel

# Log
from utils import mytensorboard, myutils


def train_val_dataset(dataset, val_split=0.25):
    lengths = [int(len(dataset)*(1.0 - val_split)), int(len(dataset)*val_split)]
    dataset_train, dataset_validation = torch.utils.data.random_split(dataset, lengths)
    return dataset_train, dataset_validation

def main(args):

    # applying logging only in the main process
    # ### OUR CODE ###
    if myutils.is_main_process():
        # passing argparse config with hyperparameters
        mytensorboard.args = vars(args)
    # ### END OF OUR CODE ###

    # Hyper parameters
    hyper_param_epoch = 30
    hyper_param_batch_train = 1
    hyper_param_batch_test = 1
    hyper_param_learning_rate = 0.00001

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    # Model
    custom_model = Net().to(device)
        
    N, C = UtilsModel.network_inputsize(custom_model)
    print('N:{} C:{}'.format(N, C))
    
    print(custom_model)
    summary(custom_model, (1, 256, 256))
    # applying logging only in the main process
    # ### OUR CODE ###
    if myutils.is_main_process():
        dummy_input = torch.rand(1, 1, 256, 256, requires_grad=True).to(device)
        with torch.onnx.select_model_mode_for_export(custom_model, False):
            mytensorboard.logger.add_model(custom_model, dummy_input)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    args = parser.parse_args()

    if args.output_dir:
        myutils.mkdir(args.output_dir)

    main(args)

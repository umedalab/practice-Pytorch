import sys
import os

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from networks.modelaction import Net
from loader.CustomPoseDataLoaderFile import CustomPoseDataLoader
from loader.SkeletonFeaturesFile import SkeletonFeatures
# Log
from utils import mytensorboard, myutils
from utils.utilsModel import UtilsModel

def main(args):

    print('run create_image_regression_out.py to create the dataset for the elaboration')
    print('run tensorboard with the followin command:')
    print('tensorboard --logdir experiments')
    # applying logging only in the main process
    # ### OUR CODE ###
    if myutils.is_main_process():
        # passing argparse config with hyperparameters
        mytensorboard.args = vars(args)
    # ### END OF OUR CODE ###

    # Hyper parameters
    hyper_param_epoch = 200
    hyper_param_batch_train = 20
    hyper_param_learning_rate = 0.00001

    # Data Loaders
    expected_observed_length = 20
    data_loader.read_data_pose_from_measurementpy(
        '../../../data/datasets/actions/sitting/scene001.txt', label_id=2)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    main(args)

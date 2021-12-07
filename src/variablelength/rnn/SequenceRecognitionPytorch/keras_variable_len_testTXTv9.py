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

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import model_from_yaml

from networks.model import Net
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


    # load YAML and create model
    yaml_file = open('data/model_keras/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    model.load_weights("data/model_keras/model.h5")
    print("Loaded model from disk")
    print(model.summary(90))

    action_range = 20

    # Data Loaders
    data_loader = CustomPoseDataLoader(observed_lenght=20, data_set_path='data')
    # Get a training and validation set from the original dataset (need enough data)
    #train, validation = train_val_dataset(data_loader)
    # Data Loaders
    test_loader = DataLoader(data_loader, batch_size=1, shuffle=False)

    for i_batch, item in enumerate(test_loader):
        
        print('i:{} : {}'.format(i_batch, item['feature'].shape))
        
        Xp = item['feature'].numpy().reshape(1, 20, 24, 1)
        #predict
        res = model.predict(Xp, verbose=1)
        # extract the maximum value and index of the position of 
        # the classified object.
        # link: https://stackoverflow.com/questions/3989016/how-to-find-all-positions-of-the-maximum-value-in-a-list
        m = max(res[0])
        idx = [i for i, j in enumerate(res[0]) if j == m]
        print("{} res {} => {} =? {}\n".format(i_batch, res[0], idx, item['label']))
                
        

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

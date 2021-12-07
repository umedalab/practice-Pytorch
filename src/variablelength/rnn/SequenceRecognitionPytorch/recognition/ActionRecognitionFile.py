import sys
import os
import datetime

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

import collections
from tracking.CollectTrackerFile import CollectTracker

class ActionRecognition():
    ''' It performs action recognition over a skeleton sequence.
        It analyzes a single unique skeleton.
    '''
    def __init__(self, device, fname_actions, fname_model, max_len, connectivity, max_keypoints):
        self.actions = []
        self.fname_actions = fname_actions
        self.fname_model = fname_model
        self.max_len = max_len
        self.device = device
        self.connectivity = connectivity
        self.max_keypoints = max_keypoints
        self.read_actions(fname_actions)
        self.load_model(fname_model)
        self.collection_tracker = CollectTracker(max_len = max_len)
        print('</ActionRecognition>')
        
    def read_actions(self, fname_actions):
        lines = []
        with open(fname_actions) as f:
            lines = f.readlines()    

        s = len(lines)
        for i in range(0, s):
            # split the index
            if len(lines[i]) > 0:
                self.actions.append(lines[i][:-1])
        print('</read_actions>')

    def clean_timeout(self, threshold):
        self.collection_tracker.timeout_sec(threshold)

    def load_model(self, fname_model):
        # Device
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
        # Model
        #custom_model = Net(input_dim, hidden_dim, layer_dim, output_dim).to(device)
        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load(fname_model, map_location=torch.device(self.device)))
        self.model.eval() 
        #print(self.model)

    def get(self, index, pose, width, height):
        #print('posein[{}] [{}]:{}'.format(type(pose), pose.shape, pose))
        self.collection_tracker.add(index, pose)
        #self.collection_tracker.show()
        if self.collection_tracker.get_poses_len(index) >= self.max_len:
            # convert the pose in features
            all_features, label = SkeletonFeatures.get_pose_featurelabel(self.connectivity, self.max_keypoints, self.collection_tracker.get_poses(index), None, 0, self.max_len)
            all_features = all_features.reshape(1, -1)
            #print('all_features:{} label:{}'.format(all_features.shape, label))       
            # estimate the action
            features = torch.tensor(all_features, dtype=torch.float).to(self.device)
            outputs = self.model(features)
            o = torch.argmax(outputs).detach().to('cpu')
            # return the result
            return True, self.actions[o]
            
        return False, 0

def main(args):

    print('run create_image_regression_out.py to create the dataset for the elaboration')
    print('run tensorboard with the followin command:')
    print('tensorboard --logdir experiments')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Data Loaders
    expected_observed_length = 20
    data_loader = CustomPoseDataLoader(
        observed_length=expected_observed_length, data_set_path='data',
        fname_bodydescriptor='connectivityCOCO18.txt')
    data_loader.read_data_pose('datatraining/DemoHouse20200428/observer.txt')
    action = ActionRecognition(device, 'data/actions.txt', 'data/model_actions.pth',
                               expected_observed_length, data_loader.connectivity, data_loader.max_keypoints)

    index_human = 14
    for i in range(0, len(data_loader.poses2d)):
        res, action_str = action.get(index_human, data_loader.poses2d[i])
        #print('res:{} action_idx:{}'.format(res, action_idx))
        print('action_str[{}]:{}'.format(data_loader.labels[i], action_str))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    args = parser.parse_args()

    main(args)

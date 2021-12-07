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


from loader.CustomPoseDataLoaderFile import CustomPoseDataLoader
from loader.SkeletonFeaturesFile import SkeletonFeatures
# Log
from utils import mytensorboard, myutils
from utils.utilsModel import UtilsModel

from tracking.CollectTrackerFile import CollectTracker
import networks.modelanomaly

class AnomalyRecognition():
    ''' It performs action recognition over a skeleton sequence.
        It analyzes a single unique skeleton.
    '''
    def __init__(self, device, fname_model, max_len, connectivity, max_keypoints):
        self.fname_model = fname_model
        self.max_len = max_len
        self.device = device
        self.connectivity = connectivity
        self.max_keypoints = max_keypoints
        self.load_model(fname_model, connectivity)
        self.collection_tracker = CollectTracker(max_len = max_len)
        self.criterion = nn.L1Loss(reduction='sum').to(device)
        print('</AnomalyRecognition>')
        
    def clean_timeout(self, threshold):
        self.collection_tracker.timeout_sec(threshold)

    def load_model(self, fname_model, connectivity):
        # Model
        seq_len = self.max_len * len(connectivity)
        n_features = 1
        self.model = networks.modelanomaly.RecurrentAutoencoder(self.device, seq_len, n_features, 128)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(fname_model, map_location=torch.device(self.device)))
        self.model.eval() 
        #print(self.model)

    def get(self, index, pose, threshold):
        '''
            return True(enough pose?),True(over threshold),loss value
        '''
        #print('posein[{}] [{}]:{}'.format(type(pose), pose.shape, pose))
        self.collection_tracker.add(index, pose)
        #self.collection_tracker.show()
        if self.collection_tracker.get_poses_len(index) >= self.max_len:
            # convert the pose in features
            all_features, label = SkeletonFeatures.get_pose_featurelabel(self.connectivity, self.max_keypoints, self.collection_tracker.get_poses(index), None, 0, self.max_len)
            all_features = all_features.reshape(-1, 1)
            #print('all_features:{} label:{}'.format(all_features.shape, label))       
            # estimate the action
            features = torch.tensor(all_features, dtype=torch.float).to(self.device)

            outputs = self.model(features)
            loss = self.criterion(features, outputs)
            # threshold
            res = loss.detach().cpu().numpy()
            
            #print('self.collection_tracker.get_poses(index):{}'.format(self.collection_tracker.get_poses(index)))
            #print('all_features:{} res:{}'.format(all_features, res))
            
            if res < threshold:
                return True, False, res
            else:
                return True, True, res
            
        return False, False, 0
       

def main(args):

    print('run create_image_regression_out.py to create the dataset for the elaboration')
    print('run tensorboard with the followin command:')
    print('tensorboard --logdir experiments')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Data Loaders
    expected_observed_length = 2
    data_loader = CustomPoseDataLoader(observed_length=expected_observed_length, data_set_path='data', fname_bodydescriptor='connectivityCOCO18.txt')
    data_loader.read_data_pose('datatraining/DemoHouse20200428/observer.txt')
    anomaly = AnomalyRecognition(device, 'data/model_anomaly.pth', expected_observed_length, data_loader.connectivity, data_loader.max_keypoints)
    
    index_human = 14
    for i in range(0, len(data_loader.poses2d)):
        res, out, loss = anomaly.get(index_human, data_loader.poses2d[i], 2.5)
        #if data_loader.num_line[i] > 245:
        #    print('data_loader lines[{}]'.format(data_loader.num_line[i]))
        #    exit(0)
        print('res:{} out:{} loss:{}'.format(res, out, loss))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    args = parser.parse_args()

    main(args)

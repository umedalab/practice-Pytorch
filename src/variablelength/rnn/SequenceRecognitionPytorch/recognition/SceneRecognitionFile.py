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

import sys
sys.path.append("D:/workspace/programs/MyPractice/practice-Pytorch/src/variablelength/rnn/SequenceRecognitionPytorch")


from networks.modelobject import Net
from loader.CustomObjectDataLoaderFile import CustomObjectDataLoader
from loader.ObjectFeaturesFile import ObjectFeatures
from loader.ObjectLabelsFile import ObjectLabels
from loader.DataloaderFolder import DataloaderFolder
# Log
from utils import mytensorboard, myutils
from utils.utilsModel import UtilsModel

import collections
from tracking.CollectTrackerFile import CollectTracker

class SceneRecognition():
    ''' It performs action recognition over a skeleton sequence.
        It analyzes a single unique skeleton.
    '''
    def __init__(self, device, fname_actions, fname_model, max_len):
        self.actions = []
        self.fname_actions = fname_actions
        self.fname_model = fname_model
        self.max_len = max_len
        self.device = device
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
                self.actions.append(lines[i].strip())
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

    def get(self, index, feature):
        #print('posein[{}] [{}]:{}'.format(type(pose), pose.shape, pose))
        self.collection_tracker.add(index, feature)
        #self.collection_tracker.show()
        if self.collection_tracker.get_poses_len(index) >= self.max_len:
            # convert the pose in features
            all_features, label = ObjectFeatures.get_featurelabel_sequence(self.collection_tracker.get_poses(index), None, 0, self.max_len)
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
    expected_observed_length = 1
    data_loader = CustomObjectDataLoader(observed_length=expected_observed_length)
    # data from video customer
    # read the label file
    filters = [] # it filters the falling action (?)
    tdata = DataloaderFolder()
    objectlabel2id = dict()
    objectlabel2id['person'] = 0.333
    objectlabel2id['personsleep'] = 0.333
    objectlabel2id['bed'] = 0.666
    objectlabel2id['wheelchair'] = 0.999

    mypath = 'D:/workspace/programs/MyPractice/practice-Pytorch/src/variablelength/rnn/SequenceRecognitionPytorch/datatraining/object/002'
    res = tdata.get_pair(mypath)
    print('res:{}'.format(res))
    # data from video customer
    # read the label file
    for r in res:
        print('r:{}'.format(r))
        labels = ObjectLabels.read_label(r[1])
        data_loader.read_data_pose_from_measurementpy_id(r[0], labels, filters, objectlabel2id)
        print('len:{}'.format(data_loader.__len__()))

    scene = SceneRecognition(device, 'data/scenes.txt', 'data/model_scenes.pth',
                               expected_observed_length)

    index_human = 0
    for i in range(0, len(data_loader.feat)):
        res, scene_str = scene.get(index_human, data_loader.feat[i])
        #print('res:{} action_idx:{}'.format(res, action_idx))
        print('scene_str[{}]:{}'.format(data_loader.lab[i], scene_str))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    args = parser.parse_args()

    main(args)

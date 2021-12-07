import sys
import os

import numpy as np
import cv2

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
from recognition.ActionRecognitionFile import ActionRecognition


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

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data Loaders
    expected_observed_length = 20
    data_loader = CustomPoseDataLoader(
        observed_length=expected_observed_length, data_set_path='data',
        fname_bodydescriptor='connectivityCOCO14.txt') #connectivityCOCO18
    #data_loader.read_data_pose('datatraining/DemoHouse20200428/observer.txt')
    filters = [] # it filters the falling action (?)
    labels = CustomPoseDataLoader.read_label('/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/customerlabels/001.txt')
    data_loader.read_data_pose_from_measurementpy_id('/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/customer/001.txt', 0, labels, filters)
    action = ActionRecognition(device, 'data/actions.txt', 'data/model_actions.pth',
                               expected_observed_length, data_loader.connectivity, data_loader.max_keypoints)

    cap = cv2.VideoCapture('/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/media/videos/RitecsTaira/001.mp4')       
                            
    index_dummy = 14
    for i in range(0, len(data_loader.poses2d)):
        # draw the pose
        if cap is not None:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                #print('{}'.format(data_loader.poses2d[i]))
                for k in range(0, len(data_loader.poses2d[i])):
                    cv2.circle(frame, data_loader.poses2d[i][k], 5, (0, 0, 255), 2)
                cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break            
        # draw the result
        res, action_str = action.get(index_dummy, data_loader.poses2d[i])
        print('res:{} action_str:{}'.format(res, action_str))
        action.clean_timeout(5)

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

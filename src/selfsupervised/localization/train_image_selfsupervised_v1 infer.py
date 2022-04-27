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
#from skimage.measure import compare_ssim
#import torch.onnx

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.append('.')
sys.path.append('../../regression')

from selfsuperdataloader import SelfSupervisedLabelDataset

# Log
from utils import mytensorboard, myutils
from modellocalization import Net


def train_val_dataset(dataset, val_split=0.25):
    lengths = [int(len(dataset)*(1.0 - val_split)), int(len(dataset)*val_split)]
    dataset_train, dataset_validation = torch.utils.data.random_split(dataset, lengths)
    return dataset_train, dataset_validation

def main(args):

    print('It trains the localization of a patch on a map in a self-supervised mode')
    print('run tensorboard with the followin command:')
    print('tensorboard --logdir experiments')

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # applying logging only in the main process
    # ### OUR CODE ###
    if myutils.is_main_process():
        # passing argparse config with hyperparameters
        mytensorboard.args = vars(args)
    # ### END OF OUR CODE ###

    # Hyper parameters
    hyper_param_epoch = 100000
    hyper_param_batch_train = 3
    hyper_param_batch_test = 2
    hyper_param_learning_rate = 1e-4

    # Transformation functions
    transforms_train = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize((256, 256)),
                                           #transforms.RandomRotation(10.),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.Resize((256, 256)),
                                          transforms.ToTensor()])

    path_data = 'D:/workspace/university/chuo/umedalab/VisionTransformers/MyPractice/practice-Pytorch/data/classification/classes4/Train/car'
    train_data_set = SelfSupervisedLabelDataset(data_set_path_color=path_data, data_set_label=None, transforms=transforms_train, do_training=True)
    item = train_data_set.__getitem__(0)
    print('item:{}'.format(item))

    #a = item['image'].detach().to('cpu').numpy().transpose(1, 2, 0)
    #cv2.imshow('a',a)
    #cv2.waitKey(0)

    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch_train, shuffle=True)

    print('draw model')
    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    # Model
    out_dim = 2
    custom_model = Net(out_dim).to(device)
    #custom_model = Net(1, 10, out_dim).to(device)
    print(custom_model)
    #summary(custom_model, (1, 256, 256))
    # applying logging only in the main process
    # ### OUR CODE ###
    if False and myutils.is_main_process():
        dummy_input = torch.rand(1, 1, 256, 256, requires_grad=True).to(device)
        with torch.onnx.select_model_mode_for_export(custom_model, False):
            mytensorboard.logger.add_model(custom_model, dummy_input)

    # Save
    custom_model.load_state_dict(torch.load("my_model.pth"))
    custom_model.eval()

    # Loss and optimizer
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.HuberLoss()
    #optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.RMSprop(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.SGD(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.ASGD(custom_model.parameters(), lr=hyper_param_learning_rate)
    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)

    iterations = 0
    dev_every = 50


    show_summary = True

    print('start evaluate')
    # switch model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for item in train_loader:
            #print("item {}/n".format(item))
            images = item['image'].to(device)
            position = item['position']
            # https://www.geeksforgeeks.org/how-to-join-tensors-in-pytorch/
            position_stack = torch.stack((position[0], position[1]), -1).type(torch.FloatTensor).to(device)
            labels = torch.tensor(item['label']).to(device).float()
            outputs = custom_model(images)

            print('output:{} position_stack:{} labels:{}'.format(outputs, position_stack, labels))

            a = images[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            cv2.imshow('a',a)
            cv2.waitKey(1)

        print('total:{}'.format(total))

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

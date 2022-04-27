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

    print('start training')
    # Train (no validation test)
    for e in range(hyper_param_epoch):
        for i_batch, item in enumerate(train_loader):

            # switch model to training mode, clear gradient accumulators
            custom_model.train(); optimizer.zero_grad()
            
            iterations += 1

            images = item['image'].to(device)
            position = item['position']

            #print('position:{}'.format(position))
            #position_stack = torch.stack(position).type(torch.FloatTensor).to(device)
            #position_stack = torch.cat((position[0], position[1]), 0)
            # https://www.geeksforgeeks.org/how-to-join-tensors-in-pytorch/
            position_stack = torch.stack((position[0], position[1]), -1).type(torch.FloatTensor).to(device)

            a = images[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            cv2.imshow('a',a)
            cv2.waitKey(1)
            #print('images:{}'.format(images.shape))
            #print('position:{}'.format(position_stack))

            if show_summary:
                show_summary = False
                summary(custom_model, images[0].detach().to('cpu').numpy().shape)
        
            # Forward pass
            outputs = custom_model(images)
            #print('outputs:{} labels:{}'.format(outputs, labels))

            loss = criterion(outputs, position_stack) # classification
            # applying logging only in the main process
            if myutils.is_main_process():
                # let's track the losses here by adding scalars
                loss_dict = {'loss':loss}
                mytensorboard.logger.add_scalar_dict(
                    # passing the dictionary of losses (pairs - loss_key: loss_value)
                    loss_dict,
                    # passing the global step (number of iterations)
                    global_step=mytensorboard.global_iter,
                    # adding the tag to combine plots in a subgroup
                    tag="loss"
                )
                # incrementing the global step (number of iterations)
                mytensorboard.global_iter += 1

            # only if process images
            #print('position: {} | outputs:{} | loss:{}'.format(position, outputs, loss))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evaluate performance on validation set periodically
            if False and iterations % dev_every == 0:

                # switch model to evaluation mode
                custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for item in validation_loader2:
                        #print("item {}/n".format(item))
                        images = item['image'].to(device)
                        labels = torch.tensor(item['label']).to(device).float()
                        outputs = custom_model(images)

                        o = (int)(outputs.detach().to('cpu').numpy() * 255)
                        l = (int)(labels.detach().to('cpu').numpy() * 255)
                        print('output:{} labels:{}'.format(o.item(), l.item()))
                        total += abs(o - l)

                        a = images[0].detach().to('cpu').numpy().transpose(1, 2, 0)
                        #grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                        grayA = a * 255
                        ret,c = cv2.threshold(grayA,o,255,cv2.THRESH_BINARY_INV)
                        cv2.imshow('a',a)
                        cv2.imshow('c',c)
                        cv2.waitKey(1)

                    print('total:{}'.format(total))

            elif iterations % dev_every == 0:
                print('iterations:{} position: {} | outputs:{} | loss:{}'.format(iterations, position_stack, outputs, loss))
                if (i_batch + 1) % hyper_param_batch_train == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, hyper_param_epoch, loss.item()))

    # Save
    torch.save(custom_model.state_dict(), "my_model.pth")

    # Serialize the module
    sm = torch.jit.script(custom_model)
    sm.save("traced_model.pt")

    #TODO:
    #  Get a image crop
    #  Get the position and orientation respect that portion of image

    return
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
    #hyper_param_learning_rate = 0.00001
    hyper_param_learning_rate = 0.001

    # Transformation functions
    transforms_train = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize((256, 256)),
                                           transforms.RandomRotation(10.),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.Resize((256, 256)),
                                          transforms.ToTensor()])

    # Data Loaders
    #path_data = '../../../../../../datasets/DeepCrack'
    path_data = 'D:/datasets/Concrete/DeepCrack'
    train_data_set = CustomImageThresholdDataset(data_set_path_color=path_data + '/train_img', data_set_label="../CreateDataset/train_csvfile.csv", transforms=transforms_train, do_training=True)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch_train, shuffle=True)

    # Get a training and validation set from the original dataset
    train, validation = train_val_dataset(train_data_set)
    print('train:{}'.format(train))
    print('validation:{}'.format(validation))
    train_loader2 = DataLoader(train, batch_size=hyper_param_batch_train, shuffle=True)
    validation_loader2 = DataLoader(validation, batch_size=hyper_param_batch_train, shuffle=True)

    # Test set
    test_data_set = CustomImageThresholdDataset(data_set_path_color=path_data + '/test_img', data_set_label="../CreateDataset/test_csvfile.csv", transforms=transforms_test, do_training=True)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch_test, shuffle=False)

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    # Model
    custom_model = Net().to(device)
    print(custom_model)
    summary(custom_model, (1, 256, 256))
    # applying logging only in the main process
    # ### OUR CODE ###
    if myutils.is_main_process():
        dummy_input = torch.rand(1, 1, 256, 256, requires_grad=True).to(device)
        with torch.onnx.select_model_mode_for_export(custom_model, False):
            mytensorboard.logger.add_model(custom_model, dummy_input)


    # Loss and optimizer
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    #criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    optimizer = torch.optim.RMSprop(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.SGD(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.ASGD(custom_model.parameters(), lr=hyper_param_learning_rate)

    iterations = 0
    dev_every = 1000

    # switch model to training mode, clear gradient accumulators
    custom_model.train(); optimizer.zero_grad()
            
    # Train (no validation test)
    for e in range(hyper_param_epoch):
        for i_batch, item in enumerate(train_loader2):
        
            iterations += 1
        
            images = item['image'].to(device)
            labels = item['label']
            labels = torch.stack(labels).to(device)
            # Forward pass
            outputs = custom_model(images)
            #print('outputs:{} labels:{}'.format(outputs, labels))
            loss = criterion(outputs, labels) # classification
            # applying logging only in the main process
            if myutils.is_main_process():
                # let's track the losses here by adding scalars
                loss_dict = {'loss':loss}
                mytensorboard.logger.add_scalar_dict(
                    # passing the dictionary of losses (pairs - loss_key: loss_value)
                    loss_dict,
                    # passing the global step (number of iterations)
                    global_step=mytensorboard.global_iter,
                    # adding the tag to combine plots in a subgroup
                    tag="loss"
                )
                # incrementing the global step (number of iterations)
                mytensorboard.global_iter += 1

            # only if process images
            print('labels: {} | outputs:{} | loss:{}'.format(labels, outputs, loss))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            
            # evaluate performance on validation set periodically
            if iterations % dev_every == 0:

                # switch model to evaluation mode
                custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for item in validation_loader2:
                        #print("item {}/n".format(item))
                        images = item['image'].to(device)
                        labels = torch.tensor(item['label']).to(device).float()
                        outputs = custom_model(images)

                        o = (int)(outputs.detach().to('cpu').numpy() * 255)
                        l = (int)(labels.detach().to('cpu').numpy() * 255)
                        print('output:{} labels:{}'.format(o, l))
                        total += abs(o - l)

                        a = images[0].detach().to('cpu').numpy().transpose(1, 2, 0)
                        #grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                        grayA = a * 255
                        ret,c = cv2.threshold(grayA,o,255,cv2.THRESH_BINARY_INV)
                        cv2.imshow('a',a)
                        cv2.imshow('c',c)
                        cv2.waitKey(1)

                    print('total:{}'.format(total))

                # switch model to training mode, clear gradient accumulators
                custom_model.train(); optimizer.zero_grad()
            
            else:

                if (i_batch + 1) % hyper_param_batch_train == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, hyper_param_epoch, loss.item()))

    # Save
    torch.save(custom_model.state_dict(), "my_model.pth")

    # Serialize the module
    sm = torch.jit.script(custom_model)
    sm.save("traced_model.pt")

    # Test the model
    custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for item in test_loader:
            #print("item {}/n".format(item))
            images = item['image'].to(device)
            labels = torch.tensor(item['label']).to(device).float()
            outputs = custom_model(images)

            o = (int)(outputs.detach().to('cpu').numpy() * 255)
            l = (int)(labels.detach().to('cpu').numpy() * 255)
            print('output:{} labels:{}'.format(o, l))
            total += abs(o - l)

            a = images[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            #grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            grayA = a * 255
            ret,c = cv2.threshold(grayA,o,255,cv2.THRESH_BINARY_INV)
            cv2.imshow('a',a)
            cv2.imshow('c',c)
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

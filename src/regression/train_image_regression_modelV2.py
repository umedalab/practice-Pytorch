# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

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

from models.model import Net
from loader.CustomDataLoader import CustomImageThresholdDataset

# Log
from utils import mytensorboard, myutils

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

    # Transformation functions
    transforms_train = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize((256, 256)),
                                           transforms.RandomRotation(10.),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.Resize((256, 256)),
                                          transforms.ToTensor()])

    # Data Loaders
    train_data_set = CustomImageThresholdDataset(data_set_path_color="/home/moro/workspace/work/Todai/Concrete/DeepSegmentor/datasets/DeepCrack/train_img", data_set_label="train_csvfile.csv", transforms=transforms_train, do_training=True)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch_train, shuffle=True)

    test_data_set = CustomImageThresholdDataset(data_set_path_color="/home/moro/workspace/work/Todai/Concrete/DeepSegmentor/datasets/DeepCrack/test_img", data_set_label="test_csvfile.csv", transforms=transforms_test, do_training=True)
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
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    #criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    optimizer = torch.optim.RMSprop(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.SGD(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.ASGD(custom_model.parameters(), lr=hyper_param_learning_rate)

    # Train (no validation test)
    for e in range(hyper_param_epoch):
        for i_batch, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label']
            labels = torch.stack(labels).to(device)
            #labels = np.array(labels, dtype=np.float32)
            #print('>>{}'.format(labels))

            #labels = torch.from_numpy(np.asarray(labels))
            #labels = torch.FloatTensor(item['label'])#.to(device)
            #labels = torch.tensor(item['label']).to(device).float()

            #o = (int)(labels.detach().to('cpu').numpy() * 255)
            #if o == 0:
            #    continue

            #print('images:{}'.format(images))
            #print('labels:{}'.format(labels))

            # Forward pass
            outputs = custom_model(images)
            #print('outputs:{}'.format(outputs))
            loss = criterion(outputs, labels) # classification
            #l = torch.tensor([[0.5]], device='cuda:0')
            #print('l:{}'.format(l))
            #loss = criterion(outputs, l) # regression
            #print('loss:{}'.format(loss))

            #loss = hingeLoss(labels * 255, outputs, custom_model)

            # applying logging only in the main process
            # ### OUR CODE ###
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
            # ### END OF OUR CODE ###



            # only if process images
            #loss = my_loss(outputs, images, labels)
            #print('#images: {} | labels: {} | outputs:{} | loss:{}'.format(images.shape, labels, outputs, loss))
            print('labels: {} | outputs:{} | loss:{}'.format(labels, outputs, loss))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch + 1) % hyper_param_batch_train == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(e + 1, hyper_param_epoch, loss.item()))

    # Save
    torch.save(custom_model.state_dict(), "my_model.pth")

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

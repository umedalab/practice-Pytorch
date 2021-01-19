# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

import os
from PIL import Image
import cv2
import csv
import numpy as np
# https://docs.python.org/3/library/statistics.html
import statistics

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

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../../..')

from modelshardcoded.model import Net
from loader.CustomDataLoader import CustomImageThresholdDataset

def main():

    # Hyper parameters
    hyper_param_epoch = 10
    hyper_param_batch_train = 1
    hyper_param_batch_test = 1
    hyper_param_learning_rate = 0.00001

    # Transformation functions
    transforms_test = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.Resize((256, 256)),
                                          transforms.ToTensor()])


    # Data Loaders
    path_data = '../../../../../../datasets/DeepCrack'
    # Test set
    test_data_set = CustomImageThresholdDataset(data_set_path_color=path_data + '/test_img', data_set_label="../CreateDataset/test_csvfile.csv", transforms=transforms_test, do_training=True)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch_test, shuffle=False)

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    custom_model = Net().to(device)
    custom_model.load_state_dict(torch.load('my_model.pth'))

    # Test the model
    custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        errors = []
        total = 0
        num_image = 0
        for item in test_loader:
            #print("item {}/n".format(item))
            images = item['image'].to(device)
            labels = torch.tensor(item['label']).to(device).float()
            outputs = custom_model(images)

            o = (int)(outputs.detach().to('cpu').numpy() * 255)
            l = (int)(labels.detach().to('cpu').numpy() * 255)
            print('output:{} labels:{}'.format(o, l))
            total += abs(o - l)
            errors.append(abs(o - l))

            a = images[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            #grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            grayA = a * 255
            ret,c = cv2.threshold(grayA,o,255,cv2.THRESH_BINARY_INV)
            cv2.imshow('a',a)
            cv2.imshow('c',c)
            cv2.waitKey(1)
            cv2.imwrite('results/threshold_regression_' + str(num_image) + '.jpg', c)
            num_image += 1

        print('total:{}'.format(total))
        print('mean:{}'.format(statistics.mean(errors)))
        print('stdev:{}'.format(statistics.stdev(errors)))

if __name__ == "__main__":
    main()

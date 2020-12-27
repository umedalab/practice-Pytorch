# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

import os
from PIL import Image
import cv2
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from skimage.metrics import structural_similarity 
#from skimage.measure import compare_ssim
from loader.CustomDataLoader import CustomImageLabelDataset


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	# link: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def preprocess_dataset(dataset, fname_out):
    ''' This function elaborates a dataset and create the input for the training/test step.
    	param[in] dataset Dataset to elaborate
    	param[in] fname_out File with the list of files to be used for training/test.
    '''
    
    # get a sample
    container_res = []
 
    for i in range(0, dataset.__len__()):
        item = dataset.__getitem__(i)
        print('image:{}'.format(item['image'].shape))
        print('label:{}'.format(item['label'].shape))
        a = item['image'].detach().to('cpu').numpy().transpose(1, 2, 0)
        b = item['label'].detach().to('cpu').numpy().transpose(1, 2, 0)
        # convert the images to grayscale
        grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) * 255
        grayB = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) * 255
        ret,bthr = cv2.threshold(grayB,10,255,cv2.THRESH_BINARY)


        best_score = 0
        best_i = 0
        for i in range(0,256):
            ret,athr = cv2.threshold(grayA,i,255,cv2.THRESH_BINARY_INV)
            (mssim, S) = structural_similarity(athr, bthr, full=True)
            #S = (S).astype("uint8")
            na_white_pix = np.sum(athr == 255)
            nb_white_pix = np.sum(bthr == 255)

            #print('athr:{}'.format(athr))
            #print('bthr:{}'.format(bthr))

            cv2.imshow('athr',athr)
            cv2.imshow('bthr',bthr)
            cv2.waitKey(1)
            # count the total number of white pixels
            #mssim = abs(na_white_pix - nb_white_pix)

            #print('score[{}]:{} -> {} {}'.format(i, mssim, na_white_pix,  nb_white_pix))

            if mssim > best_score:
                best_score = mssim
                best_i = i
                #cv2.imshow('athr',athr)
                #cv2.imshow('bthr',bthr)
                #cv2.imshow('c',c)
                #cv2.imshow('diff',diff)
                #cv2.waitKey(0)
        #loss = torch.mean((output - target)**2)
        #loss = torch.tensor(score, requires_grad=True)
        #loss = torch.tensor([[1.0 - score]], device='cuda:0', requires_grad=True)
    
        print('b:{} | {} {}'.format(item['name'], best_i, best_score))
        container_res.append([item['name'], best_i])

        ret,c = cv2.threshold(grayA,best_i,255,cv2.THRESH_BINARY_INV)
        #print('c:{}'.format(c))
        cv2.imshow('a',a)
        cv2.imshow('b',b)
        cv2.imshow('c',c)
        cv2.waitKey(1)

    ##text=List of strings to be written to file
    print('[i] write:{}'.format(fname_out))
    with open(fname_out,'w') as file:
        for item in container_res:
            print('item:{}'.format(item))
            file.write(item[0] + ',' + str(item[1]))
            file.write('\n')


def main():

    transforms_train = transforms.Compose([transforms.Resize((128, 128)),
                                           #transforms.RandomRotation(10.),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    train_data_set = CustomImageLabelDataset(data_set_path_color="/home/moro/workspace/work/Todai/Concrete/DeepSegmentor/datasets/DeepCrack/train_img", data_set_path_label="/home/moro/workspace/work/Todai/Concrete/DeepSegmentor/datasets/DeepCrack/train_lab", transforms=transforms_train)
    test_data_set = CustomImageLabelDataset(data_set_path_color="/home/moro/workspace/work/Todai/Concrete/DeepSegmentor/datasets/DeepCrack/test_img", data_set_path_label="/home/moro/workspace/work/Todai/Concrete/DeepSegmentor/datasets/DeepCrack/test_lab", transforms=transforms_test)

    # process the datasets
    preprocess_dataset(train_data_set, 'train_csvfile.csv')
    preprocess_dataset(test_data_set, 'test_csvfile.csv')


if __name__ == "__main__":
    main()

# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

import os
from PIL import Image
import cv2
import csv
import numpy as np
import random

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class SelfSupervisedLabelDataset(Dataset):
    """ Custom data loader
        The class gets: 
          images from a folder (converted to grayscale)
          expected threshold value for binarization
    """
    def read_data_set(self):

        all_img_files = []
        all_lab_files = []

        img_dir_color = self.data_set_path_color

        img_files_color = os.walk(img_dir_color).__next__()[2]

        print('img_files_color:{}'.format(img_files_color))

        for img_file in img_files_color:
            img_file = os.path.join(img_dir_color, img_file)
            img = Image.open(img_file)
            if img is not None:
                all_img_files.append(img_file)

        self.dict_1=dict() 
        if self.data_set_label != None: 
            with open(self.data_set_label, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
            for name,score in data: 
                self.dict_1.setdefault(name, []).append(float(score) / 255.0) 
        print('dict_1:{}'.format(self.dict_1))

        print('len(all_img_files), len(class_names):{} {}'.format(len(all_img_files), len(all_lab_files)))
        return all_img_files, all_lab_files, len(all_img_files), len(all_lab_files)

    def __init__(self, data_set_path_color, data_set_label, do_training, transforms=None):
        self.do_training = do_training
        self.data_set_path_color = data_set_path_color
        self.data_set_label = data_set_label
        self.image_files_path, self.labels_files_path, self.length, self.length_lab = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        #print'getitem:{}'.format(self.image_files_path[index]))
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")
        image = np.asarray(image)
        #print('image shape:{}'.format(image.shape))

        width = 100
        height = 100
        x = random.randint(width / 2, image.shape[0] - width)
        y = random.randint(height / 2, image.shape[1] - height)
        image = image[x - int(width / 2):x+int(width/2), y - int(height / 2):y + int(height/2), :]
        image = Image.fromarray(image)
        #label = Image.open(self.labels_files_path[index])
        #label = label.convert("RGB")
        if self.do_training is True:
            label = 1#self.dict_1[self.image_files_path[index]]
        else:
            label = 0

        if self.transforms is not None:
            image = self.transforms(image)
            #label = self.transforms(label)
        #print('image shape:{}'.format(image.shape))
        #print('x y:{} {}'.format(x, y))


        return {'image': image, 
            'position': [float(x) / image.shape[1], float(y) / image.shape[2]], 'label': label}

    def __len__(self):
        return self.length


import sys
import os

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from networks.modelobject import Net
from loader.CustomObjectDataLoaderFile import CustomObjectDataLoader
from loader.ObjectFeaturesFile import ObjectFeatures
from loader.ObjectLabelsFile import ObjectLabels
from loader.DataloaderFolder import DataloaderFolder

# Log
from utils import mytensorboard, myutils
from utils.utilsModel import UtilsModel

def train_val_dataset(dataset, val_split=0.25):
    lengths = [int(len(dataset)*(1.0 - val_split)), int(len(dataset)*val_split)]
    dataset_train, dataset_validation = torch.utils.data.random_split(dataset, lengths)
    return dataset_train, dataset_validation

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

    # Hyper parameters
    hyper_param_epoch = 200
    hyper_param_batch_train = 20
    hyper_param_learning_rate = 0.00001

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

    mypath = 'D:/workspace/programs/MyPractice/practice-Pytorch/src/variablelength/rnn/SequenceRecognitionPytorch/datatraining/object/001'
    res = tdata.get_pair(mypath)
    print('res:{}'.format(res))
    # data from video customer
    # read the label file
    for r in res:
        print('r:{}'.format(r))
        labels = ObjectLabels.read_label(r[1])
        data_loader.read_data_pose_from_measurementpy_id(r[0], labels, filters, objectlabel2id)
        print('len:{}'.format(data_loader.__len__()))

    # Get a training and validation set from the original dataset (need enough data)
    #train, validation = train_val_dataset(data_loader)
    # Data Loaders
    train_loader = DataLoader(data_loader, batch_size=hyper_param_batch_train, shuffle=True)
    validation_loader = DataLoader(data_loader, batch_size=hyper_param_batch_train, shuffle=True)

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    # Model
    #custom_model = Net(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    custom_model = Net().to(device)
    print(custom_model)
    
    #custom_model.load_state_dict(torch.load('data/model_actions.pth')) 
    #custom_model.eval()
    #summary(custom_model, (1, 120))
    # applying logging only in the main process
    # ### OUR CODE ###
    if False and myutils.is_main_process():

        # get the input size
        N, C = UtilsModel.network_inputsize(custom_model)
        print('N={} C={}'.format(N, C))

        dummy_input = torch.rand(1, 1, 256, 256, requires_grad=True).to(device)
        with torch.onnx.select_model_mode_for_export(custom_model, False):
            mytensorboard.logger.add_model(custom_model, dummy_input)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    #criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    optimizer = torch.optim.RMSprop(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.SGD(custom_model.parameters(), lr=hyper_param_learning_rate)
    #optimizer = torch.optim.ASGD(custom_model.parameters(), lr=hyper_param_learning_rate)

    iterations = 0
    dev_every = 100

    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print('input:{} target:{}'.format(input, target))
    output = loss(input, target)
    print('output:{}'.format(output))
    output.backward()
    
    #exit(0)

    # Train (no validation test)
    for e in range(hyper_param_epoch):
        print('epoch:{}'.format(e))
        for i_batch, item in enumerate(train_loader):
        
            #print('i_batch[{}]:{}'.format(i_batch, item))

            # switch model to training mode, clear gradient accumulators
            custom_model.train(); optimizer.zero_grad()
            
            iterations += 1
        
            features = item['feature'].to(device, dtype=torch.float)
            if e == -1:
                print('size features:{}'.format(features.shape))
            labels = item['label'].to(device, dtype=torch.long)
            #labels = torch.stack(labels).to(device)
            # Forward pass
            outputs = custom_model(features)
            #print('outputsS:{} labelsS:{}'.format(outputs.shape, labels.shape))
            #print('outputs:{} labels:{}'.format(outputs, labels))
            loss = criterion(outputs.squeeze(1), labels.squeeze(1)) # classification
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
            # evaluate performance on validation set periodically
            if iterations % dev_every == 0:
            #    print('labels: {} | outputs:{} | loss:{}'.format(labels, outputs.shape, loss))
                print('loss:{}'.format(loss))

            #print('outputs:{} | labels:{}'.format(outputs, labels))

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
                    total_samples = 0
                    total = 0
                    for item in validation_loader:

                        features = item['feature'].to(device, dtype=torch.float)
                        labels = item['label'].to(device)
                        # Forward pass
                        outputs = custom_model(features)

                        #o = (int)(outputs.detach().to('cpu').numpy())
                        o = []
                        for v in outputs:
                            #print('am:{} {}'.format(torch.argmax(v).detach().to('cpu'), type(torch.argmax(v).detach().to('cpu').item())))
                            o.append(torch.argmax(v).detach().to('cpu').item())
                        o = np.array(o).reshape(-1, 1)
                        #l = (int)(labels.detach().to('cpu').numpy())
                        l = labels.detach().to('cpu').numpy()
                        #print('output:{} labels:{}'.format(o, l))
                        for i in range(0, len(o)):
                            if o[i]== l[i]:
                                correct += 1
                            total += abs(o[i] - l[i])
                            total_samples += 1
                    print('total:{} correct:{}/{}'.format(total, correct, total_samples))
                    # Save
                    torch.save(custom_model.state_dict(), 'data/model_actions.pth')

            #else:
            #    if (i_batch + 1) % hyper_param_batch_train == 0:
            #        print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, hyper_param_epoch, loss.item()))

    # Save
    torch.save(custom_model.state_dict(), 'data/model_scenes.pth')


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

import numpy as np

import copy
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


import networks.modelanomaly

from loader.CustomPoseDataLoaderFile import CustomPoseDataLoader
from loader.SkeletonFeaturesFile import SkeletonFeatures
# Log
from utils import mytensorboard, myutils
from utils.utilsModel import UtilsModel

observed_length_expected = 2
data_loader = CustomPoseDataLoader(observed_length=observed_length_expected, data_set_path='data', fname_bodydescriptor='connectivityCOCO18.txt')

data_loader.read_data_pose('datatraining/DemoHouse20200428/observer.txt')
#for i in range(0, data_loader.__len__()):
#    var = data_loader.__getitem__(i)
#    print('var:{}'.format(var))

# Hyper parameters
hyper_param_epoch = 30
hyper_param_batch_train = 1
hyper_param_batch_test = 1
hyper_param_learning_rate = 0.00001

# Data Loaders
train_loader = DataLoader(data_loader, batch_size=hyper_param_batch_train, shuffle=True)
# https://github.com/pytorch/pytorch/issues/1917
var = next(iter(train_loader))
#print('var DataLoader:{}'.format(var))




RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print('device:{}'.format(device))
# Our Autoencoder passes the input through the Encoder and Decoder. Let's create an instance of it:

# In[27]:

seq_len = observed_length_expected * data_loader.len_connectivity()#340
n_features = 1

model = networks.modelanomaly.RecurrentAutoencoder(device, seq_len, n_features, 128)
model = model.to(device)

print(model)

MODEL_PATH = 'data/model_anomaly.pth'
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true['feature'].to(device, dtype=torch.float)
      #reshape
      seq_true = torch.reshape(seq_true, (seq_len, 1))
      seq_pred = model(seq_true)

      #print('seq_pred:{} seq_pred:{}'.format(seq_pred.shape, seq_pred.shape))
      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses


_, losses = predict(model, train_loader)
THRESHOLD = 26


# ## Evaluation
# 
# Using the threshold, we can turn the problem into a simple binary classification task:
# 
# - If the reconstruction loss for an example is below the threshold, we'll classify it as a *normal* heartbeat
# - Alternatively, if the loss is higher than the threshold, we'll classify it as an anomaly

# ### Normal hearbeats
# 
# Let's check how well our model does on normal heartbeats. We'll use the normal heartbeats from the test set (our model haven't seen those):

# In[36]:


predictions, pred_losses = predict(model, train_loader)
print('predictions:{}'.format(predictions))
print('pred_losses:{}'.format(pred_losses))

# We'll count the correct predictions:

# In[37]:


correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct[{len(train_loader)}] normal predictions: {correct}/{len(train_loader)}')

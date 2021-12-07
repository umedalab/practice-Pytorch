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


from loader.CustomPoseDataLoaderFile import CustomPoseDataLoader
from loader.SkeletonFeaturesFile import SkeletonFeatures
from loader.DataloaderFolder import DataloaderFolder
# Log
from utils import mytensorboard, myutils
from utils.utilsModel import UtilsModel

import networks.modelanomaly

# """ The function read the configuration file and pose information
# """
# num_line_tmp, poses2d_tmp, poses3d_tmp, labels_tmp = SkeletonFeatures.load_keypoints_labels(
#   'datatraining/20200427_155335/observer.txt')
# print('num_line_tmp:{}'.format(num_line_tmp.shape))
# print('poses2d_tmp:{}'.format(poses2d_tmp.shape))
# print('labels_tmp:{}'.format(labels_tmp.shape))
# num_line_tmp, poses2d_tmp, poses3d_tmp, labels_tmp = SkeletonFeatures.load_keypoints_labels_from_measurementpy(
#   'observation_MeasurementPy.txt')
# print('num_line_tmp:{}'.format(num_line_tmp.shape))
# print('poses2d_tmp:{}'.format(poses2d_tmp.shape))
# print('labels_tmp:{}'.format(labels_tmp.shape))
# exit(0)

observed_length_expected = 2
data_loader = CustomPoseDataLoader(observed_length=observed_length_expected,
                                   data_set_path='data', fname_bodydescriptor='connectivityCOCO14.txt',
                                   out_item_mode=0) #connectivityCOCO18

#data_loader.read_data_pose_from_measurementpy('datatraining/observation/observation_MeasurementPy_datastanding.txt', label_id=0)
#data_loader.read_data_pose_from_measurementpy('datatraining/observation/observation_MeasurementPy_datasitting.txt', label_id=2)
#data_loader.read_data_pose_from_measurementpy('../../../data/datasets/actions/sitting/scene001.txt', label_id=2)
#data_loader.read_data_pose('datatraining/20200427_155335/observer.txt')
#data_loader.read_data_pose('datatraining/20200427_155442/observer.txt')
#data_loader.read_data_pose('datatraining/20200428_102645/observer.txt')
#data_loader.read_data_pose('datatraining/DemoHouse20200428/observer.txt')

filters = [7] # it filters the falling action (?)
tdata = DataloaderFolder()

mypath = '/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/training/scene006'
res = tdata.get_pair(mypath)
print('res:{}'.format(res))
# data from video customer
# read the label file
for r in res:
    labels = CustomPoseDataLoader.read_label(r[1])
    data_loader.read_data_pose_from_measurementpy_id(r[0], 0, labels, filters, -1, -1)
    print('len:{}'.format(data_loader.__len__()))

mypath = '/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/training/scene007'
res = tdata.get_pair(mypath)
print('res:{}'.format(res))
# data from video customer
# read the label file
for r in res:
    labels = CustomPoseDataLoader.read_label(r[1])
    data_loader.read_data_pose_from_measurementpy_id(r[0], 16, labels, filters, -1, -1)
    print('len:{}'.format(data_loader.__len__()))

mypath = '/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/training/oita20210820video1'
res = tdata.get_pair(mypath)
print('res:{}'.format(res))
# data from video customer
# read the label file
for r in res:
    labels = CustomPoseDataLoader.read_label(r[1])
    data_loader.read_data_pose_from_measurementpy_id(r[0], 0, labels, filters, -1, -1)
    print('len:{}'.format(data_loader.__len__()))
    
mypath = '/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/training/oita20210820video2MAH'
res = tdata.get_pair(mypath)
print('res:{}'.format(res))
# data from video customer
# read the label file
for r in res:
    print('r:{}'.format(r))
    labels = CustomPoseDataLoader.read_label(r[1])
    data_loader.read_data_pose_from_measurementpy_id(r[0], 0, labels, filters, -1, -1)
    print('len:{}'.format(data_loader.__len__()))    
    
#for i in range(0, data_loader.__len__()):
#    var = data_loader.__getitem__(i)
#    print('var:{}'.format(var))

# Hyper parameters
hyper_param_epoch = 100
hyper_param_batch_train = 1
hyper_param_learning_rate = 0.00001

# Data Loaders
train_loader = DataLoader(data_loader, batch_size=hyper_param_batch_train, shuffle=True)
# https://github.com/pytorch/pytorch/issues/1917
#var = next(iter(train_loader))
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


# ## Training
# 
# Let's write a helper function for our training process:

# In[28]:


def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1000.0
  
  print('train_model')
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true['feature'].to(device, dtype=torch.float)
      #reshape
      seq_true = torch.reshape(seq_true, (seq_len, 1))
      seq_pred = model(seq_true)
      #print('seq_true:{} seq_pred:{}'.format(seq_true.shape, seq_pred.shape))

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true['feature'].to(device, dtype=torch.float)
        #reshape
        seq_true = torch.reshape(seq_true, (seq_len, 1))
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history


# At each epoch, the training process feeds our model with all training examples and evaluates the performance on the validation set. Note that we're using a batch size of 1 (our model sees only 1 sequence at a time). We also record the training and validation set losses during the process.
# 
# Note that we're minimizing the [L1Loss](https://pytorch.org/docs/stable/nn.html#l1loss), which measures the MAE (mean absolute error). Why? The reconstructions seem to be better than with MSE (mean squared error).
# 
# We'll get the version of the model with the smallest validation error. Let's do some training:

# In[29]:


model, history = train_model(
  model, 
  train_loader, 
  train_loader, 
  n_epochs=hyper_param_epoch
)

MODEL_PATH = 'data/model_anomaly.pth'
torch.save(model.state_dict(), MODEL_PATH)


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
#print('predictions:{}'.format(predictions))
print('pred_losses:{}'.format(pred_losses))

# We'll count the correct predictions:

# In[37]:


correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct[{len(train_loader)}] normal predictions: {correct}/{len(train_loader)}')

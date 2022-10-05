# check https://github.com/Lightning-AI/lightning/issues/7447#issuecomment-835695726

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class MyModelA(nn.Module):
    def __init__(self, hidden_dim = 10):
        super(MyModelA, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, 2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
        
    def forward(self, x):
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        return F.mse_loss(self.forward(x), y)
    
class MyModelB(nn.Module):
    def __init__(self, hidden_dim = 10):
        super(MyModelB, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, 2)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
        
    def forward(self, x):
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        return F.mse_loss(self.forward(x), y)

class MyEnsemble(nn.Module):
    def __init__(self, 
                 modelA_hparams, modelB_hparams,
                 modelA_params = None, modelB_params = None):
        super(MyEnsemble, self).__init__()
        self.modelA = MyModelA(**modelA_hparams)
        self.modelB = MyModelB(**modelB_hparams)

        if modelA_params:
            self.modelA.load_state_dict(modelA_params)
        if modelB_params:
            self.modelB.load_state_dict(modelB_params)

        self.modelA.freeze()
        self.modelB.freeze()
        self.classifier = torch.nn.Linear(4, 2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self.forward(x), y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self.forward(x), y)


# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dl = DataLoader(TensorDataset(torch.randn(1000, 10).to(device), 
                              torch.randn(1000, 2).to(device)), 
                batch_size = 10)

modelA = MyModelA()
modelB = MyModelB()

modelA.to(device)
modelB.to(device)

hyper_param_epoch = 100
hyper_param_learning_rate = 1e-3

# Loss and optimizer
#criterion = nn.L1Loss()
criterion = nn.MSELoss()
#criterion = nn.BCEWithLogitsLoss()
optimizerA = torch.optim.Adam(modelA.parameters(), lr=hyper_param_learning_rate)
optimizerB = torch.optim.Adam(modelB.parameters(), lr=hyper_param_learning_rate)

#optimizerA = torch.optim.RMSprop(modelA.parameters(), lr=hyper_param_learning_rate)
#optimizerB = torch.optim.RMSprop(modelB.parameters(), lr=hyper_param_learning_rate)
#optimizer = torch.optim.SGD(custom_model.parameters(), lr=hyper_param_learning_rate)
#optimizer = torch.optim.ASGD(custom_model.parameters(), lr=hyper_param_learning_rate)

iterations = 0
dev_every = 1000

# switch model to training mode, clear gradient accumulators
#custom_model.train(); optimizer.zero_grad()
modelA.train(); optimizerA.zero_grad()
modelB.train(); optimizerB.zero_grad()
# Train (no validation test)
for e in range(hyper_param_epoch):
    for i_batch, item in enumerate(dl):
        
        iterations += 1

        #print('i_batch:{} item:{}'.format(i_batch, item))

        # Forward pass
        outputsA = modelA(item[0])
        outputsB = modelB(item[0])
        #print('outputs:{} labels:{}'.format(outputs, labels))
        lossA = criterion(outputsA, item[1]) # classification
        lossB = criterion(outputsB, item[1]) # classification

        # only if process images
        print('labels: {} | outputs:{} | loss:{}'.format(item[1], outputsA, loss))

        # Backward and optimize
        optimizerA.zero_grad()
        optimizerB.zero_grad()
        lossA.backward()
        lossB.backward()
        optimizerA.step()
        optimizerB.step()

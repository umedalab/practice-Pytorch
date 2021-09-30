# https://gist.github.com/davidnvq/594c539b76fc52bef49ec2332e6bcd15
# https://github.com/songyouwei/ABSA-PyTorch/blob/master/layers/dynamic_rnn.py

import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DynamicRNN(nn.Module):
	"""
	The wrapper version of recurrent modules including RNN, LSTM
	that support packed sequence batch.
	"""

	def __init__(self, rnn_module):
		super().__init__()
		self.rnn_module = rnn_module

	def forward(self, x, len_x, initial_state=None):
		"""
		Arguments
		---------
		x : torch.FloatTensor
			padded input sequence tensor for RNN model
			Shape [batch_size, max_seq_len, embed_size]

		len_x : torch.LongTensor
			Length of sequences (b, )

		initial_state : torch.FloatTensor
			Initial (hidden, cell) states of RNN model.

		Returns
		-------
		A tuple of (padded_output, h_n) or (padded_output, (h_n, c_n))
			padded_output: torch.FloatTensor
				The output of all hidden for each elements. The hidden of padding elements will be assigned to
				a zero vector.
				Shape [batch_size, max_seq_len, hidden_size]

			h_n: torch.FloatTensor
				The hidden state of the last step for each packed sequence (not including padding elements)
				Shape [batch_size, hidden_size]
			c_n: torch.FloatTensor
				If rnn_model is RNN, c_n = None
				The cell state of the last step for each packed sequence (not including padding elements)
				Shape [batch_size, hidden_size]

		Example
		-------
		"""
		# First sort the sequences in batch in the descending order of length
		sorted_len, idx = len_x.sort(dim=0, descending=True)
		sorted_x = x[idx]

		# Convert to packed sequence batch
		packed_x = pack_padded_sequence(sorted_x, lengths=sorted_len, batch_first=True)

		# Check init_state
		if initial_state is not None:
			if isinstance(initial_state, tuple):  # (h_0, c_0) in LSTM
				hx = [state[:, idx] for state in initial_state]
			else:
				hx = initial_state[:, idx]  # h_0 in RNN
		else:
			hx = None

		# Do forward pass
		self.rnn_module.flatten_parameters()
		packed_output, last_s = self.rnn_module(packed_x, hx)

		# pad the packed_output
		max_seq_len = x.size(1)
		padded_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_len)

		# Reverse to the original order
		_, reverse_idx = idx.sort(dim=0, descending=False)

		padded_output = padded_output[reverse_idx]

		if isinstance(self.rnn_module, nn.RNN):
			h_n, c_n = last_s[:, reverse_idx], None
		else:
			h_n, c_n = [s[:, reverse_idx] for s in last_s]

		return padded_output, (h_n, c_n)



class Net(nn.Module):
    """ The model expects a source of 256x256 pixels
    """
    def __init__(self, drnn, in_size, out_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(60, out_size)
        self.drnn = drnn

        # 1D CovNet for learning the Spectral features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(56, 2)
        self.lsoftmax = nn.Softmax(dim=1)

    def forward(self, x, len_x):
        #print('x>>:{}'.format(x))
        
        d_out, (dh_n, dc_n) = self.drnn(x)
        #print('d_out>>:{}'.format(d_out.shape))
        #print('d_out>>:{}'.format(d_out))
        #exit(0)
        e = d_out[:,-1,:].reshape(x.shape[0], -1)
        f = d_out[:,-1,:].reshape(x.shape[0], 1, -1)
        #print('e:{}'.format(e.shape))
        #print('e:{}'.format(e))
        #print('f:{}'.format(f.shape))
        f = F.relu(self.conv1(f))
        #print('f:{}'.format(f.shape))
        f = F.relu(self.conv2(f))
        #print('f:{}'.format(f.shape))
        f = F.relu(self.fc1(f))
        #print('f:{}'.format(f.shape))
        #p = F.logsoftmax(f, dim=1)
        p = self.lsoftmax(f.view(x.shape[0], -1))
        #print('p:{}'.format(p.shape))
        return p
        prediction = F.softmax(self.fc(e), dim=1)
        #print('prediction:{}'.format(prediction.shape))
        return prediction



"A simple example to test"

# prepare examples
x = [torch.tensor([[1.0, 1.0],
                   [2.0, 2.0],
                   [3.0, 3.0],
                   [4.0, 4.0],
                   [5.0, 5.0]]),

     torch.tensor([[2.5, 2.5]]),

     torch.tensor([[2.2, 2.2],
                   [3.5, 3.5]])]
len_x = torch.tensor([5, 1, 2])

# pad the seq_batch
padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.01)



# prepare examples
y = [torch.tensor([[1.0, 1.0],
                   [2.0, 2.0],
                   [3.0, 3.0],
                   [4.0, 4.0],
                   [4.0, 4.0],
                   [4.0, 4.0],
                   [4.0, 4.0],
                   [4.0, 4.0],
                   [4.0, 4.0],
                   [4.0, 4.0],
                   [5.0, 5.0],
                   [6.0, 6.0]]),

     torch.tensor([[2.5, 2.5]]),
     torch.tensor([[2.5, 2.5]]),
     torch.tensor([[2.5, 2.5]]),
     torch.tensor([[2.5, 2.5]]),
     torch.tensor([[2.5, 2.5]]),

     torch.tensor([[1.0, 1.0],
                   [2.0, 2.0],
                   [4.0, 4.0],
                   [5.0, 5.0]]),

     torch.tensor([[2.2, 2.2],
                   [3.5, 3.5]])]
len_y = torch.tensor([12, 1, 1, 1, 1, 1, 4, 2])

# pad the seq_batch
padded_y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0.01)


"""
>>> padded_x
tensor([[[1.0000, 1.0000],
         [2.0000, 2.0000],
         [3.0000, 3.0000],
         [4.0000, 4.0000],
         [5.0000, 5.0000]],

        [[2.5000, 2.5000],
         [0.0100, 0.0100],
         [0.0100, 0.0100],
         [0.0100, 0.0100],
         [0.0100, 0.0100]],

        [[2.2000, 2.2000],
         [3.5000, 3.5000],
         [0.0100, 0.0100],
         [0.0100, 0.0100],
         [0.0100, 0.0100]]])
"""

# init 2 recurrent module: lstm, drnn
rnn = nn.LSTM(input_size=2, hidden_size=3, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
drnn = DynamicRNN(rnn)

# get the outputs
d_out, (dh_n, dc_n) = drnn(padded_x, len_x)
out, (h_n, c_n) = rnn(padded_x)
print(d_out.shape)
print(out.shape)
print(out[:,-1,:].shape)


d_out, (dh_n, dc_n) = drnn(padded_y, len_y)
out, (h_n, c_n) = rnn(padded_y)
print(d_out.shape)
print(out.shape)
print(out[:,-1,:].shape)
#net = Net(drnn, out[:,-1,:].view(-1).shape[0], 2)
#output = net(padded_y, len_y)
#print('output:{}'.format(output))



# compare two outputs
print(d_out == out)
"""
tensor([[[1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 0, 0, 0], # only the forward direction is the same not the backward direction 
         [0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[1, 1, 1, 0, 0, 0], # same as above
         [1, 1, 1, 0, 0, 0], # same as above
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
"""

print(dh_n == h_n)
"""
tensor([[[1, 1, 1], # since no padding in the first seq
         [0, 0, 0],
         [0, 0, 0]],
         
        [[1, 1, 1],
         [0, 0, 0],
         [0, 0, 0]]], dtype=torch.uint8)
"""

print(dc_n == c_n)
"""
tensor([[[1, 1, 1], # since no padding in the first seq
         [0, 0, 0],
         [0, 0, 0]],
         
        [[1, 1, 1],
         [0, 0, 0],
         [0, 0, 0]]], dtype=torch.uint8)
"""




n_epochs = 10000
batch_size = 2
learning_rate = 0.01


# init 2 recurrent module: lstm, drnn
rnn = nn.LSTM(input_size=5, hidden_size=30, bidirectional=True, batch_first=True)
drnn = DynamicRNN(rnn)
net = Net(rnn, 384, 2)

#loss_fn = torch.nn.MSELoss(size_average=False)
loss_fn = torch.nn.CrossEntropyLoss()#NLLLoss()
optimiser = torch.optim.RMSprop(net.parameters(), lr=learning_rate)#torch.optim.Adam(net.parameters(), lr=learning_rate)
#####################
# Train model
#####################
hist = np.zeros(n_epochs)


# create random data
x = []
len_x = []
label_y = []
for j in range(0, 1000):
    num_elems = np.random.choice(5, 1, replace=False)
    #print('num_elems:{}'.format(num_elems))
    if num_elems[0] == 0:
        num_elems[0] = 1
    x.append(torch.rand(num_elems[0], 5))
    len_x.append(num_elems[0])

    v = random.uniform(0, 1)
    if v <= 0.5:
        #label_y.append([1,0])
        label_y.append(0.)
    else:
        #label_y.append([0,1])
        label_y.append(1.)
len_x = torch.tensor(len_x, dtype=torch.float32)
#label_y = torch.tensor(label_y, dtype=torch.float32)
label_y = torch.tensor(label_y, dtype=torch.long)
#for i in range(0, len(len_x)):
#    print(x[i], ' ', len_x[i], ' ', label_y[i])

for epoch in range(n_epochs):
    net.zero_grad()

    for j in range(0, int(len(len_x) / batch_size) - batch_size):

        xd = x[j * batch_size:j * batch_size + batch_size]
        len_xd = len_x[j * batch_size:j * batch_size + batch_size]
        label_yd = label_y[j * batch_size:j * batch_size + batch_size]

        optimiser.zero_grad()
        # pad the seq_batch
        padded_x = torch.nn.utils.rnn.pad_sequence(xd, batch_first=True, padding_value=0.01)

        #print('padded_x:{}'.format(padded_x.shape))
        #print('len_x:{}'.format(len_x.shape))

        # in case you wanted a semi-full example
        y_pred = net.forward(padded_x, len_xd)
        #print('y_pred:{} label_y:{}'.format(y_pred, label_yd))
        loss = loss_fn(y_pred, label_yd)
        #print('loss:{}'.format(loss))

        if j % 10 == 0:
            print("Epoch ", epoch, "MSE: ", loss.item())
            print('y_pred:{} label_y:{}'.format(y_pred, label_yd))
        hist[epoch] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        loss.backward(retain_graph=True)# Backward pass
        optimiser.step()# Update parameters

    print("Epoch ", epoch, "MSE: ", loss.item())


# test

for epoch in range(n_epochs):
    net.eval()
    net.zero_grad()

    # create random data
    x = []
    len_x = []
    label_y = []
    for j in range(0, batch_size):
        num_elems = np.random.choice(5, 1, replace=False)
        #print('num_elems:{}'.format(num_elems))
        if num_elems[0] == 0:
            num_elems[0] = 1
        x.append(torch.rand(num_elems[0], 5))
        len_x.append(num_elems[0])

        v = np.random.choice(1, 1, replace=False)
        if v <= 0.5:
            label_y.append([1,0])
        else:
            label_y.append([0,1])
    len_x = torch.tensor(len_x, dtype=torch.float32)
    label_y = torch.tensor(label_y, dtype=torch.float32)

    if True:
        optimiser.zero_grad()
        # pad the seq_batch
        padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.01)

        #print('padded_x:{}'.format(padded_x))
        #print('len_x:{}'.format(len_x))

        # in case you wanted a semi-full example
        y_pred = net.forward(padded_x, len_x)
        print('y_pred:{}'.format(y_pred))

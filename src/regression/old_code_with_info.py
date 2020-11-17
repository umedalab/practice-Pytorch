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

def TransposeModel(Weight_Path, Model, height=448, width=448, ch=3):
    device = torch.device("cuda")
    Model.to(device)
    Model.load_state_dict(torch.load(Weight_Path))
    # jit へ変換
    traced_net = torch.jit.trace(Model, torch.rand(1, ch, height, width).to(device))
    # 後の保存(Save the transposed Model)
    traced_net.save('BSCNN_h{}_w{}_mode{}_cuda.pt'.format(height, width, ch))
    print('BSCNN_h{}_w{}_mode{}_cuda.pt is exported.'.format(height, width, ch))


class CustomImageLabelDataset(Dataset):
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

        with open(self.data_set_label, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        self.dict_1=dict() 
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
        #print('getitem:{}'.format(self.image_files_path[index]))
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")
        #label = Image.open(self.labels_files_path[index])
        #label = label.convert("RGB")
        if self.do_training is True:
            label = self.dict_1[self.image_files_path[index]]
        else:
            label = 0

        if self.transforms is not None:
            image = self.transforms(image)
            #label = self.transforms(label)

        return {'image': image, 'label': label}

    def __len__(self):
        return self.length


class CustomConvNet(nn.Module):

    num_classes_ = 0

    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        self.num_classes_ = num_classes
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.layer6 = self.conv_module(256, 128)
        self.gap = self.global_avg_pool(128, self.num_classes_)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes_)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

# https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/utkuozbulak/pytorch-cnn-visualizations
# https://github.com/tomjerrygithub/Pytorch_VehicleColorRecognition/blob/master/model.py
# https://www.guru99.com/pytorch-tutorial.html
# https://datascience.stackexchange.com/questions/67549/validation-loss-is-not-decreasing-regression-model
class Net0(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        #print('x:{}'.format(x.shape))
        x = self.pool(F.relu(self.conv1(x)))
        #print('x:{}'.format(x.shape))
        x = self.pool(F.relu(self.conv2(x)))
        #print('x:{}'.format(x.shape))
        x = x.view(-1, 32 * 29 * 29)
        #print('x:{}'.format(x.shape))
        x = F.relu(self.fc1(x))
        #print('x:{}'.format(x.shape))
        x = F.relu(self.fc2(x))
        #print('x:{}'.format(x.shape))
        x = self.fc3(x)
        #print('x:{}'.format(x.shape))
        return x

class Net1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 5)
        self.fc1 = nn.Linear(32, 16)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        print('x0:{}'.format(x.shape))
        x = self.pool(F.relu(self.conv1(x)))
        print('x1:{}'.format(x.shape))
        x = self.pool(F.sigmoid(self.conv2(x)))
        print('x2:{}'.format(x.shape))
        x = F.sigmoid(self.fc1(x))
        print('x3:{}'.format(x.shape))
        x = self.drop1(x)
        print('x4:{}'.format(x.shape))
        x = self.fc2(x)
        print('x5:{}'.format(x.shape))

        return x


# https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/5
# https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/6
class NetColor(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)  # notice the padding
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1) # again...
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1) # again...
        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) # again...
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc1 = nn.Linear(8192 * 4, 512) # it is 64....
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        #print('x0:{}'.format(x.shape))
        x = self.conv1(x)
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x)
        #print('x0:{}'.format(x.shape))
        x = self.conv2(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x) 
        #print('x0:{}'.format(x.shape))

        x = self.conv3(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x) 
        #print('x0:{}'.format(x.shape))


        #x = x.view(-1, 1024)#16 * 64*64) 
        #x = self.bn1(x)
        x = x.view(-1, 8192 * 4)#16 * 64*64) 
        #print('x0:{}'.format(x.shape))
        #x = self.fc1(x) 
        x = F.relu(self.fc1(x))
        #print('x0:{}'.format(x.shape))
        x = self.dropout(x)
        #print('x0:{}'.format(x.shape))
        x= F.relu(self.fc2(x))
        #print('x0:{}'.format(x.shape))
        prediction = self.fc3(x) 
        #print('x0:{}'.format(prediction.shape))
        return prediction


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)  # notice the padding
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1) # again...
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1) # again...
        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) # again...
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm1d(num_features=8192)
        self.fc1 = nn.Linear(8192 * 8, 1024) # it is 64....
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        #print('x0:{}'.format(x.shape))
        x = self.conv1(x)
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x)
        #print('x0:{}'.format(x.shape))
        x = self.conv2(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x) 
        #print('x0:{}'.format(x.shape))

        x = self.conv3(x) 
        #print('x0:{}'.format(x.shape))
        x = self.relu(x)
        #print('x0:{}'.format(x.shape))
        x = self.pool(x) 
        #print('x0:{}'.format(x.shape))


        #x = x.view(-1, 8192)#16 * 64*64) 
        #x = self.bn1(x)
        x = x.view(-1, 8192 * 8)#16 * 64*64) 
        #print('x0:{}'.format(x.shape))
        #x = self.fc1(x) 
        x = F.sigmoid(self.fc1(x))
        #print('x0:{}'.format(x.shape))
        x = F.dropout(x)
        #print('x0:{}'.format(x.shape))
        x= F.sigmoid(self.fc2(x))
        #x = F.dropout(x)
        x= F.sigmoid(self.fc3(x))
        #x = F.dropout(x)
        #print('x0:{}'.format(x.shape))
        prediction = self.fc4(x)
        #print('x0:{}'.format(prediction.shape))
        return prediction


# https://stackoverflow.com/questions/57802632/loss-is-not-converging-in-pytorch-but-does-in-tensorflow
# https://discuss.pytorch.org/t/sloved-why-my-loss-not-decreasing/15924/6
# https://discuss.pytorch.org/t/cnn-does-not-predict-properly-does-not-converge-as-expected/43567
class NetB(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 298x298x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3)
        # convolutional layer (sees 147x147x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3)
        # convolutional layer (sees 71x71x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3)
        # convolutional layer (sees 33x33x64 tensor)
        self.conv4 = nn.Conv2d(64, 64, 3)
        # convolutional layer (sees 14x14x64 tensor)
        self.conv5 = nn.Conv2d(64, 64, 3)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 7 * 7 -> 500)
        self.fc1 = nn.Linear(256, 256)
        # linear layer (512 -> 1)
        self.fc2 = nn.Linear(256, 1)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        #print('x0:{}'.format(x.shape))

        # flatten image input
        x = x.view(-1, 256)
        #print('x0:{}'.format(x.shape))
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        #print('x0:{}'.format(x.shape))
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer
        x = self.fc2(x)
        return x

def my_loss(output, image, target):
    """ Custom loss function
    """
    a = image[0].detach().to('cpu').numpy().transpose(1, 2, 0)
    b = target[0].detach().to('cpu').numpy().transpose(1, 2, 0)
    # convert the images to grayscale
    grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) * 255
    grayB = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) * 255
    ret,bthr = cv2.threshold(grayB,10,255,cv2.THRESH_BINARY)

    best_score = 0
    best_i = 0
    for i in range(0,256):
        ret,athr = cv2.threshold(grayA,i,255,cv2.THRESH_BINARY_INV)
        (score, diff) = compare_ssim(athr, bthr, full=True)
        diff = (diff).astype("uint8")
        if score > best_score:
            best_score = score
            best_i = i

    t = torch.tensor([[float(best_i) / 255.0]], device='cuda:0')
    print('b:{} {} | {} {}'.format(best_i, best_score, output, t))
    loss = torch.mean((output - t)**2)
    return loss

# https://discuss.pytorch.org/t/non-linear-regression-methods/62060/2
# https://www.programmersought.com/article/16614780143/
def hingeLoss(outputVal,dataOutput,model):
    #print('outputVal:{} dataOutput:{}'.format(outputVal, dataOutput))
    loss1=torch.sum(torch.clamp(1 - torch.matmul(outputVal.t(),dataOutput),min=0))
    loss2=torch.sum(model.fc2.weight ** 2)  # l2 penalty
    totalLoss=loss1+loss2
    #print('loss1:{} loss2:{}'.format(loss1, loss2))
    return(totalLoss)

def my_loss2(outputVal,dataOutput,model):
    print('outputVal:{} dataOutput:{}'.format(outputVal, dataOutput))
    loss1=torch.abs(outputVal - dataOutput)
    return(loss1)


hyper_param_epoch = 20
hyper_param_batch_train = 1
hyper_param_batch_test = 1
hyper_param_learning_rate = 0.00001

transforms_train = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                       transforms.Resize((256, 256)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                      transforms.Resize((256, 256)),
                                      transforms.ToTensor()])

train_data_set = CustomImageLabelDataset(data_set_path_color="C:/DataSets/Concrete/DeepCrack/train_img", data_set_label="train_csvfile.csv", transforms=transforms_train, do_training=True)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch_train, shuffle=True)

test_data_set = CustomImageLabelDataset(data_set_path_color="C:/DataSets/Concrete/DeepCrack/test_img", data_set_label="test_csvfile.csv", transforms=transforms_test, do_training=True)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch_test, shuffle=False)

# get a sample
item = train_data_set.__getitem__(0)
print('image:{}'.format(item['image'].shape))
print('label:{}'.format(item['label']))
a = item['image'].detach().to('cpu').numpy().transpose(1, 2, 0)
ret,c = cv2.threshold(a * 255,126,255,cv2.THRESH_BINARY_INV)
cv2.imshow('a',a)
cv2.imshow('c',c)
cv2.waitKey(1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

custom_model = Net().to(device)

print(custom_model)
summary(custom_model, (1, 256, 256))

# https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/

# Loss and optimizer
criterion = nn.L1Loss()
#criterion = nn.MSELoss()
#criterion = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
optimizer = torch.optim.RMSprop(custom_model.parameters(), lr=hyper_param_learning_rate)
#optimizer = torch.optim.SGD(custom_model.parameters(), lr=hyper_param_learning_rate)
#optimizer = torch.optim.ASGD(custom_model.parameters(), lr=hyper_param_learning_rate)

do_save = True
# tensorboard --logdir=runs
# default `log_dir` is "runs" - we'll be more specific here
#writer = SummaryWriter('runs/experiment_1')
# get some random training images
#dataiter = iter(train_loader)
#all = dataiter.next()
#print("{}".format(all))
#images = all['image'][0]
# create grid of images
#img_grid = torchvision.utils.make_grid(images)
# write to tensorboard
#writer.add_image('four_fashion_mnist_images', img_grid)
#writer.add_graph(custom_model, images)
#writer.close()

# https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
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

        if do_save:
            do_save = False
            # tensorboard --logdir=runs
            # default `log_dir` is "runs" - we'll be more specific here
            writer = SummaryWriter('runs/experiment_1')
            # get some random training images
            # create grid of images
            img_grid = torchvision.utils.make_grid(images)
            # write to tensorboard
            writer.add_image('four_fashion_mnist_images', img_grid)
            writer.add_graph(custom_model, images)
            writer.close()
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

torch.save(custom_model.state_dict(), "my_demo_model.pth")

# Set the model path
Weight_Path = 'my_demo_model.pth'
# Set the model
TransposeModel(Weight_Path, custom_model, 256, 256, 1)

# Serialize the module
sm = torch.jit.script(custom_model)
sm.save("traced_model.pt")

#print("custom_model w/n")
#for param in custom_model.parameters():
#  print(param.data)

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

        a = images[0].detach().to('cpu').numpy().transpose(1, 2, 0) * 255
        #grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        grayA = a
        ret,c = cv2.threshold(grayA,o,255,cv2.THRESH_BINARY_INV)
        cv2.imshow('a',a)
        cv2.imshow('c',c)
        cv2.waitKey(1)

    print('total:{}'.format(total))



    #1 total:5636
    #5 total:5009
    #10 total:4741
    #20 total:4198  | 4399 4302
    #50 total:6260
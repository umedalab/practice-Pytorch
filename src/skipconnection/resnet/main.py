# https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/

# import required libraries
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# basic resdidual block of ResNet
# This is generic in the sense, it could be used for downsampling of features.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], downsample=None):
        """
        A basic residual block of ResNet
        Parameters
        ----------
            in_channels: Number of channels that the input have
            out_channels: Number of channels that the output have
            stride: strides in convolutional layers
            downsample: A callable to be applied before addition of residual mapping
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride[0], 
            padding=1, bias=False
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride[1], 
            padding=1, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # applying a downsample function before adding it to the output
        if(self.downsample is not None):
            residual = downsample(residual)

        out = F.relu(self.bn(self.conv1(x)))
        
        out = self.bn(self.conv2(out))
        # note that adding residual before activation 
        out = out + residual
        out = F.relu(out)
        return out



# downsample using 1 * 1 convolution
downsample = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
    nn.BatchNorm2d(128)
)
# First five layers of ResNet34
resnet_blocks = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualBlock(64, 64),
    ResidualBlock(64, 64),
    ResidualBlock(64, 128, stride=[2, 1], downsample=downsample)
)

# checking the shape
inputs = torch.rand(1, 3, 100, 100) # single 100 * 100 color image
outputs = resnet_blocks(inputs)
print(outputs.shape)    # shape would be (1, 128, 13, 13)


# one could also use pretrained weights of ResNet trained on ImageNet
resnet34 = torchvision.models.resnet34(pretrained=True)






class Dense_Layer(nn.Module):
    def __init__(self, in_channels, growthrate, bn_size):
        super(Dense_Layer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growthrate, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bn_size * growthrate)
        self.conv2 = nn.Conv2d(
            bn_size * growthrate, growthrate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = self.conv1(F.relu(self.bn1(out1)))
        out2 = self.conv2(F.relu(self.bn2(out1)))
        return out2



class Dense_Block(nn.ModuleDict):
    def __init__(self, n_layers, in_channels, growthrate, bn_size):
        """
        A Dense block consists of `n_layers` of `Dense_Layer`
        Parameters
        ----------
            n_layers: Number of dense layers to be stacked 
            in_channels: Number of input channels for first layer in the block
            growthrate: Growth rate (k) as mentioned in DenseNet paper
            bn_size: Multiplicative factor for # of bottleneck layers
        """
        super(Dense_Block, self).__init__()

        layers = dict()
        for i in range(n_layers):
            layer = Dense_Layer(in_channels + i * growthrate, growthrate, bn_size)
            layers['dense{}'.format(i)] = layer
        
        self.block = nn.ModuleDict(layers)
    
    def forward(self, features):
        if(isinstance(features, torch.Tensor)):
            features = [features]
        
        for _, layer in self.block.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, dim=1)


# a block consists of initial conv layers followed by 6 dense layers
dense_block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(3, 2),
    Dense_Block(6, 64, growthrate=32, bn_size=4),
)

inputs = torch.rand(1, 3, 100, 100)
outputs = dense_block(inputs)
print(outputs.shape)    # shape would be (1, 256, 24, 24)

# one could also use pretrained weights of DenseNet trained on ImageNet
densenet121 = torchvision.models.densenet121(pretrained=True)
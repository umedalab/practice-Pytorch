# practice-Pytorch  
A collection of projects in pytorch.  
The projects are used as practice or base for further research.  

# Howto
Create a virtual environment.  
'''  
Linux
python3 -m venv env  
source env/bin/activate

Windows
python -m venv env
env/Script/activate.bin
'''  

## Required
Install all the third party libraries (Linux, on Windows use pip).  
  pytorch  
  pip3 install opencv-python  
  pip3 install torchsummary  
  pip3 install tensorboard  
  pip3 install tensorboardX  
  pip3 install scikit-image   

Download the dataset DeepCrack:  
https://github.com/yhlleo/DeepCrack  

In a terminal, start virtual environment.  
'''  
tensorboard --logdir experiments  
'''  

##Run
Open two terminals, and for each terminal:  

In a terminal:  
start virtual environment  
source env/bin/activate  
run tensorboard  
tensorboard --logdir experiments  
http://localhost:6006/  
 
In another terminal:  
start virtual environment  
source env/bin/activate  
run samples  

TODO:
 [x] Connect repetitive blocks
 [x] Connect different blocks
 [] Custom loss
 [] Custom node
 [] Network from configuration file
 [] Test transform ONNX an test OpenVINO or other
 [] Connect different modules (classes)

 [x] Save the model pt
 [] Run as C++

#Additional Link

##Network

https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html  
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html  
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  
https://github.com/utkuozbulak/pytorch-cnn-visualizations  
https://github.com/tomjerrygithub/Pytorch_VehicleColorRecognition/blob/master/model.py  
https://www.guru99.com/pytorch-tutorial.html  
https://datascience.stackexchange.com/questions/67549/validation-loss-is-not-decreasing-regression-model  
https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/5  
https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/6  
https://stackoverflow.com/questions/57802632/loss-is-not-converging-in-pytorch-but-does-in-tensorflow  
https://discuss.pytorch.org/t/sloved-why-my-loss-not-decreasing/15924/6  
https://discuss.pytorch.org/t/cnn-does-not-predict-properly-does-not-converge-as-expected/43567  

##Architecture

https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848  

##Visualize Filters

https://discuss.pytorch.org/t/how-to-visualize-the-actual-convolution-filters-in-cnn/13850/11  

##Loss

https://discuss.pytorch.org/t/non-linear-regression-methods/62060/2  
https://www.programmersought.com/article/16614780143/  

'''
def hingeLoss(outputVal,dataOutput,model):
    #print('outputVal:{} dataOutput:{}'.format(outputVal, dataOutput))
    loss1=torch.sum(torch.clamp(1 - torch.matmul(outputVal.t(),dataOutput),min=0))
    loss2=torch.sum(model.fc2.weight ** 2)  # l2 penalty
    totalLoss=loss1+loss2
    #print('loss1:{} loss2:{}'.format(loss1, loss2))
    return(totalLoss)
'''
https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7  

##Activation Functions
https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/  

##Other
http://alexlenail.me/NN-SVG/LeNet.html  
https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848#:~:text=VGG%2D16%20is%20a%20simpler,2%20with%20stride%20of%202.&text=The%20winner%20of%20ILSVRC%202014,also%20known%20as%20Inception%20Module.  
https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/#:~:text=The%20VGG%20network%20architecture%20was,each%20other%20in%20increasing%20depth.  
https://pytorch.org/hub/pytorch_vision_vgg/  
https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch  
https://stackoverflow.com/questions/53114882/pytorch-modifying-vgg16-architecture  
https://stackoverflow.com/questions/52963909/pytorch-getting-the-correct-dimensions-for-final-layer/52981440#52981440  
https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d  
https://discuss.pytorch.org/t/vgg-16-architecture/27024/4  
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py  

##Pytorch tutorial (beginner)
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html  

##Pytorch concatenate models
https://discuss.pytorch.org/t/combine-two-model-on-pytorch/47858/6  

import torchvision
from torch import nn

vgg16_True = torchvision.models.vgg16(pretrained=True)
vgg16_False = torchvision.models.vgg16(pretrained=False)

#不跑了，会下载500多M的预训练参数，会写会改就行

print(vgg16_True)

dataset = torchvision.datasets.CIFAR10("./dataset",train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

#会加在vgg16最外面，CIFAR10只有10个类别，所以让输出变成10个类别
vgg16_True.add_module("add_linear", nn.Linear(1000,10))

#加在Classfier里
vgg16_True.classifier.add_module(("add_linear",nn.Linear(1000, 10)))

print(vgg16_False)

#修改最后一个Linear的输出1000到10
vgg16_False.classifier[6] = nn.Linear(4096, 10)





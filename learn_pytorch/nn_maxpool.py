# 最大池化的作用：
# 最大程度保留input的特征，并将输入减小。例子：1080P的视频池化成720P，减小数据输入的同时，满足了大部分需求

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=64)

#part 1 testing dataset
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # ceil_mode 表示 input中每一步，如不够对应池化核个数，比如2.6，向上取整就是3，为TRUE保留多余部分；向下取整就是2，为FALSE舍弃多余部分.
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()

writer = SummaryWriter(log_dir="logs_maxpool")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("inputs",imgs,step)
    output = tudui(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()


# part 1 testing
# output = tudui(input)
# print(output)



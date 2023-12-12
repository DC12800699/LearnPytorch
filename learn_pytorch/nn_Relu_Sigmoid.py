import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        #inplace默认false, 不替换原有input值，如input=-1，根据ReLU原理，input=-1 output=0； 如为TRUE，则 input直接替换成0
        #多看手册：https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        self.RelU1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.RelU1(input)
        #让网络只经过sigmoid
        output = self.sigmoid1(input)
        return output

tudui = Tudui()
# part1 : testing ReLU
# output = tudui(input)
# print(output)
#把所有负数元素都变成0了


dataset = torchvision.datasets.CIFAR10("./dataset",train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

writer = SummaryWriter(log_dir="logs_Sigmoid")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input_sigmoid", imgs, step)
    output = tudui(imgs)
    writer.add_images("output_sigmoid",output, step)
    step += 1

writer.close()
#非线性变换的目的是给我们的网络引入更多非线性特征。模型的泛化能力会更好。
#只有非线性特征多，才能训练出符合各种曲线，各种特征的模型。



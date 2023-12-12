import torchvision
from torch import nn
from torch.nn import Sequential, MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1, drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

        #替代上面的属性
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)

        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss_cross = nn.CrossEntropyLoss()
tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(output)
    print(targets)
    result_loss = loss_cross(output, targets)
    result_loss.backward()
    print(result_loss)

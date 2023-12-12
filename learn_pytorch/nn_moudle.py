import torch
from torch import nn
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
# __call__里调用了forward方法
output = tudui(x)
print(output)






import torch
from torch.nn import L1Loss, MSELoss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5] , dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))


loss1 = L1Loss(reduction="mean")
# tensor(0.6667) (0+0+2)/3
loss1 = L1Loss(reduction="sum")
# tensor(2.) (0+0+2)
result = loss1(inputs,targets)

#mean square error 均方误差 先求X,Y差的平方，再求平均值

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)
# tensor(1.3333) (0+0+2^2)/3 = 4/3 = 1.3333

print(result)
print(result_mse)


#交叉熵
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))

loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
#tensor(1.1019)
# -0.2+ln(exp(0.1)+exp(0.2)+exp(0.3))


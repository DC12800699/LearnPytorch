# P9 Transforms简介

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
# python的用法 -》 Tensor的数据类型
# 通过Transform.ToTensor去看两个问题
# 1.Transform 该如何使用
# 2.为什么我们需要Tensor数据类型

img_path = "data/train/bees_image/29494643_e3410f0d37.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1.Transform 该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img)
writer.close()

print(tensor_img)
#如opencv安装失败,安装opencv 特定版本代替

#pip3 install opencv-python==4.5.3.56
#if facing some cannotbuild wheel issue,try this, do NOT upgrade to latest version 59.6.0
# pip install setuptools==59.5.0

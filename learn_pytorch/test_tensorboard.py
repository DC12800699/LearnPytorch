# P8 TensorBoard介绍
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

image_path = "data/train/bees_image/17209602_fe5a5a746f.jpg"
img = Image.open(image_path)
print(type(img))
# <class 'PIL.JpegImagePlugin.JpegImageFile'>
img_array = np.array(img)
print(type(img_array))
# <class 'numpy.ndarray'>
#!!!chec my own shape
print(img_array.shape)
# HWC format
# (512, 768, 3)
#default is CHW, if you are using diffrent format(eg: HWC), indicate it in your add_image function
#
writer.add_image("train", img_array, 1, dataformats='HWC')

#y = x
for i in range (100):
    writer.add_scalar("y=2x",
                     #y轴数值 scalar_value
                      3*i,
                      # x轴数值 global_step
                      i)

writer.close()

#######################################
# 注意事项

#1. 首次先运行上面代码，再跑terminal。第二次运行代码，直接去刷新页面更快
#2. terminal里，pytorch环境中，输入以下指令，可看到tensorboard结果：
#       tensorboard --logdir=logs --port=6008
# 默认端口为6006，可以自己指定
#3. 如有同Tag(第一个参数)，但Value 和 step不同的，会有图像拟合，很乱，删掉logs下所有event重新跑就好了
#########################################


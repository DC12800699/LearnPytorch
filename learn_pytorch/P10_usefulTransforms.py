from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("data/1_LLVL8xUiUOBE8WHgzAuY-Q.png")
print(img)

#ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("ToTensor", img_tensor)

#Normalize
# 归一化是为了消除奇异值，及样本数据中与其他数据相比特别大或特别小的数据 这样可以加快训练速度
# 归一化在训练神经网络时候有作用，可以压缩值域，防止梯度爆炸
# normalize 是一种可以加快 梯度下降速度的常规操作
print(f"img_tensor[0][0][0] --{img_tensor[0][0][0]}")
#step1
# trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
#step2
# trans_norm = transforms.Normalize([1, 3, 5],[3, 2, 1])
#step3
trans_norm = transforms.Normalize([6, 3, 2],[9, 3, 5])

img_norm = trans_norm(img_tensor)
print(f"img_norm[0][0][0] --> {img_norm[0][0][0]}")
#step1
# writer.add_image("Normalize",img_norm)
#step2
# writer.add_image("Normalize",img_norm, 1)
#step3
writer.add_image("Normalize",img_norm, 2)


#resize
print(img.size)
trans_resize = transforms.Resize((512,512))
#img PIL --> resize --> img_resize PIL
img_resize = trans_resize(img)
#img_resize PIL --> ToTensor --> img_resize tensor
img_resize = trans_toTensor(img_resize)
writer.add_image("Resize",img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
#PIL --> PIl --> Tensor 特别注意 Compose里的列表中。前后工具的数据类型是否匹配，前一个参数的输出应符合后一个参数的输入。
# eg：trans_resize_2输出为PIL, trans_toTensor输入为PIL。如反过来，则程序报错。
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
# 一个参数时
# trans_random = transforms.RandomCrop(512)
trans_random = transforms.RandomCrop((512,1000))
trans_compose_2 = transforms.Compose([trans_random, trans_toTensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    #一个参数时
    # writer.add_image("RandomCrop",img_crop, i)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()


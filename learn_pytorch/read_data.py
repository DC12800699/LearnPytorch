from torch.utils.data import Dataset
from PIL import Image
import os

#P7最后的练习，当image和label在不同文件夹下，且Label需从txt文档中读取
#视频空降 https://www.bilibili.com/video/BV1hE411t7RN?t=1402.1&p=7
#Read label from txt files
class MyData_2(Dataset):

    def __init__(self, root_dir, image_dir, label_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_dir_path = os.path.join(self.root_dir,self.image_dir)
        self.label_dir_path = os.path.join(self.root_dir,self.label_dir)
        self.img_list = os.listdir(self.image_dir_path)
        self.label_list = os.listdir(self.label_dir_path)


    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        label_name = self.label_list[idx]

        image_path = os.path.join(self.image_dir_path,img_name)
        label_path = os.path.join(self.label_dir_path,label_name)

        img = Image.open(image_path)
        label = ""
        with open(label_path,"r") as f:
            label = f.read()
        return img, label

#MyData 2 testing
root_dir = "dataset/hymenoptera_data/train"
ants_image_dir = "ants_image"
bees_image_dir = "bees_image"
ants_label_dir = "ants_label"
bees_label_dir = "bees_label"

ants_dataset = MyData_2(root_dir,ants_image_dir,ants_label_dir)
img1, label1 = ants_dataset[0]

bees_dataset = MyData_2(root_dir,bees_image_dir,bees_label_dir)
img2, label2 = bees_dataset[0]


#视频演示的例子
class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item_path =os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return  img, label

    def __len__(self):
        return len(self.img_path_list)



#MyData Testing
root_dir = "dataset/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"

ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

# img = bees_dataset[0]
# img, label = bees_dataset[0]
# img.show()

train_dataset = ants_dataset + bees_dataset
len(train_dataset)
# Out[9]: 245
len(ants_dataset)
# Out[10]: 124
len(bees_dataset)
# Out[11]: 121
img, label = train_dataset[123]
img.show()
img, label = train_dataset[124]
img.show()


#using image and label








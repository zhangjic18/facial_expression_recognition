import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 自定义无标签数据集
class DatasetWithoutLabel(Dataset):
    def __init__(self, dir_path="/home/zhangjc/cleared_data/ok/"):
        """
        功能说明：
            ——构造无标签数据集，有img，无label
        参数说明：
        dir_path:
            ——无标签数据集所在的位置
        """

        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.TenCrop((44, 44)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

        self.img_path_list = []
        self.img_list = []
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            self.img_path_list.append(img_path)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            self.img_list.append(img)

    def __getitem__(self, index):
        return self.img_list[index], self.img_path_list[index]

    def __len__(self):
        return len(self.img_list)

    # def get_img_path(self):
    #     return self.img_path_list

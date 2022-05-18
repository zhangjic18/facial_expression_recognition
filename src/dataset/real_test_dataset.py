import os.path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class test_dataset(Dataset):
    def __init__(self, flag="all"):
        assert flag in ["all", "Happy", "Neutral", "Sad"]

        parent = "/home/zhangjc/algorithm/facial_expression_recognition/data/cleared_test/"
        # parent = "/home/zhangjc/algorithm/facial_expression_recognition/data/test/"
        # parent = "/home/zhangjc/algorithm/facial_expression_recognition/data/final_test/"

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.TenCrop((44, 44)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        if flag == "all":
            dir_set = [os.path.join(parent, item) for item in ["Happy", "Neutral", "Sad"]]
        else:
            dir_set = [os.path.join(parent, flag)]

        self.img_path_list = []
        self.label_list = []
        for dir in dir_set:
            for img_name in os.listdir(dir):
                self.img_path_list.append(os.path.join(dir, img_name))
                if dir.split("/")[-1] == "Happy":
                    self.label_list.append(0)
                elif dir.split("/")[-1] == "Neutral":
                    self.label_list.append(2)
                elif dir.split("/")[-1] == "Sad":
                    self.label_list.append(1)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return len(self.label_list)

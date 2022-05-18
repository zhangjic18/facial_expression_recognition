import os.path
import sys

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

sys.path.append("/home/zhangjc/algorithm/facial_expression_recognition/")

from src.dataset.FERPlus import FERPlus


class fine_tuning_dataset(Dataset):
    def __init__(self):
        parent = "/home/zhangjc/algorithm/facial_expression_recognition/data/fine_tuning_data/"

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.TenCrop((44, 44)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

        self.img_path_list = []
        self.label_list = []
        for item in ["positive", "negative", "neutral"]:
            second_path = os.path.join(parent, item)

            for dir in os.listdir(second_path):
                third_path = os.path.join(second_path, dir)

                for img_name in os.listdir(third_path):
                    self.img_path_list.append(os.path.join(third_path, img_name))

                    if item == "positive":
                        self.label_list.append(torch.FloatTensor([1, 0, 0]))
                    elif item == "negative":
                        self.label_list.append(torch.FloatTensor([0, 1, 0]))
                    elif item == "neutral":
                        self.label_list.append(torch.FloatTensor([0, 0, 1]))

        # other_img_path, other_label = FERPlus(flag="train").get_imgpath_label(length=len(self.label_list))
        # self.img_path_list += other_img_path
        # self.label_list += other_label

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)
        label = self.label_list[index]
        label = torch.stack([label for i in range(10)])

        return img, label

    def __len__(self):
        return len(self.label_list)

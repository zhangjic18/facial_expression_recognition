import os.path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class FERPlus(Dataset):
    def __init__(self, flag="train"):
        self.parent = "/home/zhangjc/algorithm/facial_expression_recognition/original_data/FER+/"

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.TenCrop((44, 44)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

        if flag == "train":
            self.dst = os.path.join(self.parent, "train")
        elif flag == "validate":
            self.dst = os.path.join(self.parent, "validate")
        elif flag == "test":
            self.dst = os.path.join(self.parent, "test")

        self.img_name_list = os.listdir(self.dst)

        self.img_label_list = self._get_label_list(flag, self.img_name_list)

    def _get_label_list(self, flag, img_name_list):
        with open(file=os.path.join(self.parent, "label.csv"), mode="r", encoding="utf-8") as f:
            f.readline()
            label_list = f.readlines()

        train_label_dict = {}
        validate_label_dict = {}
        test_label_dict = {}
        for i in range(len(label_list)):
            item_list = label_list[i].strip().split(",")
            if item_list[1] != "":
                temp = [int(character) / 10 for character in item_list[2:-1]]

                emotion_class = temp.index(max(temp))

                if emotion_class == 1 or emotion_class == 2:  # positive
                    result = torch.FloatTensor([1, 0, 0])
                elif emotion_class == 3 or emotion_class == 4 or emotion_class == 5 or emotion_class == 6 or emotion_class == 7:  # negative
                    result = torch.FloatTensor([0, 1, 0])
                elif emotion_class == 0 or emotion_class == 8:  # neutral
                    result = torch.FloatTensor([0, 0, 1])

                if item_list[0] == "Training":
                    train_label_dict[item_list[1]] = result
                elif item_list[0] == "PublicTest":
                    validate_label_dict[item_list[1]] = result
                elif item_list[0] == "PrivateTest":
                    test_label_dict[item_list[1]] = result

        if flag == "train":
            img_label_list = [train_label_dict[item] for item in img_name_list]
        elif flag == "validate":
            img_label_list = [validate_label_dict[item] for item in img_name_list]
        elif flag == "test":
            img_label_list = [test_label_dict[item] for item in img_name_list]

        return img_label_list

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dst, self.img_name_list[index])).convert("RGB")

        img = self.transform(img)
        label = self.img_label_list[index]

        label = torch.stack([label for i in range(10)])

        return img, label

    def __len__(self):
        return len(self.img_label_list)

    # def get_imgpath_label(self):
    #     img_path_list = [os.path.join(self.dst, item) for item in self.img_name_list]
    #
    #     return img_path_list, self.img_label_list

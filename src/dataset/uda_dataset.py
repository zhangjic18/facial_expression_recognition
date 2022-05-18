import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class uda_dataset(Dataset):
    def __init__(self, total_list):
        """
        功能说明：
            ——构造训练集，有img，有label
        参数说明：
        total_list:List[List]:
            ——质量较好的文件list（包含kl差异，文件路径与label），img_path_label_list=[[kl_score,img_path0,img_label0],...]
        """
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.TenCrop((44, 44)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

        self.img_list = []
        self.label_list = []
        for _, img_path, img_label in total_list:
            self.label_list.append(img_label)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            self.img_list.append(img)

        self.len = len(self.img_list)

    def __getitem__(self, index):
        label = self.label_list[index]
        label = torch.tensor(label)
        label = torch.stack([label for i in range(10)])
        return self.img_list[index], label

    def __len__(self):
        return self.len

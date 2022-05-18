import argparse
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg19_bn
from torchvision.models.resnet import *
from torchvision.models.densenet import densenet121

sys.path.append("/home/zhangjc/algorithm/facial_expression_recognition/")

from src.dataset.real_test_dataset import test_dataset
from src.models.from_torch.vgg import vgg19_bn as modified_vgg


def main(args):
    test_dataloader = DataLoader(dataset=test_dataset(flag=args.test_dataset_flag), batch_size=32, shuffle=False)

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    if args.model == "resnet18":
        model = resnet18(num_classes=3)
    elif args.model == "resnet34":
        model = resnet34(num_classes=3)
    elif args.model == "resnet50":
        model = resnet50(num_classes=3)
    elif args.model == "resnet101":
        model = resnet101(num_classes=3)
    elif args.model == "resnet152":
        model = resnet152(num_classes=3)
    elif args.model == "vgg19_bn":
        model = vgg19_bn(num_classes=3)
    elif args.model == "modified_vgg":
        model = modified_vgg()
    elif args.model == "densenet121":
        model = densenet121(num_classes=3)

    model.to(device)
    model.load_state_dict(torch.load(f="../trained_model/fine_tuning/"+args.model+".pth", map_location=device))
    model.eval()

    print("================================real test================================")

    correct = 0
    total = 0
    with torch.no_grad():
        for bat_index, (img, label) in enumerate(test_dataloader):
            img = img.to(device)
            label = label.to(device)
            img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])

            output = model(img)

            output = torch.stack([output[i * 10:(i + 1) * 10].mean(0) for i in range(output.shape[0] // 10)])

            pred = torch.argmax(output, 1)

            correct += (pred == label).sum().item()
            total += label.shape[0]

    print(args.test_dataset_flag + " test accuracy:{:.6f}".format(correct / total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="real test")
    parser.add_argument("--model", type=str, default="vgg19_bn")
    parser.add_argument("--test_dataset_flag", type=str, default="all")
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    main(args=args)

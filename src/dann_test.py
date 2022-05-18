import argparse
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append("/home/zhangjc/algorithm/facial_expression_recognition/")

from src.dataset.real_test_dataset import test_dataset
from src.models.DANN_model import DANN


def main(args):
    test_dataloader = DataLoader(dataset=test_dataset(flag=args.test_dataset_flag), batch_size=32, shuffle=False)

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    model = DANN(device_num=args.device, num_classes=3)
    model.load_state_dict(torch.load(f="../trained_model/dann/dann.pth",
                                     map_location=device))

    model.to(device)
    model.eval()

    print("================================dann test================================")

    correct = 0
    total = 0
    with torch.no_grad():
        for bat_index, (img, label) in enumerate(test_dataloader):
            img = img.to(device)
            label = label.to(device)
            img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])

            output, _ = model(img, 0)

            output = torch.stack([output[i * 10:(i + 1) * 10].mean(0) for i in range(output.shape[0] // 10)])

            pred = torch.argmax(output, 1)

            correct += (pred == label).sum().item()
            total += label.shape[0]

    print(args.test_dataset_flag + " test accuracy:{:.6f}".format(correct / total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dann test")
    parser.add_argument("--test_dataset_flag", type=str, default="all")
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    main(args=args)

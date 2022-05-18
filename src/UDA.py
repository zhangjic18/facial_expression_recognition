import argparse
import json
import math
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models.vgg import vgg19_bn
from torchvision.models.resnet import *
from torchvision.models.densenet import densenet121

sys.path.append("/home/zhangjc/algorithm/facial_expression_recognition/")

from src.dataset.uda_dataset import uda_dataset
from src.dataset.DatasetWithoutLabel import DatasetWithoutLabel


def gen_pseudo_label(args, model, device):
    """
    函数用途：
        ——生成伪标签，并根据score分布与理想分布的差异的大小（KL散度）来排序
    返回值：
    total_list:List[List]:
        ——质量较好的文件list（包含kl差异，文件路径与label），img_path_label_list=[[kl_score,img_path0,img_label0],...]
    """
    print("================================gen_pseudo_label================================")

    model.eval()

    dir_path = "/home/zhangjc/cleared_data/ok/"
    dataset_without_label = DatasetWithoutLabel()
    data_loader = DataLoader(dataset_without_label, batch_size=args.batch_size, shuffle=False)

    total_list = []  # [KL_difference,img_path0,img_label0]
    with torch.no_grad():
        for img, img_path in data_loader:
            img = img.to(device)
            img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])

            outputs = model(img)
            outputs = torch.stack([outputs[i * 10: (i + 1) * 10].mean(0) for i in range(outputs.shape[0] // 10)])

            pred = outputs.argmax(dim=1)

            log_p_out = F.log_softmax(outputs, dim=1)

            one_hot = torch.zeros(pred.shape[0], 3)
            for i in range(len(pred)):
                one_hot[i, int(pred[i].item())] = 1

            kl_list = [0] * pred.shape[0]
            for i in range(len(kl_list)):
                kl_list[i] = F.kl_div(log_p_out.cpu()[i], one_hot.cpu()[i]).item()

            img_path_list = list(img_path)
            pred_list = [item.item() for item in pred]

            total_list += zip(kl_list, img_path_list, pred_list)

    total_list.sort()

    json_name = "./" + dir_path[1:].replace("/", "_") + ".json"
    with open(json_name, "w") as f:
        json.dump(obj=total_list, fp=f, ensure_ascii=True, indent=2)

    return total_list


def uda(args, model, device, part_list):
    print("================================uda================================")

    model.train()

    train_dataset = uda_dataset(part_list)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    for epoch in range(args.epoch):
        print("epoch:", epoch)

        correct, total = 0., 0.
        num = 0.
        for (img, label) in train_loader:
            num += 1
            img = img.to(device)
            label = label.to(device)

            img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
            label = label.reshape(-1)

            output = model(img)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (label == output.argmax(dim=1)).sum().item()
            total += output.shape[0]

        print('train epoch: {} / {}, image num: {},loss: {:.6f}, lr: {:.8f}, accuracy: {:.6f}'
              .format(epoch, args.epoch, total, loss.item(), optimizer.param_groups[0]['lr'], correct / total))

        scheduler.step()


def main(args):
    print("================================unsupervised domain adaptation================================")

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
    elif args.model == "densenet121":
        model = densenet121(num_classes=3)

    model.to(device)
    model.load_state_dict(torch.load(f="../trained_model/trained_from_scratch/" + args.model + ".pth",
                                     map_location=device))

    total_list = gen_pseudo_label(args, model, device)

    # with open(file='./home_zhangjc_cleared_data_ok_.json', mode='r', encoding='utf-8') as f:
    #     total_list = json.load(f)

    part_list = total_list[:math.floor(args.percentage * len(total_list))]

    uda(args, model, device, part_list)

    label = str(args.percentage).replace(".", "_")
    torch.save(model.state_dict(), '../trained_model/uda/' + args.model + "_" + label + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unsupervised domain adaptation")
    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--percentage", type=float, default=0.4)

    args = parser.parse_args()
    main(args=args)

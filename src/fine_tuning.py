import argparse
import sys
import torch.optim
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg19_bn
from torchvision.models.resnet import *
from torchvision.models.densenet import densenet121

sys.path.append("/home/zhangjc/algorithm/facial_expression_recognition/")

from src.dataset.fine_tuning_dataset import fine_tuning_dataset


def main(args):
    fine_tuning_dataloader = DataLoader(dataset=fine_tuning_dataset(), batch_size=args.batch_size, shuffle=True)

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

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 25], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    fine_tuning(args, model, fine_tuning_dataloader, criterion, optimizer, scheduler, device)


def fine_tuning(args, model, fine_tuning_dataloader, criterion, optimizer, scheduler, device):
    model.train()

    for epoch in range(args.epoch):
        correct = 0
        total = 0

        for batch_idx, (img, label) in enumerate(fine_tuning_dataloader):
            img = img.to(device)
            label = label.to(device)
            img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])

            label = label.reshape(-1, label.shape[2])

            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (torch.argmax(output, 1) == torch.argmax(label, 1)).sum().item()
            total += label.shape[0]

        scheduler.step()

        print("training epoch:{},loss:{:.6f},lr:{:.8f},accuracy:{:.6f}".
              format(epoch, loss.item(), optimizer.param_groups[0]['lr'], correct / total))

    torch.save(model.state_dict(), "../trained_model/fine_tuning/" + args.model + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fine tuning")
    parser.add_argument("--model", type=str, default="vgg19_bn")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=6)

    args = parser.parse_args()
    main(args=args)

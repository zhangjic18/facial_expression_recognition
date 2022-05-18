import argparse
import torch.optim
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg19_bn
from torchvision.models.resnet import *
from torchvision.models.densenet import densenet121

from dataset.FERPlus import FERPlus
from models.from_torch.vgg import vgg19_bn as modified_vgg


def main(args):
    train_dataloader = DataLoader(dataset=FERPlus(flag="train"), batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=FERPlus(flag="validate"), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=FERPlus(flag="test"), batch_size=args.batch_size, shuffle=False)

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

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(80, 250, 5)), gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)
    criterion = torch.nn.CrossEntropyLoss()

    train(args, model, train_dataloader, validation_dataloader, criterion, optimizer, scheduler, device)

    test_accuracy = test(model, test_dataloader, device,
                         model_path="../trained_model/trained_from_scratch/" + args.model + ".pth")
    print("real test accuracy:", test_accuracy)


def train(args, model, train_dataloader, validation_dataloader, criterion, optimizer, scheduler, device):
    model.train()

    best_validate_accuracy = 0.
    best_validate_epoch = -1
    for epoch in range(args.epoch):
        correct = 0
        total = 0

        for batch_idx, (img, label) in enumerate(train_dataloader):
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

        print("training epoch:{},loss:{:.6f},lr:{:.8f},accuracy:{:.6f}".
              format(epoch, loss.item(), optimizer.param_groups[0]['lr'], correct / total))

        val_accuracy = validate(model, validation_dataloader, device)
        print("validate accuracy:{:.6f}".format(val_accuracy))

        if val_accuracy > best_validate_accuracy:
            torch.save(model.state_dict(), "../trained_model/trained_from_scratch/" + args.model + ".pth")
            best_validate_accuracy = val_accuracy
            best_validate_epoch = epoch

        scheduler.step(best_validate_accuracy)

    print("best_validate_epoch:", best_validate_epoch)
    print("best_validate_accuracy:", best_validate_accuracy)


def validate(model, validation_dataloader, device):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(validation_dataloader):
            img = img.to(device)
            label = label.to(device)
            img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
            label = label.reshape(-1, label.shape[2])

            output = model(img)

            output = torch.stack([output[i * 10:(i + 1) * 10].mean(0) for i in range(output.shape[0] // 10)])
            label = torch.stack([label[i * 10:(i + 1) * 10].mean(0) for i in range(label.shape[0] // 10)])

            correct += (torch.argmax(output, 1) == torch.argmax(label, 1)).sum().item()
            total += label.shape[0]

    return correct / total


def test(model, test_dataloader, device, model_path):
    model.load_state_dict(torch.load(model_path))

    return validate(model, test_dataloader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Fer2013+ CNN Training")
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=6)

    args = parser.parse_args()
    main(args=args)

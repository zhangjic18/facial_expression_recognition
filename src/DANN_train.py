import argparse
import math
import sys
import torch.optim
from torch.utils.data import DataLoader

sys.path.append("/home/zhangjc/algorithm/facial_expression_recognition/")

from src.dataset.FERPlus import FERPlus
from src.dataset.DatasetWithoutLabel import DatasetWithoutLabel
from src.models.DANN_model import DANN
from src.utils import AverageMeter


def main(args):
    src_train_dataloader = DataLoader(dataset=FERPlus(flag="train"),
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=True)
    tgt_train_dataloader = DataLoader(dataset=DatasetWithoutLabel(),
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=True)
    min_length = min(len(src_train_dataloader), len(tgt_train_dataloader))
    src_train_dataloader = list(src_train_dataloader)[:min_length]
    tgt_train_dataloader = list(tgt_train_dataloader)[:min_length]

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    model = DANN(device_num=args.device, num_classes=3)

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    lr_list = [args.learning_rate / math.pow(1 + 10 * epoch / args.epoch, 0.75) for epoch in range(args.epoch)]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(50, 250, 5)), gamma=0.8)
    criterion = torch.nn.CrossEntropyLoss()

    train(args, model, src_train_dataloader, tgt_train_dataloader, criterion, optimizer, scheduler, device, lr_list)


def train(args, model, src_train_dataloader, tgt_train_dataloader, criterion, optimizer, scheduler, device, lr_list):
    model.train()

    for epoch in range(args.epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_list[epoch]

        task_pred_correct, task_pred_total = 0, 0
        domain_pred_correct, domain_pred_total = 0, 0

        average_task_loss = AverageMeter()
        average_domain_loss = AverageMeter()

        gamma = 10
        alpha = 2 / (1 + math.exp(-gamma * epoch / args.epoch)) - 1

        for src_data, tgt_data in zip(src_train_dataloader, tgt_train_dataloader):
            src_img = src_data[0].to(device)
            src_label = src_data[1].to(device)

            tgt_img = tgt_data[0].to(device)

            src_img = src_img.reshape(-1, src_img.shape[2], src_img.shape[3], src_img.shape[4])
            tgt_img = tgt_img.reshape(-1, tgt_img.shape[2], tgt_img.shape[3], tgt_img.shape[4])
            src_label = src_label.reshape(-1, src_label.shape[2])

            src_domain = torch.stack([torch.FloatTensor([1., 0.])] * src_img.shape[0]).to(device)
            tgt_domain = torch.stack([torch.FloatTensor([0., 1.])] * tgt_img.shape[0]).to(device)

            src_pred_label, src_pred_domain = model(src_img, alpha)
            _, tgt_pred_domain = model(tgt_img, alpha)

            task_loss = criterion(src_pred_label, src_label)
            average_task_loss.update(task_loss.item())

            domain_loss = criterion(src_pred_domain, src_domain) + criterion(tgt_pred_domain, tgt_domain)
            average_domain_loss.update(domain_loss.item())

            loss = task_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            task_pred_correct += (torch.argmax(src_pred_label, 1) == torch.argmax(src_label, 1)).sum().item()
            task_pred_total += src_label.shape[0]

            domain_pred_correct += (torch.argmax(src_pred_domain, 1) == torch.argmax(src_domain, 1)).sum().item()
            domain_pred_correct += (torch.argmax(tgt_pred_domain, 1) == torch.argmax(tgt_domain, 1)).sum().item()
            domain_pred_total += src_domain.shape[0] + tgt_domain.shape[0]

        # scheduler.step()

        print("training epoch: {}/{}, task loss: {:.6f}, domain loss: {:.6f}, lr: {:.8f}, "
              "alpha: {:.6f}, task pred accuracy: {:.6f}, domain pred accuracy: {:.6f}".
              format(epoch, args.epoch, average_task_loss.avg, average_domain_loss.avg, optimizer.param_groups[0]['lr'],
                     alpha, task_pred_correct / task_pred_total, domain_pred_correct / domain_pred_total))

    torch.save(model.state_dict(), "../trained_model/dann/" + "dann.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="domain adversarial neural network")
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    main(args=args)

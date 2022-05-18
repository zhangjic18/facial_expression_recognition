from torch.autograd import Function
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19_bn


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, device_num, num_classes=3):
        super().__init__()

        pretrained_vgg = vgg19_bn(num_classes=num_classes)

        # device = torch.device("cuda:" + str(device_num) if torch.cuda.is_available() else "cpu")
        # pretrained_vgg.load_state_dict(torch.load(
        # f="/home/zhangjc/algorithm/facial_expression_recognition/trained_model/trained_from_scratch/" + "vgg19_bn"
        # + ".pth", map_location=device))

        modules = list(pretrained_vgg.children())[0]

        self.features = nn.Sequential(modules)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.task_classifier = list(pretrained_vgg.children())[-1]

        self.domain_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        self.GRL = GRL()

    def forward(self, x, alpha):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        task_predict = self.task_classifier(x)
        x = GRL.apply(x, alpha)
        domain_predict = self.domain_classifier(x)

        return task_predict, domain_predict

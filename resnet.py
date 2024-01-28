import torch
import torchvision.models as models


class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # 修改最后一个全连接层为输出2个值
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x

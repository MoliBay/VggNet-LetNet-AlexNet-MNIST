import torch
from torch import nn
from torch.nn import functional as F
'''
定义模型：搭建网络模型AlexNet
'''


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=48,
                               kernel_size=11,
                               stride=4,
                               padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=48)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2,
                                     stride=2,
                                     padding=0)
        self.conv2 = nn.Conv2d(in_channels=48,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=192,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=192)
        self.conv4 = nn.Conv2d(in_channels=192,
                               out_channels=192,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=192)
        self.conv5 = nn.Conv2d(in_channels=192,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,
                                     stride=2,
                                     padding=0)
        self.linear1 = nn.Linear(in_features=128 * 6 * 6,
                                 out_features=2048)
        self.linear2 = nn.Linear(in_features=2048,
                                 out_features=2048)
        self.linear3 = nn.Linear(in_features=2048,
                                 out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    print('AlexNet')
    model = AlexNet()
    imgs = torch.randn(2, 3, 224, 224)
    out = model(imgs)
    print(out.shape)
    print(out)
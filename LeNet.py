import torch
from torch import nn
from torch.nn import functional as F
'''
定义模型：搭建网络模型LeNet
'''

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层用来提取特征，特征图变多
        # 池化层用来减小特征图
        self.conv2d1 = nn.Conv2d(
                                in_channels=3,
                                out_channels=6,
                                kernel_size=5,
                                stride=1,
                                padding=0)
        self.maxp1 = nn.MaxPool2d(
                                kernel_size=2,
                                stride=2)
        # 输入通道数来自上一层的输出通道数
        self.conv2d2 = nn.Conv2d(
                                in_channels=6,
                                out_channels=16,
                                kernel_size=5,
                                stride=1,
                                padding=0)
        self.maxp2 = nn.MaxPool2d(
                                kernel_size=2,
                                stride=2)
        # 全连接层，输出类别
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.maxp1(x)
        x = self.conv2d2(x)
        x = self.maxp2(x)
        # view 展平x, 一维向量
        # relu 激活函数，非线性
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    print('LetNet')
    model = LeNet()
    imgs = torch.randn(2, 3, 32, 32)
    out = model(imgs)
    print(out.shape)
    print(out)


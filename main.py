import torch
from torch import nn
from torch.utils.data import DataLoader

from MyDataset import MyDataset
from VggNet import VggNet


def train():
    for epoch in range(epochs):
        model.train()
        for idx, (batch_X, batch_y) in enumerate(train_dataloader):
            # 数据搬家
            batch_X = batch_X.to(device=device)
            batch_y = batch_y.to(device=device)
            # 正向传播
            y_pred = model(batch_X)
            # 计算损失
            loss = loss_fn(y_pred, batch_y)
            # 梯度下降
            loss.backward()
            # 优化
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()

            # 内部监控
        #             if idx % 100:
        #                 print(loss.item())
        train_acc = get_acc(train_dataloader)
        test_acc = get_acc(test_dataloader)
        print(f"Epoch: {epoch + 1}, Train_Acc: {train_acc}, Test_Acc: {test_acc}")


def get_acc(data_loader):
    model.eval()
    accs = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # 数据搬家
            batch_X = batch_X.to(device=device)
            batch_y = batch_y.to(device=device)
            # 正向传播
            y_pred = model(batch_X)
            # 计算结果
            y_pred = y_pred.argmax(dim=1)
            # 计算准确率
            acc = (y_pred == batch_y).to(dtype=torch.float32).mean().item()
            accs.append(acc)
        result = torch.tensor(data=accs, dtype=torch.float32).mean().item()
        return round(number=result, ndigits=5)


if __name__ == '__main__':

    # 检测设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 构建模型
    model = VggNet()
    model.to(device=device)

    # 数据集目录
    train_root = "./MNIST/train/"
    test_root = "./MNIST/test/"
    # 打包训练集
    train_dataset = MyDataset(data_root=train_root, img_size=(224, 224))
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=32)
    # 打包测试集
    test_dataset = MyDataset(data_root=test_root, img_size=(224, 224))
    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=32)
    # 定义优化函数
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # 损失函数 交叉商损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 训练轮次
    epochs = 10

    train()

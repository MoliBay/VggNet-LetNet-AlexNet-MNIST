import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_root, img_size):
        self.data_root = data_root
        self.img_paths = []
        self.img_labels = []
        self._read_data_info()
        self.img_size = img_size

    def _read_data_info(self):
        for label in os.listdir(self.data_root):
            img_root = os.path.join(self.data_root, label)
            for img in os.listdir(img_root):
                img_path = os.path.join(img_root, img)
                # 图像路径
                self.img_paths.append(img_path)
                # 对应的标签
                self.img_labels.append(label)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_label = self.img_labels[idx]
        img = Image.open(fp=img_path)
        # 把图像转为模型的输入
        img = np.array(img.resize(size=self.img_size))
        img = torch.tensor(data=img, dtype=torch.float32)
        img = torch.permute(input=img, dims=(2, 0, 1))
        label = torch.tensor(data=int(img_label), dtype=torch.long)
        return img, label
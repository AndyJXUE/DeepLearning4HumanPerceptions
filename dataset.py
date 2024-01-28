import pandas as pd
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 自定义数据集


class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_dir, self.dataframe.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        lb = self.dataframe.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        # 将标签转换为torch.long类型
        lb = torch.tensor(lb, dtype=torch.long)

        return image, lb

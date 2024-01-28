import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.tensorboard import SummaryWriter  # 使用PyTorch的SummaryWriter
from torch.utils.data import DataLoader
from dataset import CustomDataset
from resnet import ClassificationModel
from tqdm import tqdm

# 创建tensorboardX的writer
writer = SummaryWriter()

num_epochs = 100  # 训练轮数
csv_file = r'D:\Data\Place Pulse 2.0\binary_q\beautiful.csv'  # CSV文件路径
dataframe = pd.read_csv(csv_file)
img_dir = r'D:\Data\Place Pulse 2.0\final_photo_dataset'  # 图片文件夹路径

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
dataset = CustomDataset(dataframe=dataframe,
                        img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ClassificationModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
for epoch in range(num_epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    total_loss = 0
    correct = 0
    total = 0

    for idx, (inputs, labels) in loop:
        # 前向传播和计算损失
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 计算并记录损失和准确率
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=total_loss / total, accuracy=f'{accuracy:.2f}%')

        # 将损失和精度写入tensorboardX
        writer.add_scalar('Loss/train', loss.item(),
                          epoch * len(dataloader) + idx)
        writer.add_scalar('Accuracy/train', accuracy,
                          epoch * len(dataloader) + idx)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 关闭writer
writer.close()

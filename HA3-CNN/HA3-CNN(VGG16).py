"""
Hands-on Assignment 3
实现一个基于CNN的图像分类器，没有参考的code，需要自己从头写。在iris数据集上训练并预测，用pytorch实现。

要求：
1.CNN的具体网络结构选：基本的CNN（convolution+pooling）和VGG。
2.优化算法先尝试sgd，再尝试adam。
3.效果要能收敛，自己认为效果正常。
4.打印出在训练集和测试集上的效果，把代码及打印出的运行结果截图（上述一共4组实验），一起发给我。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision.models as models


class IrisDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath, header=None, skiprows=1)

        self.X = data.iloc[:, :-1].values.astype(np.float32)
        self.y = data.iloc[:, -1].values.astype(np.int64)

        self.X = self.X.reshape(-1, 1, 2, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# CNN
class BasicCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(BasicCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2), 
            nn.ReLU(),
            nn.MaxPool2d(1)            
        )
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        # 展平
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

# VGG16
class VGG16_Iris(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16_Iris, self).__init__()
        #调用VGG16
        self.vgg16 = models.vgg16(pretrained=False)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

        # 上采样，把 (1,2,2) 变成 (1,224,224)
        self.upsample = nn.Upsample(size=(224,224), mode='bilinear')

    def forward(self, x):
        x = self.upsample(x)       
        x = x.repeat(1,3,1,1)     
        x = self.vgg16(x)       
        return x


def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=30):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # 测试集
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        test_acc = correct / total

        # 每个epoch打印
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return train_acc, test_acc


if __name__ == "__main__":
    train_dataset = IrisDataset("dataset/iris_training.csv")
    test_dataset = IrisDataset("dataset/iris_test.csv")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    experiments = [
        ("BasicCNN", "SGD"),
        ("BasicCNN", "Adam"),
        ("VGG16", "SGD"),
        ("VGG16", "Adam"),
    ]
    
    #排列组合 模型和优化器
    for model_name, opt_name in experiments:
        print(f"\n===== Running {model_name} with {opt_name} =====")
        if model_name == "BasicCNN":
            model = BasicCNN()
        elif model_name == "VGG16":
            model = VGG16_Iris()

        criterion = nn.CrossEntropyLoss()
        if opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.01)

        train_acc, test_acc = train_model(model, train_loader, test_loader, optimizer, criterion, epochs=50)
        print(f"Result: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

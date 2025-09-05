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


#VGG
class VGG16_Iris_Improved(nn.Module):
    def __init__(self, num_classes=3, dropout_p=0.5):
        super(VGG16_Iris_Improved, self).__init__()

        self.vgg16 = models.vgg16(pretrained=False)

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

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

        # 每个 epoch 打印
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return train_acc, test_acc


if __name__ == "__main__":
    train_dataset = IrisDataset("dataset/iris_training.csv")
    test_dataset = IrisDataset("dataset/iris_test.csv")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    experiments = [
        ("VGG16", "SGD"),
        ("VGG16", "Adam"),
    ]

    for model_name, opt_name in experiments:
        print(f"\n===== Running {model_name} with {opt_name} =====")

        model = VGG16_Iris_Improved()

        criterion = nn.CrossEntropyLoss()

        if opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


        train_acc, test_acc = train_model(model, train_loader, test_loader, optimizer, criterion, epochs=50)
        print(f"Result: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

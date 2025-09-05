"""
1、从论文找到Yelp数据集
从网上下载Yelp数据集（根据这篇文章找https://aclanthology.org/2021.acl-long.8.pdf）
实现一个基于RNN的文本分类器，没有参考的code，需要自己从头写。用pytorch实现。

1.CNN的具体网络结构选：基本的RNN、GRU、LSTM，并对比这三种的效果。
2.优化算法先尝试sgd，再尝试adam，并对比其效果。
3.效果要能收敛，自己认为效果正常。
4.打印出在训练集和测试集上的效果，把代码及打印出的运行结果截图（上述一共6组实验），一起发给我。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

#数据加载
class SentimentDataset(Dataset):
    #和python任务一差不多，先分词，然后统计词频排序，然后把句子单词替换成索引
    def __init__(self, pos_file, neg_file, vocab=None, max_length=None):
        self.sentences = []
        self.labels = []

        #负面样本
        with open(neg_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.sentences.append(line.strip())
                self.labels.append(0)

        #正面样本
        with open(pos_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.sentences.append(line.strip())
                self.labels.append(1)

        # 分词变成单词列表
        self.tokenized = [sent.split() for sent in self.sentences]

        # 按词频从高到低存入vocab
        if vocab is None:
            counter = Counter()
            for tokens in self.tokenized:
                counter.update(tokens)
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            #降序
            for word, _ in counter.most_common():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
            self.vocab_size = len(self.vocab)
        else:
            #用传经来的表
            self.vocab = vocab
            self.vocab_size = len(vocab)

        # 确定最大长度
        self.max_length = max_length or max(len(tokens) for tokens in self.tokenized)

        # 编码、填充
        self.data = []
        for tokens in self.tokenized:
            indexed = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
            #统一长度
            if len(indexed) > self.max_length:
                indexed = indexed[:self.max_length]
            else:
                indexed += [self.vocab['<PAD>']] * (self.max_length - len(indexed))
            self.data.append(indexed)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), self.labels[idx]



# 模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, rnn_type='RNN'):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'RNN', 'GRU', or 'LSTM'")

        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)  
        rnn_out, hidden = self.rnn(embedded)  
        if isinstance(hidden, tuple): 
            hidden = hidden[0]
        h = hidden[-1] 
        h = self.dropout(h)
        out = self.fc(h)
        return out

#训练
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for texts, labels in dataloader:
        # 直接使用，无需 .to()
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc

#测试
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc




if __name__ == "__main__":
    train_neg = './dataset/yelp/sentiment.train.0'
    train_pos = './dataset/yelp/sentiment.train.1'
    test_neg = './dataset/yelp/sentiment.test.0'
    test_pos = './dataset/yelp/sentiment.test.1'

    # 加载训练集 建词汇表
    train_dataset = SentimentDataset(train_pos, train_neg)
    #词汇表
    vocab = train_dataset.vocab
    max_length = train_dataset.max_length
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}, Max sequence length: {max_length}")

    # 使用相同词汇表编码测试集
    test_dataset = SentimentDataset(test_pos, test_neg, vocab=vocab, max_length=max_length)

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rnn_types = ['RNN', 'GRU', 'LSTM']
    optimizers = ['SGD', 'Adam']

    results = []

    #排列组合 模型和优化器
    for rnn_type in rnn_types:
        for opt_name in optimizers:
            print(f"\n" + "============================================================")
            print(f"Training {rnn_type} with {opt_name}")
            print("============================================================")

            # 初始化模型
            model = RNNClassifier(vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, rnn_type=rnn_type)

            criterion = nn.CrossEntropyLoss()
            if opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.01)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 训练参数
            epochs = 10
            best_test_acc = 0.0

            for epoch in range(epochs):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
                test_loss, test_acc = evaluate(model, test_loader, criterion)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                if epoch % 5 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch+1:2d}/{epochs} | "
                          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

            # 记录最终结果
            results.append({
                'Model': rnn_type,
                'Optimizer': opt_name,
                'Final_Train_Acc': train_acc,
                'Final_Test_Acc': test_acc,
                'Best_Test_Acc': best_test_acc
            })

            print(f"{rnn_type} + {opt_name} finished. Best Test Acc: {best_test_acc:.4f}")

    # 打印总结果表
    print("\n" + "======================================================================")
    print("总结")
    print("======================================================================")
    print(f"{'Model':<6} {'Optimizer':<8} {'Train Acc':<10} {'Best Test Acc':<12}")
    print("----------------------------------------------------------------------")
    for res in results:
        print(f"{res['Model']:<6} {res['Optimizer']:<8} {res['Final_Train_Acc']:<10.4f} {res['Best_Test_Acc']:<12.4f}")




import torch
from sklearn.metrics import f1_score
import os
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from osgeo import gdal
from osgeo import osr
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import math

bidirectional_set = True
bidirectional = 2 if bidirectional_set else 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_list = []
accracy_list = []
iteration_list = []


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=21):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算 div_term 时处理奇数 d_model 的情况
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:div_term.size(0) - 1])  # 处理奇数情况
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# 定义Transformer编码器和LSTM压缩器
class TransformerLSTMEncoder(nn.Module):
    def __init__(self, input_size, nhead, lstm_input_size, num_layers, lstm_hidden_size, lstm_hidden_size2, layer_dim, output_dim):
        super(TransformerLSTMEncoder, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim1 = lstm_hidden_size
        self.hidden_dim2 = lstm_hidden_size2

        # 位置编码
        self.pos_encoder = PositionalEncoding(input_size)

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead), num_layers=num_layers)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, layer_dim, batch_first=True,
                            bidirectional=bidirectional_set)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.lstm2 = nn.LSTM(lstm_hidden_size * bidirectional, lstm_hidden_size2, layer_dim, batch_first=True,
                             bidirectional=bidirectional_set)
        self.fc = nn.Linear(lstm_hidden_size2 * bidirectional, output_dim)

    def forward(self, x):
        # 使用Transformer编码器对数据信息进行提取
        x = self.pos_encoder(x.unsqueeze(0))
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)

        # 初始化隐层状态全为0
        h1 = torch.zeros(self.layer_dim * bidirectional, x.size(0), self.hidden_dim1).requires_grad_().to(device)
        c1 = torch.zeros(self.layer_dim * bidirectional, x.size(0), self.hidden_dim1).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h1.detach(), c1.detach()))  # 将输入数据和初始化隐层、记忆单元信息传入

        out = self.dropout(out)

        h2 = torch.zeros(self.layer_dim * bidirectional, out.size(0), self.hidden_dim2).requires_grad_().to(device)
        c2 = torch.zeros(self.layer_dim * bidirectional, out.size(0), self.hidden_dim2).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out_lstm2, (hn, cn) = self.lstm2(out, (h2.detach(), c2.detach()))  # 将输入数据和初始化隐层、记忆单元信息传入
        out = self.fc(out_lstm2)

        return out[:, -1, :], out_lstm2


# 定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, filepath):
        print(f'reading {filepath}')

        df = pd.read_csv(
            filepath, header=0, index_col=0,
            encoding='utf-8',
            dtype={'label': np.int32}
        )
        feat = df.iloc[:, :].values[:, 1:22]  # 逆序读取列，时间顺序
        print(f'the shape of feature is {feat.shape}')
        label = df.iloc[:, 0].values

        self.x = torch.from_numpy(feat).float().to(device)
        self.y = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# 定义一个函数以提取降维后的特征向量
def extract_features(model, dataloader):
    with torch.no_grad():
        for wdrvi, _ in dataloader:
            # 将整个序列输入到Transformer编码器和LSTM压缩器中，得到降维后的特征向量
            _, features = model(wdrvi)
            return features.cpu().numpy()


if __name__ == "__main__":
    print(device)
    # 加载数据集
    dataset = TimeSeriesDataset(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI.csv')
    trainset, testset = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(trainset, batch_size=20, shuffle=True)
    test_loader = DataLoader(testset, batch_size=20, shuffle=True)

    # 训练模型
    model = TransformerLSTMEncoder(input_size=21, nhead=3, lstm_input_size=21, num_layers=2, lstm_hidden_size=54,
                                   lstm_hidden_size2=10, layer_dim=2, output_dim=2).to(device)  # d_model=20,
    # input_size = 70, nhead = 5, num_layers = 2, lstm_hidden_size = 35,
    # lstm_hidden_size2 = 20, layer_dim = 1, output_dim = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    iter = 0
    for epoch in range(10000):
        for i, (wdrvi, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            #  前向传播
            outputs, _ = model(wdrvi)
            #  计算损失
            loss = criterion(outputs, labels.long())
            #  反向传播
            loss.backward()
            #  更新参数
            optimizer.step()
            iter += 1
            if (iter-1) % 10 == 0:
                model.eval()
                correct = 0
                total = 0
                for wdrvi, labels in test_loader:
                    outputs, _  = model(wdrvi)
                    # print(outputs)
                    predict = torch.max(outputs, 1)[1]
                    # print(predict)
                    total += labels.size(0)
                    correct += (predict == labels.to(device)).sum()
                    loss_test = criterion(outputs, labels.long())

                accrucy = correct / total
                loss_list.append(loss_test.data.cpu())
                accracy_list.append(accrucy.cpu())
                iteration_list.append(iter)

                print('loop:{} Loss:{} Accuracy:{}'.format(iter, loss.item(), accrucy))
    plt.plot(iteration_list, loss_list)
    plt.xlabel('Number of Iteraion')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(iteration_list, accracy_list)
    plt.xlabel('Number of Iteraion')
    plt.ylabel('Accuracy')
    plt.show()

    # 提取整个数据集的降维特征向量
    # features = extract_features(model, test_loader)
    PATH = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\model_wdrvi_seq1.pt'
    torch.save(model, PATH)

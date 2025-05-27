import torch
from torch.utils.data import Subset, random_split
from sklearn.metrics import f1_score
import os
from torch import nn
from torch.utils.data import Dataset, random_split
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
    def __init__(self, input_size, nhead, lstm_input_size, num_layers, lstm_hidden_size, lstm_hidden_size2, lstm_hidden_size3,layer_dim, output_dim):
        super(TransformerLSTMEncoder, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim1 = lstm_hidden_size
        self.hidden_dim2 = lstm_hidden_size2
        self.hidden_dim3 = lstm_hidden_size3

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
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        self.lstm3 = nn.LSTM(lstm_hidden_size2 * bidirectional, lstm_hidden_size3, layer_dim, batch_first=True,
                             bidirectional=bidirectional_set)

        self.fc = nn.Linear(lstm_hidden_size3 * bidirectional, output_dim)

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

        h3 = torch.zeros(self.layer_dim * bidirectional, out.size(0), self.hidden_dim3).requires_grad_().to(device)
        c3 = torch.zeros(self.layer_dim * bidirectional, out.size(0), self.hidden_dim3).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out_lstm3, (hn, cn) = self.lstm3(out_lstm2, (h3.detach(), c3.detach()))  # 将输入数据和初始化隐层、记忆单元信息传入

        out = self.fc(out_lstm3)

        return out.squeeze(1) , out_lstm3


# 定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, filepath):
        print(f'reading {filepath}')

        df = pd.read_csv(
            filepath, header=0, index_col=0,
            encoding='utf-8',
            dtype={'label': np.int32}
        )
        sorted_columns = sorted(df.columns)
        df = df[sorted_columns]
        feat = df.iloc[:, :].values[:, 0:24]  # 逆序读取列，时间顺序
        print(f'the shape of feature is {feat.shape}')
        label = df['label'].values

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
    dataset = TimeSeriesDataset(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI_sample_merge方案5.csv')

    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=18)
    val_idx, test_idx = train_test_split(temp_idx, test_size=2 / 3, random_state=18)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练模型
    model = TransformerLSTMEncoder(input_size=24, nhead=8, lstm_input_size=24, num_layers=4, lstm_hidden_size=128,
                                   lstm_hidden_size2=64, lstm_hidden_size3=32, layer_dim=2, output_dim=2).to(device)  # d_model=20,
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=10, verbose=True)

    num_epochs = 1000
    early_stop_patience = 200
    best_val_loss = np.inf
    epochs_no_improve = 0
    iter = 0
    loss_list, accuracy_list, iteration_list = [], [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (wdrvi, labels) in enumerate(train_loader):
            wdrvi, labels = wdrvi.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(wdrvi)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算训练集平均 loss
        avg_train_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for wdrvi, labels in val_loader:
                wdrvi, labels = wdrvi.to(device), labels.to(device)
                outputs, _ = model(wdrvi)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()
                preds = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        iter += 1
        loss_list.append(avg_val_loss)
        accuracy_list.append(accuracy)
        iteration_list.append(iter)

        print(
            f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}')

        # 调整学习率
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), r'E:\\model_wdrvi_input21_seq1_lstm16_batch200_merge方案5.pt')  # 保存最好模型
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print('Early stopping triggered!')
                break
    torch.save(model, r'E:\\model_wdrvi_input24_seq1_lstm16_batch200_merge方案5.pt')  # 保存最好模型
    # 绘制曲线
    fig, ax1 = plt.subplots()

    # 画 loss 曲线（左 y 轴）
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(iteration_list, loss_list, color=color, label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个 y 轴，共享 x 轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(iteration_list, accuracy_list, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和图例
    plt.title('Loss and Accuracy over Epochs')
    plt.show()
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for wdrvi, labels in test_loader:
            wdrvi, labels = wdrvi.to(device), labels.to(device)
            outputs, _ = model(wdrvi)
            preds = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_accuracy = correct / total
    print(f'\nFinal Test Accuracy: {test_accuracy:.4f}')
    # model = TransformerLSTMEncoder(input_size=24, nhead=8, lstm_input_size=24, num_layers=3, lstm_hidden_size=128,
    #                                lstm_hidden_size2=64, lstm_hidden_size3=16, layer_dim=2, output_dim=2).to(
    #     device)  # d_model=20,
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #
    # iter = 0
    # for epoch in range(10000):
    #     for i, (wdrvi, labels) in enumerate(train_loader):
    #         model.train()
    #         optimizer.zero_grad()
    #         labels = labels.to(device)
    #         # 梯度清零
    #         optimizer.zero_grad()
    #         #  前向传播
    #         outputs, _ = model(wdrvi)
    #         #  计算损失
    #         loss = criterion(outputs, labels.long())
    #         #  反向传播
    #         loss.backward()
    #         #  更新参数
    #         optimizer.step()
    #         iter += 1
    #         if (iter - 1) % 10 == 0:
    #             model.eval()
    #             correct = 0
    #             total = 0
    #             for wdrvi, labels in val_loader:
    #                 outputs, _ = model(wdrvi)
    #                 # print(outputs)
    #                 predict = torch.max(outputs, 1)[1]
    #                 # print(predict)
    #                 total += labels.size(0)
    #                 correct += (predict == labels.to(device)).sum()
    #                 loss_test = criterion(outputs, labels.long())
    #
    #             accrucy = correct / total
    #             loss_list.append(loss_test.data.cpu())
    #             accracy_list.append(accrucy.cpu())
    #             iteration_list.append(iter)
    #
    #             print('loop:{} Loss:{} Accuracy:{}'.format(iter, loss.item(), accrucy))
    # # 绘制曲线
    # fig, ax1 = plt.subplots()
    # # 画 loss 曲线（左 y 轴）
    # color = 'tab:red'
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Validation Loss', color=color)
    # ax1.plot(iteration_list, loss_list, color=color, label='Validation Loss')
    # ax1.tick_params(axis='y', labelcolor=color)
    # # 创建第二个 y 轴，共享 x 轴
    # ax2 = ax1.twinx()
    # color = 'tab:blue'
    # ax2.set_ylabel('Validation Accuracy', color=color)
    # ax2.plot(iteration_list, accracy_list, color=color, label='Validation Accuracy')
    # ax2.tick_params(axis='y', labelcolor=color)
    # # 添加标题和图例
    # plt.title('Loss and Accuracy over Epochs')
    # plt.show()
    # torch.save(model.state_dict(), r'E:\\best_model.pt')

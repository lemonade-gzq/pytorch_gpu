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

bidirectional_set = True
bidirectional = 2 if bidirectional_set else 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_list = []
accracy_list = []
iteration_list = []


# 定义Transformer编码器和LSTM压缩器
class TransformerLSTMEncoder(nn.Module):
    # d_model,
    def __init__(self, input_size, nhead, num_layers, lstm_hidden_size, lstm_hidden_size2, layer_dim,
                 output_dim):
        super(TransformerLSTMEncoder, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim1 = lstm_hidden_size
        self.hidden_dim2 = lstm_hidden_size2

        #  构建模型
        # self.linear = nn.Linear(input_size, d_model)
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead),
            num_layers=num_layers)
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, layer_dim, batch_first=True,
                            bidirectional=bidirectional_set)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.lstm2 = nn.LSTM(lstm_hidden_size * bidirectional, lstm_hidden_size2, layer_dim, batch_first=True,
                             bidirectional=bidirectional_set)
        # 全连接层
        self.fc = nn.Linear(lstm_hidden_size2 * bidirectional, output_dim)

    def forward(self, x):
        # x = self.linear(x)
        # 使用Transformer编码器对数据信息进行提取
        x = x.unsqueeze(0)
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

        # 以最后一层隐层状态为输出
        out_liner = self.fc(out_lstm2[:, -1, :])

        return out_liner, out_lstm2


class TIFDataSet(Dataset):
    def __init__(self, filrpath):
        print(f'reading{filrpath}')
        self.tifNameList = os.listdir(filrpath)
        self.tifNameList.sort(reverse=True)
        self.wdrvi_all = np.zeros((6203 * 6273, 18))
        m = 17
        for i in range(len(self.tifNameList)):
            #  判断当前文件是否为HDF文件
            if (os.path.splitext(self.tifNameList[i])[-1] == ".tif"):
                rds = gdal.Open(filrpath + "\\" + self.tifNameList[i])
                cols = rds.RasterXSize
                rows = rds.RasterYSize
                band = rds.GetRasterBand(1)
                data = band.ReadAsArray(0, 0, cols, rows)
                data = data.reshape(-1, 1)
                self.wdrvi_all[:, m] = data[:, 0]
                m -= 1
        self.x = torch.from_numpy(self.wdrvi_all).float().to(device)
        del self.wdrvi_all
        # self.y = torch.from_numpy(label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


# 定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, filepath):
        print(f'reading {filepath}')

        df = pd.read_csv(
            filepath, header=0, index_col=0,
            encoding='utf-8',
            dtype={'label': np.int32}
        )
        feat = df.iloc[:, ::-1].values  # 逆序读取列，时间顺序
        feat = feat[:, 0:18]
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
    model = TransformerLSTMEncoder(input_size=18, nhead=3, num_layers=2, lstm_hidden_size=54,
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
            if iter % 100 == 0:
                model.eval()
                correct = 0
                total = 0
                for wdrvi, labels in test_loader:
                    outputs, out_of_lstm = model(wdrvi)
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
    PATH = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\inputsize18_nhead3_numlayer2_lstmsize18_2_lr1e-3.pt'
    torch.save(model, PATH)

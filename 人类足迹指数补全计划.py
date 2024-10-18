import os
from osgeo import gdal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from osgeo import osr

from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_list = []
iteration_list = []
# 设置文件夹路径
folder_path = r"E:\城市与区域生态\大熊猫和竹\凉山种群动态模拟\三山系\环境数据\人类足迹"
filename = os.listdir(folder_path)

# 按年份读取TIFF文件
data = np.zeros((496, 650, 21))

for i in range(6):
    file_path = os.path.join(folder_path, "dxlhfp{}.tif".format(2005 - i))
    dataset = gdal.Open(file_path)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()  # geotransform
    projection = dataset.GetProjectionRef()  # projection
    band = dataset.GetRasterBand(1)
    band_data = band.ReadAsArray(0, 0, cols, rows)
    np.nan_to_num(band_data, nan=-1, copy=False)
    data[:, :, i] = band_data
data = data.reshape(-1, 6)


class TimeSeriesDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(data[:, 0:5]).float().to(device)
        self.y = torch.from_numpy(data[:, 5]).float().to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index].reshape(-1)


class predictDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(data[:, 1:]).float().to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


class predictDataset2(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(data[:, 2:]).float().to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


# 构造时间序列数据
dataset = TimeSeriesDataset()
trainset, testset = train_test_split(dataset, test_size=0.3, random_state=42)
train_loader = DataLoader(trainset, batch_size=200, shuffle=True)
test_loader = DataLoader(testset, batch_size=200, shuffle=True)
predictset = predictDataset()
predict_loader = DataLoader(predictset, batch_size=200, shuffle=True)


# 定义LSTM压缩器
class LSTMEncoder(nn.Module):
    # d_model,
    def __init__(self, input_size, lstm_hidden_size, lstm_hidden_size2, layer_dim,
                 output_dim):
        super(LSTMEncoder, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim1 = lstm_hidden_size
        self.hidden_dim2 = lstm_hidden_size2

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, layer_dim, batch_first=True,
                            bidirectional=False)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.lstm2 = nn.LSTM(lstm_hidden_size, lstm_hidden_size2, layer_dim, batch_first=True,
                             bidirectional=False)
        # 全连接层
        self.fc = nn.Linear(lstm_hidden_size2, output_dim)

    def forward(self, x):
        # x = self.linear(x)
        # 使用Transformer编码器对数据信息进行提取
        x = x.unsqueeze(0)
        # x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)

        # 初始化隐层状态全为0
        h1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim1).requires_grad_().to(device)
        c1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim1).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h1.detach(), c1.detach()))  # 将输入数据和初始化隐层、记忆单元信息传入

        out = self.dropout(out)

        h2 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim2).requires_grad_().to(device)
        c2 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim2).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out_lstm2, (hn, cn) = self.lstm2(out, (h2.detach(), c2.detach()))  # 将输入数据和初始化隐层、记忆单元信息传入

        # 以最后一层隐层状态为输出
        out_liner = self.fc(out_lstm2[:, -1, :])

        return out_liner



model = LSTMEncoder(input_size=5, lstm_hidden_size=20, lstm_hidden_size2=10, layer_dim=2, output_dim=1).to(
    device)  # d_model=20,
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
iter = 0
for i in range(epochs):
    for seq, labels in train_loader:
        model.train()
        optimizer.zero_grad()
        labels = labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        #  前向传播
        outputs = model(seq)
        #  计算损失
        loss = loss_function(outputs, labels)
        #  反向传播
        loss.backward()
        #  更新参数
        optimizer.step()
    iter += 1
    if iter % 10 == 0:
        model.eval()
        correct = 0
        total = 0
        for test_seq, test_labels in test_loader:
            outputs = model(test_seq)
            total += test_labels.size(0)
            loss_test = loss_function(outputs, test_labels)
        loss_list.append(loss_test.data.cpu())
        iteration_list.append(iter)
        print('loop:{} Loss:{}'.format(iter, loss.item()))

# 预测1999和1998年数据
model.eval()
for i in range(2):
    for seq in predict_loader:
        with torch.no_grad():
            pred = model(seq)
            dtat_new = np.c_(data, pred)
            pred = pred.reshape(496, 650)
            print(f'Predicted value for year {2000 - (i + 1)}: {pred}')

            driver = gdal.GetDriverByName('GTiff')
            dst_filename = r'E:\城市与区域生态\大熊猫和竹\凉山种群动态模拟\三山系\环境数据\人类足迹\dxlhfp{}.tif'.format(2000 - (i + 1))
            dst_ds = driver.Create(dst_filename, 496, 650, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(list(geotransform))
            srs = osr.SpatialReference()
            srs.ImportFromWkt(projection)
            dst_ds.SetProjection(srs.ExportToWkt())
            dst_ds.GetRasterBand(1).WriteArray(pred)

            # 更新数据，以便下一次预测

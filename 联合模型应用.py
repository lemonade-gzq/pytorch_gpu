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
import datetime
import math
from tqdm import tqdm

bidirectional_set = True
bidirectional = 2 if bidirectional_set else 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_list = []
accuracy_list = []
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
    def __init__(self, input_size, nhead, lstm_input_size, num_layers, lstm_hidden_size, lstm_hidden_size2, layer_dim):
        super(TransformerLSTMEncoder, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim1 = lstm_hidden_size
        self.hidden_dim2 = lstm_hidden_size2

        # 位置编码
        self.pos_encoder = PositionalEncoding(input_size)

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead),
            num_layers=num_layers)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, layer_dim, batch_first=True,
                            bidirectional=bidirectional_set)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.lstm2 = nn.LSTM(lstm_hidden_size * bidirectional, lstm_hidden_size2, layer_dim, batch_first=True,
                             bidirectional=bidirectional_set)

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

        return out_lstm2


# 定义联合模型
class CombinedModel(nn.Module):
    def __init__(self, wdrvi_params, nmdi_params, mid_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.wdrvi_model = TransformerLSTMEncoder(**wdrvi_params)
        self.nmdi_model = TransformerLSTMEncoder(**nmdi_params)
        combined_dim = wdrvi_params['lstm_hidden_size2'] * bidirectional + nmdi_params[
            'lstm_hidden_size2'] * bidirectional
        self.fc = nn.Linear(combined_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, wdrvi, nmdi):
        wdrvi_out = self.wdrvi_model(wdrvi)
        nmdi_out = self.nmdi_model(nmdi)
        combined = torch.cat((wdrvi_out, nmdi_out), dim=2)
        mid = self.fc(combined)
        out = self.fc2(mid)
        return out[:, -1, :], combined


# 定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, wdrvi_filepath, nmdi_filepath):
        print(f'reading {wdrvi_filepath} and {nmdi_filepath}')

        wdrvi_df = pd.read_csv(wdrvi_filepath, header=0, index_col=0, encoding='utf-8', dtype={'label': np.int32})
        nmdi_df = pd.read_csv(nmdi_filepath, header=0, index_col=0, encoding='utf-8', dtype={'label': np.int32})

        wdrvi_feat = wdrvi_df.iloc[:, :].values[:, 1:22]
        nmdi_feat = nmdi_df.iloc[:, :].values[:, 1:22]
        label = wdrvi_df.iloc[:, 0].values

        self.wdrvi = torch.from_numpy(wdrvi_feat).float().to(device)
        self.nmdi = torch.from_numpy(nmdi_feat).float().to(device)
        self.y = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.wdrvi[index], self.nmdi[index]


class TIFDataSet(Dataset):
    def __init__(self, filepath_wdrvi, filepath_nmdi):

        self.WdrviNameList, self.NmdiNameList = os.listdir(filepath_wdrvi), os.listdir(filepath_nmdi)
        self.wdrvi_all, self.nmdi_all = np.zeros((6203 * 6273, 21)), np.zeros((6203 * 6273, 21))

        m, n = 0, 0
        for i in range(len(self.WdrviNameList)):
            if (os.path.splitext(self.WdrviNameList[i])[-1] == ".tif"):
                rds = gdal.Open(filepath_wdrvi + "\\" + self.WdrviNameList[i])
                print(f'reading{filepath_wdrvi}' + "\\" + self.WdrviNameList[i])
                cols = rds.RasterXSize
                rows = rds.RasterYSize
                band = rds.GetRasterBand(1)
                data = band.ReadAsArray(0, 0, cols, rows)
                data = data.reshape(-1, 1)
                self.wdrvi_all[:, m] = data[:, 0]
                m += 1
        for i in range(len(self.NmdiNameList)):
            if (os.path.splitext(self.NmdiNameList[i])[-1] == ".tif"):
                print(f'reading{filepath_wdrvi}' + "\\" + self.NmdiNameList[i])
                rds = gdal.Open(filepath_nmdi + "\\" + self.NmdiNameList[i])
                band = rds.GetRasterBand(1)
                data = band.ReadAsArray(0, 0, cols, rows)
                data = data.reshape(-1, 1)
                self.nmdi_all[:, n] = data[:, 0]
                n += 1
        self.wdrvi = torch.from_numpy(self.wdrvi_all).float().to('cpu')
        self.nmdi = torch.from_numpy(self.nmdi_all).float().to('cpu')
        del self.wdrvi_all, self.nmdi_all

    def __len__(self):
        return self.wdrvi.shape[0]

    def __getitem__(self, index):
        return self.wdrvi[index], self.nmdi[index]


class AddDataSet(Dataset):
    def __init__(self, filrpath):
        rds = gdal.Open(filrpath)
        self.geotransform = rds.GetGeoTransform()  # geotransform
        self.projection = rds.GetProjectionRef()  # projection
        self.cols = rds.RasterXSize
        self.rows = rds.RasterYSize
        self.band = rds.GetRasterBand(1)
        self.data = self.band.ReadAsArray(0, 0, self.cols, self.rows)
        self.data = self.data.reshape(-1, 1)
        self.x = torch.from_numpy(self.data.astype("int32")).to(device)
        del self.data
        # self.y = torch.from_numpy(label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


# 定义一个函数以提取降维后的特征向量
def extract_features(model, dataloader):
    with torch.no_grad():
        for wdrvi, _ in dataloader:
            # 将整个序列输入到Transformer编码器和LSTM压缩器中，得到降维后的特征向量
            _, features = model(wdrvi)
            return features.cpu().numpy()


if __name__ == '__main__':
    wdrvi_params = {
        'input_size': 21,
        'nhead': 3,
        'lstm_input_size': 21,
        'num_layers': 2,
        'lstm_hidden_size': 54,
        'lstm_hidden_size2': 10,
        'layer_dim': 2
    }

    nmdi_params = {
        'input_size': 21,
        'nhead': 3,
        'lstm_input_size': 21,
        'num_layers': 2,
        'lstm_hidden_size': 54,
        'lstm_hidden_size2': 10,
        'layer_dim': 2
    }
    starttime = datetime.datetime.now()
    lstm_feat = np.zeros((1, 2 * nmdi_params['lstm_hidden_size2'] * bidirectional))  # 第二层的输出大小*是否双向lstm,2是因为两个变量拼接在一起了
    lstm_feat_4train = np.zeros((1, 2 * nmdi_params['lstm_hidden_size2'] * bidirectional))
    probability_Y = []

    tifset = TIFDataSet(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\植被指数序列\WDRVI",
                        r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\植被指数序列\NMDI")
    pre_loader = DataLoader(tifset, batch_size=200, shuffle=False)

    demset = AddDataSet(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_chm0921re.tif")
    dem_loader = DataLoader(demset, batch_size=200, shuffle=False)

    csvset = TimeSeriesDataset(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\wdrvi.csv',
                               r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\nmdi.csv')
    train_loader = DataLoader(csvset, batch_size=20, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(
        r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\combined_model_seq1.pt')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for wdrvi, nmdi in train_loader:
            _, out_of_lstm = model(wdrvi, nmdi)
            lstm_feat_4train = np.append(lstm_feat_4train,
                                         out_of_lstm.view(out_of_lstm.size(0), -1).cpu().detach().numpy(), 0)
    lstm_feat_4train = np.delete(lstm_feat_4train, 0, 0)
    print("--------用于训练xgb的wdrvi数据经transformer和LSTM结束----------")

    df_4train = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\wdrvi.csv', header=0, index_col=0,
                            encoding='utf-8')
    dem_4train = df_4train.iloc[:, -1].values
    dem_4train = dem_4train.reshape(-1, 1)
    label_4train = df_4train.iloc[:, 0].values
    lstm_feat_4train = np.append(lstm_feat_4train, dem_4train, 1)
    print("--------用于训练xgb的数据准备结束（补充dem在最后一列）----------")

    """-----------开始构建xgb模型------------------"""
    X_train, X_test, y_train, y_test = train_test_split(lstm_feat_4train, label_4train, test_size=0.3, random_state=5,
                                                        shuffle=True)
    params = {'objective': 'reg:logistic', 'booster': 'gbtree', 'max_depth': 3, 'silent': 1}
    clf = XGBClassifier(seed=1024, learning_rate=0.1, max_depth=9, min_child_weight=1, gamma=0,
                        subsample=0.9, colsample_bytree=1, alpha=0.06, reg_lambda=1)
    clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)

    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
    print("AUC 得分 (测试集): %f" % metrics.roc_auc_score(y_test, test_predict))
    print("MSE:", mean_squared_error(y_test, test_predict))
    print("MAE:", mean_absolute_error(y_test, test_predict))
    print("R2:", r2_score(y_test, test_predict))
    print("acc:", accuracy_score(y_test, test_predict))
    print("F1:", f1_score(y_test, test_predict))

    # 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print('The confusion matrix result:\n', confusion_matrix_result)
    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    """-----------xgb模型训练结束------------------"""

    probability_list = []
    with torch.no_grad():
        # 提取特征并进行预测
        for (wdrvi, nmdi), dem in tqdm(zip(pre_loader, dem_loader), total=len(pre_loader), desc="Processing"):
            _, out_of_lstm = model(wdrvi.to(device), nmdi.to(device))
            lstm_features = out_of_lstm.view(out_of_lstm.size(0), -1).cpu()
            dem_features = dem.cpu()
            combined_features = torch.cat((lstm_features, dem_features), dim=1).numpy()  # 直接转换为 NumPy 数组

            # 预测概率
            probabilities = clf.predict_proba(combined_features)[:, 1]
            probability_list.append(probabilities)
    probability_Y = np.concatenate(probability_list)  # 合并所有概率
    probability_Y = probability_Y.reshape(demset.rows, demset.cols)

    # 保存预测结果
    geotransform = demset.geotransform
    projection = demset.projection
    driver = gdal.GetDriverByName('GTiff')
    dst_filename = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\竹子概率_transfomer_lstm_xgb0921_combine_seq1.tif'
    dst_ds = driver.Create(dst_filename, demset.cols, demset.rows, 1, gdal.GDT_Float64)
    dst_ds.SetGeoTransform(list(geotransform))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(probability_Y)

    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)

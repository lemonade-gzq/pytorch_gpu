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
        # Transformer编码器d_model是输入输出的维度
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, device=device),
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
        x = x.unsqueeze(0)  # 第0维度改为的大小改为1
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # 维度换位函数

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
        # self.tifNameList.sort(reverse=True)
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


class ADDDataSet(Dataset):
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


# 定义数据集 只读取指数
class TimeSeriesDataset(Dataset):
    def __init__(self, filepath):
        print(f'reading {filepath}')

        df = pd.read_csv(
            filepath, header=0, index_col=0,
            encoding='utf-8',
            dtype={'label': np.int32}
        )
        feat = df.iloc[:, ::-1].values  # 逆序读取列，时间顺序
        feat = feat[:, 1:19]
        print(f'the shape of feature is {feat.shape}')
        label = df.iloc[:, 0].values

        self.x = torch.from_numpy(feat).float().to(device)
        self.y = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return len(self.y)

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
    starttime = datetime.datetime.now()
    lstm_feat = np.zeros((1, 36))  # 第二层的输出大小*是否双向lstm
    lstm_feat_4train = np.zeros((1, 36))
    probability_Y = []

    tifset = TIFDataSet(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\植被指数序列")
    pre_loader = DataLoader(tifset, batch_size=20, shuffle=False)

    demset = ADDDataSet(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_all_chm.tif")
    dem_loader = DataLoader(demset, batch_size=20, shuffle=False)

    csvset = TimeSeriesDataset(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI.csv')
    train_loader = DataLoader(csvset, batch_size=20, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(
        r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\inputsize18_nhead3_numlayer2_lstmsize18_2_lr1e-3.pt')
    model.to(device)
    model.eval()

    for wdrvi in train_loader:
        _, out_of_lstm = model(wdrvi)
        for j in range(out_of_lstm.shape[0]):
            lstm_feat_4train = np.append(lstm_feat_4train, out_of_lstm[j, :, :].cpu().detach().numpy(), 0)
    lstm_feat_4train = np.delete(lstm_feat_4train, 0, 0)
    print("--------用于训练xgb的wdrvi数据经transformer和LSTM结束----------")

    df_4train = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI.csv', header=0, index_col=0,
                            encoding='utf-8')
    dem_4train = df_4train.iloc[:, -1].values
    dem_4train = dem_4train.reshape(-1, 1)
    label_4train = df_4train.iloc[:, 0].values
    lstm_feat_4train = np.append(lstm_feat_4train, dem_4train, 1)
    print("--------用于训练xgb的数据准备结束（补充dem在最后一列）----------")

    """-----------开始构建xgb模型------------------"""
    X_train, X_test, y_train, y_test = train_test_split(lstm_feat_4train, label_4train, test_size=0.1, random_state=5,
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

    # for wdrvi, dem in zip(pre_loader, dem_loader):
    #     _, out_of_lstm = model(wdrvi)
    #     for j in range(out_of_lstm.shape[0]):
    #         lstm_feat = out_of_lstm[j, :, :].cpu().detach().numpy()
    #         dem_feat = dem[j, :].cpu().detach().numpy()
    #         lstm_feat = np.concatenate((lstm_feat, dem_feat[:, None]), axis=1)
    #         probability = clf.predict_proba(lstm_feat)  # 分类概率
    #         probability_Y.append(probability[:, 1])
    #
    # print("--------用于预测的wdrvi数据经transformer和LSTM结束----------")
    #
    # geotransform = demset.geotransform  # geotransform
    # projection = demset.projection  # projection
    # probability_Y = np.array(probability_Y)
    # probability_Y.shape = (demset.rows, demset.cols)
    # driver = gdal.GetDriverByName('GTiff')
    # dst_filename = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\竹子概率_transfomer_lstm_xgb.tif'
    # dst_ds = driver.Create(dst_filename, demset.cols, demset.rows, 1, gdal.GDT_Float64)
    # dst_ds.SetGeoTransform(list(geotransform))
    # srs = osr.SpatialReference()
    # # srs.SetWellKnownGeogCS('EPSG:32648')
    # srs.ImportFromWkt(projection)
    # dst_ds.SetProjection(srs.ExportToWkt())
    # dst_ds.GetRasterBand(1).WriteArray(probability_Y)
    # 预分配数组
    # 预分配数组

    probability_Y = np.zeros((demset.rows, demset.cols))

    # 提取特征并进行预测
    lstm_features = []
    dem_features = []

    # 收集所有批次的特征
    for wdrvi, dem in zip(pre_loader, dem_loader):
        _, out_of_lstm = model(wdrvi)
        out_of_lstm = out_of_lstm.cpu().detach().numpy()
        dem = dem.cpu().detach().numpy()
        lstm_features.append(out_of_lstm)
        dem_features.append(dem)

    # 合并所有批次的特征
    lstm_features = np.concatenate(lstm_features, axis=0)
    dem_features = np.concatenate(dem_features, axis=0)

    # 将DEM特征添加到LSTM特征的最后一列
    combined_features = np.concatenate((lstm_features, dem_features[:, None]), axis=2)

    # 预测概率
    probabilities = clf.predict_proba(combined_features.reshape(-1, combined_features.shape[2]))[:, 1]

    # 将预测结果重塑回原始影像的形状
    probability_Y = probabilities.reshape(demset.rows, demset.cols)

    print("--------用于预测的wdrvi数据经transformer和LSTM结束----------")

    # 保存预测结果
    geotransform = demset.geotransform
    projection = demset.projection
    driver = gdal.GetDriverByName('GTiff')
    dst_filename = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\竹子概率_transfomer_lstm_xgb0903.tif'
    dst_ds = driver.Create(dst_filename, demset.cols, demset.rows, 1, gdal.GDT_Float64)
    dst_ds.SetGeoTransform(list(geotransform))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(probability_Y)

    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)
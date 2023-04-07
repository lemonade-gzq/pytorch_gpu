import torch
from osgeo import gdal
from osgeo import osr
import numpy as np
import os
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from osgeo import gdal
from osgeo import osr
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from matplotlib import pyplot

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import seaborn as sns
import matplotlib
import sklearn.model_selection as ms
import shap
import matplotlib.pyplot as plt

BATCH_SIZE = 5
EPOCHS = 1000
bidirectional_set = True
bidirectional = 2 if bidirectional_set else 1
input_dim = 24
hidden_dim1 = 15
hidden_dim2 = 5
layer_dim = 1
output_dim = 2
sequence_dim = 1
learning_rate = 0.001
loss_list = []
accracy_list = []
iteration_list = []


class LSTM_Model(nn.Module):
    def __init__(self, my_input_dim, my_hidden_dim1, my_hidden_dim2, my_layer_dim, my_output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim1 = my_hidden_dim1
        self.hidden_dim2 = my_hidden_dim2
        self.layer_dim = my_layer_dim
        # 构建lstm模型
        self.lstm = nn.LSTM(my_input_dim, my_hidden_dim1, my_layer_dim, batch_first=True,
                            bidirectional=bidirectional_set)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.lstm2 = nn.LSTM(my_hidden_dim1 * bidirectional, my_hidden_dim2, my_layer_dim, batch_first=True,
                             bidirectional=bidirectional_set)
        # 全连接层
        self.fc = nn.Linear(my_hidden_dim2 * bidirectional, my_output_dim)

    def forward(self, x):
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
        self.wdrvi_all = np.zeros((658 * 776, input_dim))
        m = 0
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
                m += 1
        self.x = torch.from_numpy(self.wdrvi_all).to(torch.float32)
        del self.wdrvi_all
        # self.y = torch.from_numpy(label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


class CSVDataSet(Dataset):
    def __init__(self, filrpath):
        print(f'reading{filrpath}')

        df = pd.read_csv(
            filrpath, header=0, index_col=0,
            encoding='utf-8',
            dtype={'label': np.int32}
        )

        feat = df.iloc[:, 2:].values
        print(f'the shape of feature is {feat.shape}')
        label = df.iloc[:, 0].values

        self.x = torch.from_numpy(feat).to(torch.float32)
        self.y = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index]


if __name__ == '__main__':
    lstm_feat = np.zeros((1, 20))  # 第二层的输出大小*是否双向lstm
    lstm_feat_4train = np.zeros((1, 20))

    tifset = TIFDataSet(r"E:\城市与区域生态\大熊猫和竹\种群动态模拟\岷山竹子分布\岷山矩形四调wdrvi")
    pre_loader = DataLoader(tifset, BATCH_SIZE, shuffle=False)


    csvset = CSVDataSet(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\岷山竹子分布\岷山四调样本点.csv')
    train_loader = DataLoader(csvset, BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    model = torch.load(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\岷山竹子分布\四调model_b5_epo1k_h15_24_h5_lr1e-3.pt')
    model.to(device)
    model.eval()

    for wdrvi in pre_loader:
        wdrvi = wdrvi.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
        _, out_of_lstm = model(wdrvi)
        for j in range(out_of_lstm.shape[0]):
            lstm_feat = np.append(lstm_feat, out_of_lstm[j, :, :].cpu().detach().numpy(), 0)
    lstm_feat = np.delete(lstm_feat, 0, 0)  # 删除了lstm_feat数字数组的第一行，因为都是append所以第一行都是0
    print("--------用于预测的wdrvi数据经LSTM降维结束----------")

    for wdrvi in train_loader:
        wdrvi = wdrvi.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
        _, out_of_lstm = model(wdrvi)
        for j in range(out_of_lstm.shape[0]):
            lstm_feat_4train = np.append(lstm_feat_4train, out_of_lstm[j, :, :].cpu().detach().numpy(), 0)
    lstm_feat_4train = np.delete(lstm_feat_4train, 0, 0)
    print("--------用于训练xgb的wdrvi数据经LSTM降维结束----------")

    rds = gdal.Open(r"E:\城市与区域生态\大熊猫和竹\种群动态模拟\岷山竹子分布\MS_dem.tif")
    cols = rds.RasterXSize
    rows = rds.RasterYSize
    band = rds.GetRasterBand(1)
    dem = band.ReadAsArray(0, 0, cols, rows)
    dem = dem.reshape(-1, 1)
    lstm_feat = np.append(lstm_feat, dem, 1)
    print("--------用于预测的数据准备结束（补充dem在最后一列）----------")

    df_4train = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\岷山竹子分布\岷山四调样本点.csv', header=0, index_col=0,
                            encoding='utf-8')
    dem_4train = df_4train.iloc[:, 1].values
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
    f1_score = ms.cross_val_score(clf, lstm_feat_4train, label_4train, cv=5, scoring="f1")  # f1得分
    print(f"F1分数为：{np.mean(f1_score)}")
    f1_score = ms.cross_val_score(clf, lstm_feat_4train, label_4train, cv=5, scoring="r2")  # f1得分
    print(f"r2分数为：{np.mean(f1_score)}")


    # 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print('The confusion matrix result:\n', confusion_matrix_result)
    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()



    probability_Y = clf.predict_proba(lstm_feat)  # 分类概率
    print(probability_Y.shape)
    data_probability = probability_Y[:, 1]
    print(data_probability)

    geotransform = rds.GetGeoTransform()  # geotransform
    projection = rds.GetProjectionRef()  # projection
    data_probability.shape = (rows, cols)
    driver = gdal.GetDriverByName('GTiff')
    dst_filename = r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\岷山竹子分布\岷山四调竹子概率_lstm_xgb.tif'
    dst_ds = driver.Create(dst_filename, cols, rows, 1, gdal.GDT_Float64)
    dst_ds.SetGeoTransform(list(geotransform))
    srs = osr.SpatialReference()
    # srs.SetWellKnownGeogCS('EPSG:32648')
    srs.ImportFromWkt(projection)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(data_probability)

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
from xgboost.sklearn import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import math
import shap
from scipy.stats import gaussian_kde
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


class TIFDataSet(Dataset):
    def __init__(self, filepath_wdrvi):

        self.WdrviNameList = os.listdir(filepath_wdrvi)
        self.wdrvi_all = np.zeros((6203 * 6273, 24))

        m = 0
        for i in range(len(self.WdrviNameList)):
            if (os.path.splitext(self.WdrviNameList[i])[-1] == ".tif"):
                rds = gdal.Open(filepath_wdrvi + "\\" + self.WdrviNameList[i])
                print(f'reading{filepath_wdrvi}' + "\\" + self.WdrviNameList[i])
                cols = rds.RasterXSize
                rows = rds.RasterYSize
                band = rds.GetRasterBand(1)
                data = band.ReadAsArray(0, 0, cols, rows)
                nodata = data[0, 0]
                np.nan_to_num(data, nan=-9, copy=False)
                data = np.where(data == nodata, -9, data)
                data = data.reshape(-1, 1)
                self.wdrvi_all[:, m] = data[:, 0]
                m += 1
        self.wdrvi = torch.from_numpy(self.wdrvi_all).float().to('cpu')
        del self.wdrvi_all

    def __len__(self):
        return self.wdrvi.shape[0]

    def __getitem__(self, index):
        return self.wdrvi[index]




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
        self.x = torch.from_numpy(self.data).to('cpu')
        del self.data
        # self.y = torch.from_numpy(label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


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




if __name__ == '__main__':
    starttime = datetime.datetime.now()

    lstm_feat_4train = np.zeros((1, 32))# 第三层的输出大小
    tifset = TIFDataSet(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\植被指数序列\new_wdrvi")
    pre_loader = DataLoader(tifset, batch_size=20, shuffle=False)

    demset = ADDDataSet(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_chm_2502re.tif")
    dem_loader = DataLoader(demset, batch_size=20, shuffle=False)

    wsciset = ADDDataSet(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\vsc\wsci_re.tif")
    wsci_loader = DataLoader(wsciset, batch_size=20, shuffle=False)

    csvset = TimeSeriesDataset(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI_sample_merge方案5.csv')  #WDRVI_sample_merge方案3new_wdrvi
    train_loader = DataLoader(csvset, batch_size=20, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(
        r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\model_wdrvi_input24_seq1_lstm16_batch200_merge方案5.pt')
    model.to(device)
    model.eval()

    for wdrvi,_ in train_loader:
        _, out_of_lstm = model(wdrvi)
        lstm_feat_4train = np.append(lstm_feat_4train, out_of_lstm.view(out_of_lstm.size(0), -1).cpu().detach().numpy(),0)
    lstm_feat_4train = np.delete(lstm_feat_4train, 0, 0)
    print("--------用于训练xgb的wdrvi数据经transformer和LSTM结束----------")
    # pd.DataFrame(lstm_feat_4train).to_csv('E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\lstm_4_train_inputsize21_more.csv')

    df_4train = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI_sample_merge方案5.csv', header=0, index_col=0,
                            encoding='utf-8')
    dem_4train = df_4train.iloc[:, -2:].values.reshape(-1, 2)
    # wsci_4train = df_4train.iloc[:, -1].values.reshape(-1, 1)
    label_4train = df_4train.iloc[:, 0].values
    lstm_feat_4train = np.append(lstm_feat_4train, dem_4train, 1)
    print("--------用于训练xgb的数据准备结束（补充dem在最后一列）----------")
    cols_feature = ['feature{}'.format(i) for i in range(32)]
    cols_feature = cols_feature + ['chm', 'wsci']
    # cols_feature2 = ['SAR{}'.format(i) for i in range(38)]
    # cols_feature = cols_feature + cols_feature2
    lstm_feat_4train = pd.DataFrame(lstm_feat_4train, columns=cols_feature)

    """-----------开始构建xgb模型------------------"""
    X_train, X_test, y_train, y_test = train_test_split(lstm_feat_4train, label_4train, test_size=0.3, random_state=5,
                                                        shuffle=True)
    clf = XGBClassifier(objective="binary:logistic", seed=1024, learning_rate=0.1,
                        max_depth=6, min_child_weight=6,gamma=0.14,colsample_bytree=0.97,subsample=0.67,)

    # 方案3  max_depth=2,min_child_weight=0.97,gamma=0.38,subsample=0.16,colsample_bytree=0.89,alpha=0.0,reg_lambda=0.11,n_estimators=106
    # new: max_depth=1,min_child_weight=3,gamma=0.88,subsample=0.12,colsample_bytree=0.21,alpha=0.0,reg_lambda=0.75,n_estimators=200
    # 方案4::::max_depth=1,min_child_weight=2,gamma=0,subsample=0.97,colsample_bytree=0.71,alpha=0.19,reg_lambda=0.27,n_estimators=51
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



    explainer = shap.TreeExplainer(clf, X_train, feature_perturbation="interventional", model_output='probability')
    shap_values = explainer.shap_values(lstm_feat_4train[cols_feature])
    shap.summary_plot(shap_values, lstm_feat_4train[cols_feature], max_display=12)
    shap.summary_plot(shap_values, lstm_feat_4train[cols_feature], plot_type="bar")
    shap.dependence_plot('chm', shap_values, lstm_feat_4train, interaction_index=None, show=True)
    shap.dependence_plot('wsci', shap_values, lstm_feat_4train, interaction_index=None, show=True)



    """-----------xgb模型训练结束------------------"""

    probability_Y = np.zeros((demset.rows, demset.cols))

    # 提取特征并进行预测
    lstm_features = []
    dem_features = []
    wsci_features = []

    # 收集所有批次的特征
    for wdrvi, dem, wsci in zip(pre_loader, dem_loader, wsci_loader):
        _, out_of_lstm = model(wdrvi.to(device))
        out_of_lstm = out_of_lstm.cpu().detach().numpy()
        dem = dem.cpu().detach().numpy()
        lstm_features.append(out_of_lstm)
        dem_features.append(dem)
        wsci_features.append(wsci)

    # 合并所有批次的特征
    lstm_features = np.concatenate(lstm_features, axis=0)
    dem_features = np.concatenate(dem_features, axis=0)
    wsci_features = np.concatenate(wsci_features, axis=0)


    # 将DEM特征添加到LSTM特征的最后一列
    combined_features = np.concatenate((lstm_features, dem_features[:, None], wsci_features[:, None]), axis=2)

    # 预测概率
    probabilities = clf.predict_proba(combined_features.reshape(-1, combined_features.shape[2]))[:, 1]

    # 将预测结果重塑回原始影像的形状
    probability_Y = probabilities.reshape(demset.rows, demset.cols)

    print("--------用于预测的wdrvi数据经transformer和LSTM结束----------")

    # 保存预测结果
    geotransform = demset.geotransform
    projection = demset.projection
    driver = gdal.GetDriverByName('GTiff')
    dst_filename = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\bamboo_inputsize24_seq1_lstm4_batch200_more_merge方案5_xgb默认参数.tif'
    dst_ds = driver.Create(dst_filename, demset.cols, demset.rows, 1, gdal.GDT_Float64)
    dst_ds.SetGeoTransform(list(geotransform))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(probability_Y)

    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)

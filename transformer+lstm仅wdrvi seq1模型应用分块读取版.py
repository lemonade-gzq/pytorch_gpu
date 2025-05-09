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

class TIFDataSet2(torch.utils.data.Dataset):
    def __init__(self, file_dir, block_size=400):
        self.file_dir = file_dir
        self.block_size = block_size
        self.tif_files = [f for f in os.listdir(file_dir) if f.lower().endswith(('.tif', '.tiff'))]
        # self.tif_files.sort()
        self.datasets = []
        for filename in self.tif_files:
            filepath = os.path.join(self.file_dir, filename)
            ds = gdal.Open(filepath)
            self.datasets.append(ds)

        # 获取基本信息
        self.width = self.datasets[0].RasterXSize
        self.height = self.datasets[0].RasterYSize
        self.num_bands = len(self.datasets)

        # 计算分块索引
        self.blocks = []
        for y in range(0, self.height, self.block_size):
            for x in range(0, self.width, self.block_size):
                bw = min(self.block_size, self.width - x)
                bh = min(self.block_size, self.height - y)
                self.blocks.append((x, y, bw, bh))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        x, y, bw, bh = self.blocks[idx]
        block_data = []
        nodata_masks = []

        for ds in self.datasets:
            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            data = band.ReadAsArray(x, y, bw, bh).astype(np.float32)

            if nodata is not None:
                mask = data != nodata
                nodata_masks.append(mask)
            data = np.nan_to_num(data, nan=-9)
            if nodata is not None:
                data[data == nodata] = -9

            block_data.append(data.reshape(-1, 1))

        combined = np.concatenate(block_data, axis=1)
        tensor = torch.from_numpy(combined).float()
        if nodata_masks:
            valid_mask = np.logical_or.reduce(nodata_masks).reshape(-1)
        else:
            valid_mask = np.ones((bw * bh,), dtype=bool)

        return {
            'tensor': tensor,
            'x': x,
            'y': y,
            'width': bw,
            'height': bh,
            'valid_mask': valid_mask
        }

    def __del__(self):
        # 确保 Dataset 被销毁时关闭 gdal 资源
        for ds in self.datasets:
            ds = None

def no_stack_collate(batch):
    return batch

class ADDDataSet2(Dataset):
    def __init__(self, filepath, block_size=400):
        self.filepath = filepath
        self.block_size = block_size
        ds = gdal.Open(filepath)
        self.geotransform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()
        self.cols = ds.RasterXSize
        self.rows = ds.RasterYSize
        self.band = 1  # 默认取第1波段
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        self.nodata_value = nodata if nodata is not None else -9999
        ds = None

        # 计算分块
        self.blocks = []
        for y in range(0, self.rows, self.block_size):
            for x in range(0, self.cols, self.block_size):
                bw = min(self.block_size, self.cols - x)
                bh = min(self.block_size, self.rows - y)
                self.blocks.append((x, y, bw, bh))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        x, y, bw, bh = self.blocks[idx]
        ds = gdal.Open(self.filepath)
        band = ds.GetRasterBand(self.band)
        data = band.ReadAsArray(x, y, bw, bh).astype(np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = np.ones((bh, bw), dtype=bool)
        data = np.nan_to_num(data, nan=self.nodata_value)
        if nodata is not None:
            data[data == nodata] = self.nodata_value
        tensor = torch.from_numpy(data.reshape(-1, 1)).float()
        ds = None
        return {
            'tensor': tensor,
            'x': x,
            'y': y,
            'width': bw,
            'height': bh,
            'valid_mask': valid_mask.reshape(-1)
        }

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

    lstm_feat_4train = np.zeros((1, 32))  # 第三层的输出大小
    tifset = TIFDataSet2(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\植被指数序列\new_wdrvi")
    pre_loader = DataLoader(tifset, batch_size=20, shuffle=False, collate_fn=no_stack_collate)
    #DataLoader的默认：default_collate，会把 batch 里的每个元素拼接成一个大的tensor，这在分块读取栅格的时候会造成报错：每个块的大小不一致

    demset = ADDDataSet2(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_chm_2502re.tif")
    dem_loader = DataLoader(demset, batch_size=20, shuffle=False, collate_fn=no_stack_collate)

    wsciset = ADDDataSet2(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\vsc\wsci_re.tif")
    wsci_loader = DataLoader(wsciset, batch_size=20, shuffle=False, collate_fn=no_stack_collate)

    csvset = TimeSeriesDataset(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI_sample_merge方案5.csv')
    train_loader = DataLoader(csvset, batch_size=20, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_result = np.zeros((tifset.height, tifset.width), dtype=np.float32)
    count_map = np.zeros((tifset.height, tifset.width), dtype=np.uint8)

    model = torch.load(
        r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\model_wdrvi_input24_seq1_lstm16_batch200_merge方案5.pt')
    model.to(device)
    model.eval()

    for wdrvi, _ in train_loader:
        _, out_of_lstm = model(wdrvi.to(device))
        lstm_feat_4train = np.append(lstm_feat_4train, out_of_lstm.view(out_of_lstm.size(0), -1).cpu().detach().numpy(), 0)
    lstm_feat_4train = np.delete(lstm_feat_4train, 0, 0)
    print("--------用于训练xgb的wdrvi数据经transformer和LSTM结束----------")

    df_4train = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\WDRVI_sample_merge方案5.csv', header=0, index_col=0,
                            encoding='utf-8')
    dem_4train = df_4train.iloc[:, -2:].values.reshape(-1, 2)
    label_4train = df_4train.iloc[:, 0].values
    lstm_feat_4train = np.append(lstm_feat_4train, dem_4train, 1)
    print("--------用于训练xgb的数据准备结束（补充dem在最后一列）----------")
    cols_feature = ['feature{}'.format(i) for i in range(32)] + ['chm', 'wsci']
    lstm_feat_4train = pd.DataFrame(lstm_feat_4train, columns=cols_feature)

    """-----------开始构建xgb模型------------------"""
    X_train, X_test, y_train, y_test = train_test_split(lstm_feat_4train, label_4train, test_size=0.3, random_state=5, shuffle=True)
    clf = XGBClassifier(objective="binary:logistic", seed=1024, learning_rate=0.1,
                        max_depth=6, min_child_weight=6, gamma=0.14, colsample_bytree=0.97, subsample=0.67)
    clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)

    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
    print("AUC 得分 (测试集): %f" % metrics.roc_auc_score(y_test, test_predict))
    print("MSE:", mean_squared_error(y_test, test_predict))
    print("MAE:", mean_absolute_error(y_test, test_predict))
    print("R2:", r2_score(y_test, test_predict))
    print("acc:", accuracy_score(y_test, test_predict))
    print("F1:", f1_score(y_test, test_predict))

    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print('The confusion matrix result:\n', confusion_matrix_result)
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

    probability_Y = np.zeros((demset.rows, demset.cols), dtype=np.float32)
    # 计算总块数
    total_blocks = sum(len(w_batches) for w_batches in pre_loader)

    # 初始化进度条
    with tqdm(total=total_blocks, desc='Processing blocks') as pbar:
        for w_batches, d_batches, ws_batches in zip(pre_loader, dem_loader, wsci_loader):
            for w_batch, d_batch, ws_batch in zip(w_batches, d_batches, ws_batches):
                wdrvi = w_batch['tensor'].to(device)
                dem = d_batch['tensor'].cpu().numpy()
                wsci = ws_batch['tensor'].cpu().numpy()
                x = w_batch['x']
                y = w_batch['y']
                width = w_batch['width']
                height = w_batch['height']
                valid_mask = w_batch['valid_mask']

                with torch.no_grad():
                    _, out_of_lstm = model(wdrvi)
                out_of_lstm = out_of_lstm.cpu().numpy()
                out_of_lstm = out_of_lstm.reshape(out_of_lstm.shape[0], -1)

                dem = dem.reshape(dem.shape[0], -1)
                wsci = wsci.reshape(wsci.shape[0], -1)

                combined_features = np.concatenate((out_of_lstm, dem, wsci), axis=1)
                probabilities = clf.predict_proba(combined_features)[:, 1]

                # 填入结果图
                prob_block = np.full((height, width), np.nan, dtype=np.float32)
                prob_block.flat[valid_mask] = probabilities

                probability_Y[y:y + height, x:x + width] = prob_block

                # 显式释放内存
                del wdrvi, dem, wsci, out_of_lstm, combined_features, probabilities, prob_block
                torch.cuda.empty_cache()
                # 更新进度条
                pbar.update(1)

    # 保存预测结果
    geotransform = demset.geotransform
    projection = demset.projection
    driver = gdal.GetDriverByName('GTiff')
    dst_filename = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\bamboo_inputsize24_seq1_lstm4_batch200_more_merge方案5_xgb默认参数.tif'
    dst_ds = driver.Create(dst_filename, demset.cols, demset.rows, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(list(geotransform))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(probability_Y)

    endtime = datetime.datetime.now()
    print(f"总用时: {(endtime - starttime).seconds} 秒")

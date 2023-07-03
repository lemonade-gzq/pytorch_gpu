import torch
import torch.nn as nn
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
input_dim = 23
hidden_dim1 = 15
hidden_dim2 = 5
layer_dim = 1
output_dim = 2
sequence_dim = 1
learning_rate = 0.001
loss_list = []
accracy_list = []
iteration_list = []

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



class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return x
# 设置 Transformer 参数
input_dim = 1
d_model = 128
nhead = 4
num_layers = 3
dim_feedforward = 256

# 创建 Transformer 编码器实例
transformer_encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers, dim_feedforward)

# 准备输入数据
data = torch.randn(500, 60, 1)
data = data.permute(1, 0, 2)
# 提取时间序列特征
features = transformer_encoder(data)
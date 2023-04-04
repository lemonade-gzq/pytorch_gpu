## 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import torch
from osgeo import gdal
from osgeo import osr
import numpy as np
import os
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index]


lstm_feat_4train = np.zeros((1, 10))
lstm_feat_train = np.zeros((1, 10))
csvset = CSVDataSet(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\竹分布\三调bamboo_4xgb.csv ')
train_loader = DataLoader(csvset, BATCH_SIZE, shuffle=False)
model = torch.load(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\竹分布\三调model_b5_epo1k_h1_23_h25_lr1e-3.pt')
model.to(device)
model.eval()

for wdrvi in train_loader:
    wdrvi = wdrvi.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
    _, out_of_lstm = model(wdrvi)
    for j in range(BATCH_SIZE):
        lstm_feat_4train = np.append(lstm_feat_4train, out_of_lstm[j, :, :].cpu().detach().numpy(), 0)
lstm_feat_4train = np.delete(lstm_feat_4train, 0, 0)
print("--------用于训练xgb的wdrvi数据经LSTM降维结束----------")
df_4train = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\竹分布\三调bamboo_4xgb.csv ', header=0, index_col=0,
                        encoding='utf-8')
dem_4train = df_4train.iloc[:, 1].values
dem_4train = dem_4train.reshape(-1, 1)
label_4train = df_4train.iloc[:, 0].values
lstm_feat_4train = np.append(lstm_feat_4train, dem_4train, 1)
print("--------用于训练xgb的数据准备结束（补充dem在最后一列）----------")

X_train, X_test, y_train, y_test = train_test_split(lstm_feat_4train, label_4train, test_size=0.3, random_state=5)
param_test1 = {
    'max_depth': range(3, 10, 1),
    'min_child_weight': range(1, 6, 1)}
param_test2 = {
    "learning_rate": [round(num, 1) for num in np.arange(0, 0.3, 0.01)],
    "n_estimators": range(5, 30, 5)
}
param_test3 = {'gamma': [i / 10.0 for i in range(0, 20)]}
param_test4 = {
    'subsample': [i / 100.0 for i in range(80, 120, 5)],
    'colsample_bytree': [i / 100.0 for i in range(80, 120, 5)]
}
param_test6 = {
    'lambda': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    #  0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4, 0.5,1
    # 0.5,0.6,0.7,0.8,0.9, 1
}
param_test7 = {
    'n_estimators': range(15, 50, 1)
}
gsearch1 = GridSearchCV(estimator=XGBClassifier(seed=1024, learning_rate=0.1, max_depth=3, min_child_weight=3, gamma=0,
                                                subsample=1, colsample_bytree=1, alpha=0, reg_lambda=1),
                        param_grid=param_test7, scoring='roc_auc', cv=5, n_jobs=-1)
# max_depth=3, min_child_weight=1,
#                                                 n_estimators=47,
#                                                 gamma=0, subsample=1, colsample_bytree=0.7, alpha=0.05,
#                                                 reg_lambda=0.21
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

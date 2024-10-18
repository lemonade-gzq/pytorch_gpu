# -*- coding: utf-8 -*-


import os

import shap
from osgeo import gdal
from osgeo import osr
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# rc = {'font.sans-serif': ['Times New Roman']}
# sns.set( font_scale=1.5)
plt.figure(dpi=150)
filePath = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\遥感变量'
filename = os.listdir(filePath)
var = []
remote_all = np.zeros((6430 * 6379, 26))
for i in filename:
    if i.split(".")[-1] == "tif":
        rds = gdal.Open(filePath + '\\' + i)
        cols = rds.RasterXSize
        rows = rds.RasterYSize
        geotransform = rds.GetGeoTransform()  # geotransform
        projection = rds.GetProjectionRef()  # projection
        band = rds.GetRasterBand(1)
        print(cols, rows)
        print(rds.GetGeoTransform())
        var_name = i.split('.')[0]
        var.append(var_name)
        print(var_name)
        globals()[var_name] = band.ReadAsArray(0, 0, cols, rows)
        nodata = globals()[var_name][0, 0]
        np.nan_to_num(globals()[var_name], nan=-9999, copy=False)
        globals()[var_name] = np.where(globals()[var_name] == nodata, -9999, globals()[var_name])

"""模型构建"""
# print(data_all[:, 2])
data = pd.read_csv(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_CHM_all3.csv", engine='python', header=0)
data = data.iloc[:, 1:]
data.info()
X, Y = data[[x for x in data.columns if x != 'chm' and x != 'id']], data['chm']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5, shuffle=True)

clf = xgb.XGBRegressor(objective="reg:squarederror", seed=1024, learning_rate=0.1,max_depth=6,min_child_weight=0.9,
                       gamma=0.31,subsample=0.99,colsample_bytree=0.93,alpha=0.1,reg_lambda=0.9,n_estimators=436
                       )
"""GEDI: max_depth=3,min_child_weight=14,
                        gamma=0,subsample=1.0,colsample_bytree=1.0,alpha=0.56,reg_lambda=1.0,n_estimators=103"""
"""ICESat: max_depth=9,min_child_weight=5,
                        gamma=0.78,subsample=1,colsample_bytree=0.54,alpha=0.58,reg_lambda=1.0,n_estimators=203
                        """

clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)

"""精度指标计算"""
# print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
# print("AUC 得分 (测试集): %f" % metrics.roc_auc_score(y_test, test_predict))
print("MSE:", mean_squared_error(y_test, test_predict))
print("MAE:", mean_absolute_error(y_test, test_predict))
print("R2:", r2_score(y_test, test_predict))
print("绘制重要性图")
plot_importance(clf)
plt.show()
cols_feature = ['0114VH', '0114VV', 'aspect', 'dem', 'ECO', 'NDVI', 'S2REP', 'slope', 'VH_COH', 'VHCON', 'VHCOR',
                'VHDIS', 'VHENT', 'VHHOM', 'VHMEA', 'VHSEC', 'VHVAR', 'VV_COH', 'VVCON', 'VVCOR', 'VVDIS',
                'VVENT', 'VVHOM', 'VVMEA', 'VVSEC', 'VVVAR']
"""旧版'0114VH', 'VHCON', 'VHDIS', 'VHENT', 'VHHOM', 'VHMEA', 'VHSEC', 'VHVAR', '0114VV', 'VVCON', 'VVCOR',
                'VVDIS', 'VVENT', 'VVHOM', 'VVMEA', 'VVSEC', 'VVVAR', 'VHCOR', 'aspect',
                'dem', 'ECO', 'NDVI', 'S2REP', 'slope', 'VH_COH', 'VV_COH'"""
explainer = shap.TreeExplainer(clf, X_train, feature_perturbation="interventional", model_output='raw')
shap_values = explainer.shap_values(X_train[cols_feature])

shap.summary_plot(shap_values, X_train[cols_feature])
shap.summary_plot(shap_values, X_train[cols_feature], plot_type="bar")
# f1_score = ms.cross_val_score(clf, X, Y, cv=5, scoring="f1")  # f1得分
# print(f"5折交叉验证F1分数为：{np.mean(f1_score)}")
# r2_score = ms.cross_val_score(clf, X, Y, cv=5, scoring="r2")  # f1得分
# print(f"5折交叉验证r2分数为：{np.mean(r2_score)}")
print("测试集拟合图")
xy = np.vstack([y_test, test_predict])  # 将两个维度的数据叠加
z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
plt.scatter(y_test, test_predict, s=20, c=z, cmap='Spectral')

# plt.plot([min(y_test), max(y_test)], [min(test_predict), max(test_predict)])
plt.plot([0, 60], [0, 60])

plt.xlabel('Real')

plt.ylabel('Predict')
plt.show()

# 计算并输出预测结果
data_all = np.zeros((rows, cols, len(var)))
for i, var_name in enumerate(var):
    data_all[:, :, i] = globals()[var_name]

data_all = data_all.reshape(-1, len(var))
# mask = (data_all[:, 0] != -9999)  # 假设第一个变量的-9999表示无效值
# data_all = data_all[mask]
probability_Y = clf.predict(data_all)

# 重新构建影像
# data_probability = np.zeros((rows, cols))
data_probability = probability_Y.reshape(rows, cols)

driver = gdal.GetDriverByName('GTiff')
dst_ds = driver.Create(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_chm0921.tif", cols, rows, 1, gdal.GDT_Float64)
dst_ds.SetGeoTransform(list(geotransform))
srs = osr.SpatialReference()
srs.ImportFromWkt(projection)
dst_ds.SetProjection(srs.ExportToWkt())
dst_ds.GetRasterBand(1).WriteArray(data_probability)

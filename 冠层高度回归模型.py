# -*- coding: utf-8 -*-


import os
from tqdm import tqdm
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
import sklearn.model_selection as ms
BLOCK_SIZE = 512  # 分块大小

# rc = {'font.sans-serif': ['Times New Roman']}
# sns.set( font_scale=1.5)
plt.figure(dpi=150)

"""模型构建"""
# print(data_all[:, 2])
data = pd.read_csv(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\vsc\feature_wsci.csv", engine='python', header=0) #  icesat_CHM_all2502.csv   vsc\feature_wsci_shp
sorted_columns = sorted(data.columns)
data = data[sorted_columns]
data = data.iloc[:, 0:].reset_index(drop=True)
data.info()
# X, Y = data[[x for x in data.columns if x != 'CHM' and x != 'FID'and x != 'lon'and x != 'lat']], data['CHM']
X, Y = data[[x for x in data.columns if x != 'WSCI' and x != 'FID'and x != 'lon'and x != 'lat']], data['WSCI']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5, shuffle=True)

clf = xgb.XGBRegressor(objective="reg:squarederror", seed=1024, learning_rate=0.1,
                       max_depth=7,min_child_weight=0.0,gamma=0.37,subsample=0.92,colsample_bytree=0.95,alpha=0.74,reg_lambda=0.61,n_estimators=117)
"""wsci: max_depth=7,min_child_weight=0.0,gamma=0.37,subsample=0.92,colsample_bytree=0.95,alpha=0.74,reg_lambda=0.61,n_estimators=117"""
"""chm:   max_depth=9,min_child_weight=0.0,gamma=0.73,subsample=0.96,colsample_bytree=0.77,alpha=0.1,reg_lambda=0.96,n_estimators=139"""

clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)

"""精度指标计算"""
# print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
# print("AUC 得分 (测试集): %f" % metrics.roc_auc_score(y_test, test_predict))
print("MSE:", mean_squared_error(y_test, test_predict))
print("MAE:", mean_absolute_error(y_test, test_predict))
print("R2:", r2_score(y_test, test_predict))
print("测试集拟合图")
xy = np.vstack([y_test, test_predict])  # 将两个维度的数据叠加
z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
plt.scatter(y_test, test_predict, s=20, c=z, cmap='Spectral')
plt.plot([0, 60], [0, 60])
plt.xlabel('Real')
plt.ylabel('Predict')
# 设置 x 和 y 轴的显示范围
plt.xlim(7, 12)#7, 12
plt.ylim(7, 12)
plt.show()


print("绘制重要性图")
plot_importance(clf)
plt.show()
cols_feature = [x for x in sorted_columns if x != 'WSCI' and x != 'FID']
"""chm:'0114VH', '0114VV', 'aspect', 'dem', 'ECO', 'NDVI', 'S2REP', 'slope', 'VH_COH', 'VHCON', 'VHCOR',
                'VHDIS', 'VHENT', 'VHHOM', 'VHMEA', 'VHSEC', 'VHVAR', 'VV_COH', 'VVCON', 'VVCOR', 'VVDIS',
                'VVENT', 'VVHOM', 'VVMEA', 'VVSEC'"""
"""wsci:'aspect',	'bio_10',	'bio_13',	'bio_15',	'bio_17',	'bio_18',	'bio_2',	'bio_3',	'bio_4',
                'dem',	'DNI',	'eco',	'GHI',	'ndvi3re',	'PET',	'PH',	's2rep',	'slope',	'VV',	'vv_mean'"""
explainer = shap.TreeExplainer(clf, X_train, feature_perturbation="interventional", model_output='raw')
shap_values = explainer.shap_values(X_train[cols_feature])

shap.summary_plot(shap_values, X_train[cols_feature])
shap.summary_plot(shap_values, X_train[cols_feature], plot_type="bar")
# f1_score = ms.cross_val_score(clf, X, Y, cv=5, scoring="f1")  # f1得分
# print(f"5折交叉验证F1分数为：{np.mean(f1_score)}")
# r2_score = ms.cross_val_score(clf, X, Y, cv=5, scoring="r2")  # f1得分
# print(f"5折交叉验证r2分数为：{np.mean(r2_score)}")


rds = gdal.Open(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\遥感变量\DEM.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
geotransform = rds.GetGeoTransform()  # geotransform
projection = rds.GetProjectionRef()  # projection
driver = gdal.GetDriverByName('GTiff')
dst_ds = driver.Create(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\wsci2502.tif", cols, rows, 1, gdal.GDT_Float64)
dst_ds.SetGeoTransform(list(geotransform))
srs = osr.SpatialReference()
srs.ImportFromWkt(projection)
dst_ds.SetProjection(srs.ExportToWkt())


filePath = r'E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\vsc\feature'#  E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\遥感变量       \vsc\feature
filename = os.listdir(filePath)
var = []
# 计算总迭代次数
total_blocks = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE  # 向上取整计算总块数[4]

for y in tqdm(range(0, cols, BLOCK_SIZE),
             total=total_blocks,
             desc="外层循环",
             dynamic_ncols=True):  # 支持动态调整进度条宽度
    for x in range(0, rows, BLOCK_SIZE):
        block_width = min(BLOCK_SIZE, cols - x)
        block_height = min(BLOCK_SIZE, rows - y)
        for i in filename:
            if i.split(".")[-1] == "tif":
                rds = gdal.Open(filePath + '\\' + i)
                band = rds.GetRasterBand(1)
                var_name = i.split('.')[0]
                globals()[var_name] = band.ReadAsArray(x, y, block_width, block_height)
                if x==0 and y==0:
                    var.append(var_name)
                    print(var_name, ":", globals()[var_name].shape)
                    nodata = globals()[var_name][0, 0]
                np.nan_to_num(globals()[var_name], nan=-9999, copy=False)
                globals()[var_name] = np.where(globals()[var_name] == nodata, -9999, globals()[var_name])
                # 计算并输出预测结果
        data_all = np.zeros((block_height, block_width, len(var)))
        for i, var_name in enumerate(var):
            data_all[:, :, i] = globals()[var_name]
            del globals()[var_name]
        data_all = data_all.reshape(-1, len(var))
        probability_Y = clf.predict(data_all)
        data_probability = probability_Y.reshape(block_height, block_width)
        dst_ds.GetRasterBand(1).WriteArray(data_probability, x, y)
rds = None  # 关闭输入影像
dst_ds = None  # 关闭输出影像

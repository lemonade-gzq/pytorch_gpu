## 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
import pandas as pd
import xgboost as xgb
import numpy as np

data = pd.read_csv(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_CHM_all2502.csv", engine='python', header=0) #  icesat_CHM_all1126.csv   vsc\feature_wsci_shp
sorted_columns = sorted(data.columns)
data = data[sorted_columns]
data = data.iloc[:, 0:].reset_index(drop=True)
data.info()
X, Y = data[[x for x in data.columns if x != 'CHM' and x != 'FID'and x != 'lon'and x != 'lat']], data['CHM']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5, shuffle=True)

clf = xgb.XGBRegressor(objective="reg:squarederror", seed=1024, learning_rate=0.1,)

param_test1 = {
    'max_depth': range(1, 55, 1),
}
param_test2 = {
    "n_estimators": range(0, 250, 1)
}
param_test3 = {'gamma': [0, 1e-5, 1e-2, 0.1, 0.5, 1, 100]}
param_test4 = {
    'reg_lambda': [i / 100.0 for i in range(0, 100)],
}
param_test6 = {
    'min_child_weight': [i / 100.0 for i in range(0, 5, 1)],
    'gamma': [i / 100.0 for i in range(0, 5, 1)],
    'subsample': [i / 100.0 for i in range(90, 95, 1)],
    'colsample_bytree': [i / 100.0 for i in range(85, 90, 1)],
    'reg_lambda': [i / 100.0 for i in range(70, 75, 1)],

}
param_test7 = {
    'colsample_bytree': [1e-5, 1e-2, 0.1, 1, 100]
}
gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(objective="reg:squarederror", seed=1024, learning_rate=0.1,
                        max_depth=9,min_child_weight=0.0,gamma=0.73,subsample=0.96,colsample_bytree=0.77,alpha=0.1,reg_lambda=0.96,n_estimators=139),
                        param_grid=param_test2, scoring='r2', cv=5, n_jobs=-1)
# max_depth=2,min_child_weight=0.0,gamma=0.01,subsample=0.94,colsample_bytree=0.86,alpha=0.0,reg_lambda=0.72,n_estimators=100
#  0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4, 0.5,1
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

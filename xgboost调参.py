## 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
import pandas as pd
import xgboost as xgb
import numpy as np

data = pd.read_csv(r"E:\城市与区域生态\大熊猫和竹\竹子分布模拟\冠层高度模型\icesat_CHM_all3.csv", engine='python', header=0)
data = data.iloc[:, 1:]
data.info()
X, Y = data[[x for x in data.columns if x != 'chm' and x != 'id']], data['chm']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5, shuffle=True)

clf = xgb.XGBRegressor(objective="reg:squarederror", seed=1024, learning_rate=0.1,)

param_test1 = {
    'max_depth': range(1, 25, 1),
}
param_test2 = {
    "n_estimators": range(398, 450, 1)
}
param_test3 = {'gamma': [0, 1e-5, 1e-2, 0.1, 0.5, 1, 100]}
param_test4 = {
    'colsample_bytree': [i / 100.0 for i in range(50, 90)],
}
param_test6 = {
    # 'subsample': [i / 100.0 for i in range(90, 101, 1)],
    # 'colsample_bytree': [i / 100.0 for i in range(90, 101, 1)],
    # 'alpha': [i / 100.0 for i in range(0, 11, 1)],
    # 'reg_lambda': [i / 100.0 for i in range(90, 101, 1)],
    'gamma': [i / 100.0 for i in range(20, 41, 1)],
    'min_child_weight': [i / 100.0 for i in range(90, 101, 1)],

}
param_test7 = {
    'colsample_bytree': [1e-5, 1e-2, 0.1, 1, 100]
}
gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(objective="reg:squarederror", seed=1024, learning_rate=0.1,
                      max_depth=6,min_child_weight=0.9,gamma=0.31,subsample=0.99,colsample_bytree=0.93,alpha=0.1,reg_lambda=0.9,n_estimators=436),
                        param_grid=param_test1, scoring='r2', cv=5, n_jobs=-1)
# max_depth=8,min_child_weight=1,gamma=0.3,subsample=0.93,colsample_bytree=1.0,alpha=0,reg_lambda=1.0,n_estimators=373
#  0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4, 0.5,1
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

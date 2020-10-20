# 課題1: データセット「mobile-price-classfication」は携帯電話の端末ごとの情報と価格ランク（price_range）が記録されている。
#
# このデータを元に、価格ランク（price_range）を予測するモデルを構築せよ。
#
# 【データ】
#
# - train.csv
#   - 学習用のデータ
#   - 携帯電話、スマートフォン各機種のスペックと価格ランク（price_range）が記録されたデータ。価格ランクは0~3の4段階ある。
# - test.csv
#   - 精度検証用のデータ
#   - train.csvと同じカラム構成。test.csvに格納されている「price_range」を正とした予測結果を算出すること。
#
# 【提出要件】
#
# 以下の項目をJupyter Notebook上で出力した状態で提出すること
#
# - 予測結果と実際のランクによる混合行列
# - 予測において最も重要と考えられる変数Top5を列挙

# 以下よりコードを記入してください  ##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pandas import plotting 
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from keras.layers import Dense,Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import accuracy_score 
from sklearn.metrics import log_loss     
from sklearn.metrics import roc_auc_score 

%matplotlib inline

# 使用するデータの読み込み
train = pd.read_csv('../mobile-price-classification/train.csv')
test = pd.read_csv('../mobile-price-classification/test.csv')
train.head()

test.head()

#欠損値と詳細の確認
train.info()
train.describe()
train.corr().sort_values("price_range")

#あとで使うので2つのprice_rangeは変数に入れておく
price_range = train["price_range"]
test_price_range = test["price_range"]

#price_rangeの分布を確認
plt.hist(price_range)

#列を掛け算して新しい列を作り、元のものを削除する関数を作成
def multiplication(g_col,col1,col2):
    train[g_col] = train[col1] * train[col2]
    test[g_col] = test[col1] * test[col2]
    train.drop([col1,col2], axis=1, inplace=True)
    test.drop([col1,col2],axis=1, inplace=True)
    

#trainとtestから同じカラムを削除する関数を作成
def delete_column(col):
    train.drop(col,axis=1,inplace=True)
    test.drop(col,axis=1,inplace=True)
    
    
delete_column("id")

multiplication("sc_s", "sc_w", "sc_h")

multiplication("px", "px_width", "px_height")

#battery_powerの相関係数が高かったので詳しく見てみる
battery_power = train["battery_power"]
battery_power.describe()
plt.hist(battery_power)

#trainを目的変数のprice_rangeとそれ以外に分ける
trainx = train.drop("price_range", axis=1)
trainx_columns = trainx.columns

#正規化してからモデルに学習させる
scaler = StandardScaler()

scaler.fit(trainx)

train_std = pd.DataFrame(scaler.transform(trainx), columns=trainx_columns)

train_std["price_range"] = price_range
train_std.head()

train_std.corr().sort_values("price_range")

#blueが0か1の変数だったので、正規化してから特徴量作成
train_std["function"] = train_std["wifi"] * train_std["blue"]
train_std.drop(["wifi", "blue"], axis=1, inplace=True)

#ここからはデータを分ける→モデルを作る→チューニングの流れ
x = train_std.drop("price_range",axis=1)
y = price_range

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,random_state=1)

svm=SVC(random_state=1)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

params = {"C":[1,0.1,0.25,0.5,2,0.75],
         "kernel":["linear","rbf"],
         "gamma":["auto",0.01,0.001,0.0001,1],
         "decision_function_shape":["ovo","ovr"],
         "tol":[0.1,0.01,0.001,0.0001]}

svm=SVC(random_state=1)
grid_svm=GridSearchCV(estimator=svm,cv=5,param_grid=params)
grid_svm.fit(x_train,y_train)
print("best score: ", grid_svm.best_score_)
print("best param: ", grid_svm.best_params_)

svm_model=SVC(C=0.1,decision_function_shape="ovo",gamma="auto",kernel="linear",random_state=1,tol=0.01)

svm_model.fit(x_train,y_train)

print("train_accuracy:",svm_model.score(x_train,y_train))
print("test_accuracy: ", svm_model.score(x_test,y_test))

#テストデータに訓練データと同じ処理を加える
test.drop("price_range",axis=1,inplace=True)
test_columns = test.columns

test_std = pd.DataFrame(scaler.transform(test), columns=test_columns)

test_std["function"] = test_std["blue"] * test_std["wifi"]
test_std.drop(["blue","wifi"], axis=1, inplace=True)

pred = svm_model.predict(test_std)
train_pred = svm_model.predict(x)

scores = cross_val_score(estimator=svm_model, X=x, y=y, cv=8 )
scores.mean()

#混合行列を作成
confusion_matrix(test_price_range, pred)

#lgbのモデルも作成　こちらは手作業で何度かチューニングした
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass', 
        'num_class': 4,
        'metric': {'multi_error'},
         "learning_rate":0.3,
         "num_leaves":26, 
        
}


lgb_model = lgb.train(params,
train_set=lgb_train, 
valid_sets=lgb_eval, 
)

y_pred_prob = lgb_model.predict(x_test)

y_pred = np.argmax(y_pred_prob, axis=1) 

acc = accuracy_score(y_test,y_pred)
print('Acc :', acc)

lgb_pred_prob = lgb_model.predict(test_std)

lgb_pred = np.argmax(lgb_pred_prob, axis=1) 

confusion_matrix(test_price_range, lgb_pred)

#特徴量の重要度をDataFrameにして上から5つを表示
f_importance = np.array(lgb_model.feature_importance()) 
df_importance = pd.DataFrame({'feature':x_train.columns, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) 
df_importance.iloc[0:5]
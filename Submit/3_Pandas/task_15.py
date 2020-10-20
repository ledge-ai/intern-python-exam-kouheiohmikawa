# 課題15： 自分で好きなデータセットを作成 or 取得してきてデータ分析してみよう。
# - 使用したデータ
# - データからどんなことが言えそうか？（考察）
# - 考察に対してどうやってアプローチしたか？（仮説などもあると良い）
# 
# 参考サイト[Kaggle Dataset](https://www.kaggle.com/datasets)などから取ってくると良いと思います。
# 
# データセットの探し方がわからない場合は適宜メンターに聞いてみてください。
# 
# 例えば、[タイタニック号沈没事故に関するデータ](https://www.kaggle.com/c/titanic/overview)の場合
# - 乗客の生存率に大きく影響のありそうなデータは何か？
# - どういう前処理をすると良さそうか？
# 
# などなど


#・使用データ
#housepriceを予測するデータセットを使用

train = pd.read_csv('../data/train.csv')
train.head()

train.shape

#とりあえず欠損値の確認と特徴を調べる
train.info()

train["PoolQC"].value_counts()

Pool_df = train[train["PoolQC"].isin(["Gd","Fa","Ex"])]
print("プールがある時")
Pool_df.groupby("PoolQC")["SalePrice"].mean()
print("プールがない時")
train["SalePrice"].mean()

Pool_df.groupby("PoolQC")["SalePrice"].mean()

train["MiscFeature"].value_counts()
train.groupby("MiscFeature")["SalePrice"].mean()

#MiscFeatureによって違いはありそうだが欠損値はよくわからないのでそのまま

#Fenceについて
train["Fence"].value_counts()
train.groupby("Fence")["SalePrice"].describe()
train["SalePrice"].describe()

#Fenceはそれほど価格に影響しなさそう

#ここからは目的変数への影響度が大きそうな特徴量を調べてみる
train.corr().sort_values("SalePrice")

#とりあえず0.6以上に注目する
df_2 = train[["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]
df_2.isnull().sum()
df_2

#GarageCarsとGarageAreaに着目
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
train["GarageCars"].hist()
plt.title("GarageCars")

plt.subplot(1,2,2)
train["GarageArea"].hist()
plt.title("GarageArea")

#この2つで新しい特徴量を作ってみる
garage = train["GarageArea"] * train["GarageCars"]
garage.corr(train["SalePrice"])

#GarageAreaとGarageCars単体よりも相関係数が高くなった

sns.pairplot(df_2)
#GrLivArea,TotalBsmtSF,lstFlrSFなどは明らかな外れ値があるので学習するときは取り除いてみる
#1stFlrSF,2ndFlrSF,GrLivArea,TotalBsmtSFは広さに関する変数なので合わせてみる

sf_list = ["1stFlrSF","2ndFlrSF","GrLivArea","TotalBsmtSF","SalePrice"]
train[sf_list].corr()

SF = train["1stFlrSF"] + train["2ndFlrSF"] + train["GrLivArea"] + train["TotalBsmtSF"]
SF.corr(train["SalePrice"])

#これも相関係数の高い特徴量を作れた

#次は相関係数が負だったものに着目する
mlist = ["KitchenAbvGr","EnclosedPorch","MSSubClass","OverallCond","SalePrice"]
sns.pairplot(train[mlist])

#これも学習に使うなら外れ値を消してみるが、そもそも絶対値が小さいので多分使わない
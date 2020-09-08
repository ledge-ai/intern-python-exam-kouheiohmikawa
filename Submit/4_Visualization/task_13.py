# 課題13: データから自分なりの仮説を立て、可視化することによりその仮説が正しいかどうか検証せよ
# 
# 仮説は自由に設定して構いません。自動車の販売価格に関係することでも良し、一般的と思われそうなことを確かめるでも良いです
# 
# 設定した仮説を提唱し、それを立証できるようなグラフを作成してみましょう。そのグラフを元に仮説検証してみましょう
# 
# 仮説の例として。。。
# 
# - 車種（CarName）と価格は関係ありそう
# 
# - 馬力（horsepower）とエンジンの大きさ（enginesize）は比例しそう
# 
# - 車両重量（curbweight）と車の体積（長さと幅と高さから算出できる）との関係はどうなのか


# 以下よりコードを記入してください  ##############################

#車の体積と価格の関係
df["carvolume"] = df["carlength"] * df["carwidth"] * df["carheight"]

x = df["carvolume"]
y = df["price"]

plt.scatter(x,y)

df.head()

sns.scatterplot(x,y,hue = df["fueltype"])

data1 = df["stroke"]
data2 = df["price"]

sns.jointplot(data1,data2)

sns.jointplot(data1,data2,kind="hex")

sns.rugplot(data2)
plt.hist(data2,alpha=0.3)

sns.kdeplot(data2)

sns.rugplot(data2,color="black")

plt.hist(data2,cumulative=True)

sns.kdeplot(data1, data2)

sns.kdeplot(data1, data2,shade=True)

x1 = df["carvolume"]
y1 = df["horsepower"]

sns.scatterplot(x1,y1,hue = df["fueltype"])
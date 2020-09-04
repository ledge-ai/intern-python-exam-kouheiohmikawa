# 課題6: エンジンサイズ（enginesize）をx軸、車の価格（Price）をy軸とした散布図を作成せよ
# 
# それぞれの軸ラベルに変数名を記載すること


# 以下よりコードを記入してください  ##############################
x = df["enginesize"]
y = df["price"]


sns.scatterplot(x,y)


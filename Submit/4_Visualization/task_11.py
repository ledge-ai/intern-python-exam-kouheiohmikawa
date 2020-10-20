# 課題11: 車種の駆動車輪（drivewheel）をx軸、駆動車輪ごとの車の価格（price）の平均値をy軸で表現した棒グラフを作成せよ


# 以下よりコードを記入してください  ##############################
df.groupby("drivewheel")["price"].mean().plot(kind="bar")


# 課題3: 車種ごとの燃料タイプ（fueltype）の割合を円グラフで表現せよ
# 
# ラベルや構成割合を図に記載すること


# 以下よりコードを記入してください  ##############################

fueltype = df["fueltype"].value_counts().values
labels = ["gas","diesel"]

plt.pie(fueltype, labels = labels, autopct="%1.1f%%",startangle = 90)

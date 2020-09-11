# 課題7: 車体の長さ（carlength）をx軸に、車体の幅（carwidth）をy軸にとり、車の車体（carbody）で色分けした散布図を作成せよ
# 
# それぞれの軸ラベルに変数名を記載し、グラフ内に色の凡例を表示すること


# 以下よりコードを記入してください  ##############################

sns.scatterplot(x, y, hue=df["carbody"])



# 解き直し ##############################
# 以下よりコードを記入してください  ##############################
x = df["carlength"]
y = df["carwidth"]
sns.scatterplot(x, y, hue=df["carbody"])
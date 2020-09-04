# 課題4: 自動車の車体（carbody）の数を集計し、横棒グラフで表現せよ
# 
# y軸のラベルは数が多い順に並べ替えること


# 以下よりコードを記入してください  ##############################

df["carbody"].value_counts(ascending=True).plot(kind="barh")

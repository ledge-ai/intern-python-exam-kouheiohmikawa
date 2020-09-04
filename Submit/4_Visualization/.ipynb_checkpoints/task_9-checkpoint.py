# 課題9: 上で得られた相関行列を参考に、車の価格（Price）に対する相関が高い変数を5つ選べ
# 
# その変数と車の価格を合わせた計6変数において散布図行列を作成せよ


# 以下よりコードを記入してください  ##############################
sort_corr = corr.sort_values("price",ascending=False)
list_9 = sort_corr[0:6].index.to_list()
df_9 = df[list_9]
df_9.corr()


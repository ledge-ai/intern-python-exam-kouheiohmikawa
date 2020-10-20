# 課題10: 上で得られた「C」に対して、全ての要素における「最大値」, 「最小値」, 「平均値」, 「中央値」, 「第1四分位数」, 「第3四分位数」を求めよ
# 
# それぞれ「C_max」, 「C_min」, 「C_mean」, 「C_median」, 「C_quartile_1」, 「C_quartile_3」という変数に格納すること


# 以下よりコードを記入してください  ##############################
C_max = np.max(C)
C_min = np.min(C)
C_mean = np.mean(C)
C_median = np.median(C)
C_quartile_1, C_quartile_3 = np.percentile(C,q = [25,75])

# 出力  #################################################
print(C_max)
print(C_min)
print(C_mean)
print(C_median)
print(C_quartile_1)
print(C_quartile_3)

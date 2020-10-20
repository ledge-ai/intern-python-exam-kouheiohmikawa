# 課題8: データから数値型の変数（car_IDを除く）の相関行列を作成せよ
# 
# 相関行列はヒートマップで作成し、図中に相関係数を表示すること。ヒートマップの色は問わない


# 以下よりコードを記入してください  ##############################

df_8 = df.select_dtypes(["int64","float64"])
df_8 = df_8.drop("car_ID",axis=1)
corr = df_8.corr()

plt.figure(figsize=(9,9))
sns.heatmap(corr,annot=True)

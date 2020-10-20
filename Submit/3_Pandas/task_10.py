# 課題10： numeric_dfを行方向に7対3に分割し、DataFrameの行数、列数がわかるように出力せよ（変数名はdf_10_1, df_10_2）
# 
# ※ただしdf_10_1の方がdf_10_2よりも行数が多くなるように設定せよ


numeric_df = pd.DataFrame(np.random.randint(1,100, (20,7)))
# 以下よりコードを記入してください  ##############################
df_10_1 = numeric_df.iloc[0:14]
df_10_2 = numeric_df.iloc[14:20]

# 出力  #################################################
print(df_10_1.shape)
print(df_10_2.shape)






# 解き直し  ##############################
numeric_df = pd.DataFrame(np.random.randint(1,100, (20,7)))
# 以下よりコードを記入してください  ##############################
l = len(numeric_df)
t = int(l * 0.7)

df_10_1 = numeric_df.iloc[0:t]
df_10_2 = numeric_df.iloc[t:l]

# 出力  #################################################
print(df_10_1.shape)
print(df_10_2.shape)


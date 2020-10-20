# 課題7： 
# - nan_dfの欠損値のある行をすべて取り除き出力せよ（変数名はdf_7_1）
# - nan_dfの欠損値を各カラムの平均値で埋めて出力せよ（変数名はdf_7_2）


nan_df = pd.DataFrame(np.random.rand(10,4))
nan_df.iloc[1, 0] = np.nan
nan_df.iloc[2:4, 3] = np.nan
nan_df.iloc[5:, 2] = np.nan
# 以下よりコードを記入してください  ##############################
df_7_1 = nan_df.dropna()
df_7_2 = nan_df.fillna(nan_df.mean())


# 出力  #################################################
display(df_7_1)
display(df_7_2)

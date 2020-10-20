# 課題9： sample_df_2のidをfloatに変換し、birth_yearはintに変換しsample_df_2から数値タイプのカラムのみを取り出せ（変数名はdf_9）
# 
# ※カラム名を直接指定するのはNG


# 以下よりコードを記入してください  ##############################

sample_df_2 = pd.DataFrame(sample_dict_2)
sample_df_2["id"] = sample_df_2["id"].astype("float")
sample_df_2["birth_year"] = sample_df_2["birth_year"].astype("int")
df_9 = sample_df_2.select_dtypes(["int64","float64"])
# 出力  #################################################
display(df_9)

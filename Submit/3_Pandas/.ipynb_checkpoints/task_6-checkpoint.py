# 課題6： sample_df_1のデータに対してCityごとにデータ数をカウントせよ（変数名はdf_6_1）
# 
# また、cityごとのbirth_yearの平均値を求めよ（変数名はdf_6_2）
# 
# ※出力はDataFrame指定


# 以下よりコードを記入してください  ##############################
df_6_1 = df_1["City"].value_counts() 
df_6_2 = df_1.groupby("City")["Birth_year"].mean()

# 出力  #################################################
display(df_6_1)
display(df_6_2)

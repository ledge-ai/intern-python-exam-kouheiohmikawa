# 課題3： df_1からBirth_yearとIDのカラムの値を取り出しDataFrameとして出力せよ（変数名はdf_3）
# 
# またBirth_yearのカラムのみを取り出した場合の変数のタイプを出力せよ（変数名は by_type)


# 以下よりコードを記入してください  ##############################

df_3 = df_1[["Birth_year","ID"]]
by_type = type(df_1["Birth_year"])
# 出力  #################################################
display(df_3)
print(by_type)

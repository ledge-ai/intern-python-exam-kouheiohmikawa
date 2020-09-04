# 課題5： sample_dict_2とsample_dict_3をDataFrameに変換し、
# - idをキーに内部結合（変数名はdf_5_1）
# - 全結合（変数名はdf_5_2)
# - 左外部結合（変数名はdf_5_3)
# 
# 上記3つの結合結果を表示せよ


sample_dict_2 = {
    'id': ['100', '101', '102', '103', '104', '106', '108', '110', '111', ' 113'],
    'city': ['Tokyo', 'Osaka', 'Kyoto', 'Hokkaido', 'Tokyo', 'Tokyo', 'Osaka', 'Kyoto', 'Hokkaido', 'Tokyo'],
    'birth_year': [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
    'name': ['Hiroshi', 'Akiko', 'Yuki', 'Satoru', 'Steeve', 'Mituru', 'Aoi', 'Tarou', 'Suguru', 'Mitsuo']
}
sample_dict_3 =  {
    'id': ['100', '101', '102', '105', '107'],
    'math': [50, 43, 33, 76, 98],
    'english': [90, 30, 20, 50, 30],
    'sex': ['M', 'F', 'F', 'M', 'M'], 
    'index_num': [0, 1, 2, 3, 4]
}
# 以下よりコードを記入してください  ##############################
df_s_2 = pd.DataFrame(sample_dict_2)
df_s_3 = pd.DataFrame(sample_dict_3)

df_5_1 = pd.merge(df_s_2,df_s_3, on = "id")
df_5_2 = pd.merge(df_s_2,df_s_3, how = "outer")
df_5_3 = pd.merge(df_s_2,df_s_3, how = "left")

# 出力  #################################################
display(df_5_1)
display(df_5_2)
display(df_5_3)

# 課題1： sample_dictをDataFrameに変換せよ
# 
# その際に"a, b, c, d, e"をインデックスとして追加すること（変数名はdf_1）


sample_dict = {'ID':['100','101','102','103','104'],
               'City':['Tokyo','Osaka','Kyoto','Hokkaido','Tokyo'],
               'Birth_year':[1990,1989,1992,1997,1982],
               'Name':['Hiroshi','Akiko','Yuki','Satoru','Steve']}
# 以下よりコードを記入してください  ##############################
df_1 = pd.DataFrame(sample_dict,index=["a","b","c","d","e"])

# 出力  #################################################
display(df_1)

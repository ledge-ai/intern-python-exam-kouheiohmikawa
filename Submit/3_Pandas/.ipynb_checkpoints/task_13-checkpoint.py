# 課題13： data/CarPrice_Assignment.csvを読み込み 'CarName'のトップ3をlistで出力し、その値を持つデータを取り出して出力せよ
# 
# （変数名はそれぞれlist_13, df_13）

# In[ ]:


# データの読み込み
car_price_assign_df = pd.read_csv('../data/CarPrice_Assignment.csv')


# 以下よりコードを記入してください  ##############################
list_13 = car_price_assign_df["CarName"].value_counts().index[0:3].tolist()
df_13 = car_price_assign_df[car_price_assign_df["CarName"].isin(list_13)]


# 出力  #################################################
display(list_13)
display(df_13)

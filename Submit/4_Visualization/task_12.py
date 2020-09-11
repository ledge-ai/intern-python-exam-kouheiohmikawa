# 課題12: 課題11と同じ内容を、データセットの文字列型変数全て（CarNameを除く全9変数）に適用せよ
# 
# 可視化にあたり、複数グラフを並列に表示し、一つの図として表現せよ
# 
# 配置は行方向に3つ、列方向に3つずつ並べること。並び順は問わない
# 
# 各グラフのタイトルに変数名を記載すること


# 以下よりコードを記入してください  ##############################
df_12 = df.select_dtypes("object").drop("CarName",axis=1)
df_12_list = df_12.columns.values
df_12_list[0]

plt.figure(figsize=(12,12))
for i in range(0,9):
    a = df_12_list[i]
    
    plt.subplot(3,3,i+1)
    x_list = df.groupby(a)["price"].mean().index.to_list()
    y_array = df.groupby(a)["price"].mean().values
    plt.bar(x_list,y_array)
    plt.title(a)


    
    
    

# 解き直し ##############################
# 以下よりコードを記入してください  ##############################
df_12 = df.select_dtypes("object").drop("CarName",axis=1)
df_12_list = df_12.columns.values
df_12_list[0]

plt.figure(figsize=(16,12))
for i in range(0,9):
    a = df_12_list[i]
    
    plt.subplot(3,3,i+1)
    x_list = df.groupby(a)["price"].mean().index.to_list()
    y_array = df.groupby(a)["price"].mean().values
    plt.bar(x_list,y_array)
    plt.title(a)
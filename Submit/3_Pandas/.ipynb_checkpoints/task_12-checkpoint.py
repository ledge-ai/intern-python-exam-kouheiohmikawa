# 課題12： name_dictをDataFrameに変換し以下の二つの条件を満たすDataFrameを出力せよ（変数名はname_df）
# - 'name'の値はfirst,middle,lastに分割し、それぞれを個別のカラムに格納してname_dfに追加
# - 'location'の","より前の値をcityカラムに格納してname_dfに追加


name_dict = {"name":["John Artur Doe", "Jane Ann Smith", "Nico P"], 
     "location":["Los Angeles, CA", "Washington, DC", "Barcelona, Spain"]}
# 以下よりコードを記入してください  ##############################

name_df = pd.DataFrame(name_dict)

first = name_df["name"].map(lambda x:x.split()[0])
middle = name_df["name"].map(lambda x:x.split()[1])
last = name_df["name"][0:2].map(lambda x:x.split()[2])
last[2] = np.nan

name_df["first"] = first
name_df["middle"] = middle
name_df["last"] = last

city = name_df["location"].map(lambda x:x.split(",")[0])

name_df["city"] = city

# 出力  #################################################
display(name_df)

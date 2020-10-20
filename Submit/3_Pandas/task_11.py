# 課題11： age_dfの'age'カラムの値に対して'age_group'カラムを追加して、
# 
# - 0 ~ 18までを'kids'
# - 19 ~ 65までを'adult'
# - 65 ~ 99までを'elderly'
# 
# と分類し表示せよ（変数名はage_df）


age_df = numeric_df.rename(columns={0: 'age'}, inplace=False)
# 以下よりコードを記入してください  ##############################
bins = [0,18,65,99]
labels = ["kids","adult","elderly"]
age_df["age_group"] = pd.cut(age_df["age"],bins,labels = labels)

# 出力  #################################################
display(age_df)

# 課題14： car_price_assign_dfの'carbody'に対して頻出度10以下を'others'に置き換えて出力せよ。以下の変数で対応する値を出力せよ。
# - frequencies: car_price_assign_dfの'carbody'の値と出現回数
# - small_categories: 閾値10回に満たない'carbody'の値と出現回数
# - others_freq: othersに置き換えた後の'carbody'の値と出現回数


# 以下よりコードを記入してください  ##############################
frequencies = car_price_assign_df["carbody"].value_counts()

small_categories = frequencies[frequencies<10].index.to_list()

car_price_assign_df = car_price_assign_df.replace({small_categories[0]:"others"})
car_price_assign_df = car_price_assign_df.replace({small_categories[1]:"others"})
others_freq = car_price_assign_df["carbody"].value_counts()

# 出力  #################################################
print(frequencies)
print('#' * 20)
print(small_categories)
print('#' * 20)
print(others_freq)



# 解き直し  ##############################
# 以下よりコードを記入してください  ##############################
frequencies = car_price_assign_df["carbody"].value_counts()

t = 10
small_categories = frequencies[frequencies<t].index.to_list()

car_price_assign_df = car_price_assign_df.replace({small_categories[0]:"others"})
car_price_assign_df = car_price_assign_df.replace({small_categories[1]:"others"})
others_freq = car_price_assign_df["carbody"].value_counts()
# 出力  #################################################
print(frequencies)
print('#' * 20)
print(small_categories)
print('#' * 20)
print(others_freq)
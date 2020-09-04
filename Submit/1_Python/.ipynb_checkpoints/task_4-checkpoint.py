# 課題4: FuzzBuzz問題
# 
# 1~50の数値で、4で割り切れる場合は「Fuzz」、9で割り切れる場合は「Buzz」、両者で割り切れる場合は「FuzzBuzz」に変換し、リスト「ans_list」に格納せよ
# 
# なお、上記条件に当てはまらないものは元の数値のまま格納すること

ans_list = []

# 以下よりコードを記入してください  ##############################
for i in range(1,51):
    if i % 36 == 0:
        ans_list.append("FuzzBuzz")
    elif i % 4 == 0:
        ans_list.append("Fuzz")
    elif i % 9 == 0:
        ans_list.append("Buzz")
    else:
        ans_list.append(i)
        
# 出力  #################################################
print(ans_list)

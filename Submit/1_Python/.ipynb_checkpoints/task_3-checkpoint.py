# 課題3: 与えられたリスト「l」に対して、要素ごとの文字数が偶数のものと奇数のものを分けるプログラムを作成せよ
# 
# 偶数のものは「even」、奇数のものは「odd」に格納すること

l = ['you', 'are', 'so', 'cool', 'data', 'scientist']
even = [] # 偶数
odd = []  # 奇数

# 以下よりコードを記入してください  ##############################
for i in range(6):
    if len(l[i]) %2 == 0:
        even.append(l[i])
    else:
        odd.append(l[i])

# 出力  #################################################
print('Even: ', even)
print('Odd: ', odd)

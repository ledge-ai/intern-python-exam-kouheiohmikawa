# 課題８: オリジナルのプログラムを実装してみよう
# 
# 自分のアイデアを活かしたプログラムを自由に作成せよ
# 
# プログラムの説明（どういうプログラムなのか）と実装コード、および出力結果を記載すること
# 
# 工夫した点などが記載されていると尚良い
# 
# アイデア例として、
# 
# - 特殊な性質を持つ整数を出力するプログラム： 「友愛数」・「カプレカ数」など
# 
# - 数学的に有名な数列を出力するプログラム： 「パスカルの三角形」など
# 
# - 日常的に使えそうなプログラム: 「日付を入力すると曜日を算出する」

# 以下よりコードを記入してください  ##############################

#1~2000までの友愛数を求める

def div_sum(p):
    return sum([n for n in range(1,p) if p % n == 0])

def Fraternity(min, max):
    Fraternity_list = []
    for n in range(min, max + 1):
        div = div_sum(n)
        if n == div_sum(div) and n != div:
            m = sorted([n,div])
            if not Fraternity_list.count(m):
                Fraternity_list.append(m)
    return Fraternity_list

Fraternity(1, 2000)
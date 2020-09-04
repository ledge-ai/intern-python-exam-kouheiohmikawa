# 課題6: フィボナッチ数列
# 
# 1~100,000までの整数の範囲のフィボナッチ数列を表示せよ
# 
# 数列はリスト型で作成し、変数名は「Fibonacci」とすること

Fibonacci = [0, 1]
# 以下よりコードを記入してください  ##############################
i = 0
while max(Fibonacci) <= 100000:
    a = Fibonacci[i] + Fibonacci[i+1]
    i += 1
    Fibonacci.append(a)

Fibonacci.pop(-1)

# 出力  #################################################
print(Fibonacci)

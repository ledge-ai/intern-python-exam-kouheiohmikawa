# 課題5: 素数判定プログラムを作成し、5桁の素数で最も大きいものを出力せよ（回答結果は「ans」に格納すること）

# 以下よりコードを記入してください  ##############################
def calc(N):
    ans_list = range(2, N+1)
    
    for i in range(2,int(N ** 0.5) + 1):
        ans_list = [a for a in ans_list if (a == i or a % i != 0) ]
    
    return max(ans_list)

ans = calc(100000)

# 出力  #################################################
print(ans)

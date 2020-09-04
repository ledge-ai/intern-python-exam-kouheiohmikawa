# 課題11: 課題8で得られた「A」「B」それぞれの行列式を算出せよ
# 
# （変数名はそれぞれ「det_A」, 「det_B」）


# 以下よりコードを記入してください  ##############################
import scipy.linalg as linalg
    
det_A = linalg.det(A)
det_B = linalg.det(B)

# 出力  #################################################
print(det_A)
print(det_B)

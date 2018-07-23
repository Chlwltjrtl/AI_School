# 실습에 필요한 라이브러리를 불러옵니다.
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 선형회귀분석에 필요한 LinearRegression 클래스를 불러옵니다. 
from sklearn.linear_model import LinearRegression

# 결과를 출력하기 위한 엘리스 유틸리티 툴을 불러옵니다.
import elice_utils
eu = elice_utils.EliceUtils()


# [과제 1]에서 작성한 loss 함수를 붙여 넣으세요.
def loss(x, y, beta_0, beta_1):    
    N = len(x)
    y = np.array(y)
    x = np.array(x) 
    l = np.square((beta_0*x+ beta_1)-y)
    sum=np.sum(l)
    return sum


# 실습에 사용되는 데이터입니다.
X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

# X, Y 데이터를 NumPy 행렬로 변환하세요.
train_X = np.array(X)
train_X = train_X.reshape(-1,1)
train_Y = np.array(Y)

# 선형회귀 모델을 학습하는 코드를 작성하세요.
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

# 학습한 모델을 통해 얻은 기울기와 절편을 입력하세요.
beta_0 = lrmodel.coef_
beta_1 = lrmodel.intercept_


# 결과를 출력합니다.
print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("Loss: %f" % loss(X, Y, beta_0, beta_1))

# 데이터와 y = beta_0 * x + beta_1 에 해당하는 선을 표시합니다. 
plt.scatter(X, Y)
plt.plot([0, 10], [beta_1, 10 * beta_0 + beta_1], c='r')

 # 그래프의 X, Y축을 설정합니다.
plt.xlim(0, 10)
plt.ylim(0, 10)

# 엘리스에 이미지를 출력합니다.
plt.savefig("test.png") 
eu.send_image("test.png")

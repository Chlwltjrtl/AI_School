# using data.png
# 실습에 필요한 라이브러리를 불러옵니다.
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# x, y, beta_0, beta_1 을 이용해 loss 값을 계산합니다.
# 밑 코드가 과제입니다. 지우고 해보시길 바랍니다.
def loss(x, y, beta_0, beta_1):
    N = len(x)
    y = np.array(y)
    x = np.array(x)
    l = np.square((beta_0 * x + beta_1) - y)
    sum = np.sum(l)
    return sum


# 실습에 사용되는 데이터입니다.
X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513,
     5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441,
     5.19692852]

# 기울기와 절편이 1, 0.5일 때 Loss를 계산합니다.
beta_0 = 1
beta_1 = 0.5

# 결과를 출력합니다.
print("Loss: %f" % loss(X, Y, beta_0, beta_1))

# 데이터와 y = beta_0 * x + beta_1 에 해당하는 선을 표시합니다.
plt.scatter(X, Y)
plt.plot([0, 10], [beta_1, 10 * beta_0 + beta_1], c='r')

# 그래프의 X, Y축을 설정합니다.
plt.xlim(0, 10)
plt.ylim(0, 10)

# 엘리스에 이미지를 출력합니다.
plt.savefig("test.png")

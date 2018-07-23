# 실습에 필요한 라이브러리를 불러옵니다.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 결과를 출력하기 위한 엘리스 유틸리티 툴을 불러옵니다.
import elice_utils
eu = elice_utils.EliceUtils()


# 데이터를 불러옵니다.
data = pd.read_csv('data/data.txt', header=None)
data = np.array(data)
X = data[:, [0,1]]
y = data[:, 2]
#y = y.reshape(len(y), -1)

# Logistic 회귀 모델을 구현합니다.
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=333333, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    # sigmoid 함수를 작성하세요.
    def sigmoid(self, z):
        Gz = 1.0 / (1.0 + np.exp(-z))
        return Gz

    # loss 함수를 작성하세요.
    def loss(self, h, y):
        m = y.size
        cost = -(1 / m) * sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    # 모델을 학습하는 fit 함수를 작성하세요.
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h-y)) / y.size
            self.theta = self.theta - self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            loss = self.loss(h, y)

        return z, h, loss

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(np.dot(X, self.theta))

    # predict_prob 함수를 이용해 predict 함수를 완성하세요.
    def predict(self, X):
        return np.round(self.predict_prob(X))


# 모델을 학습합니다.
model = LogisticRegression()
model.fit(X, y)


# 결과를 출력합니다.
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()

x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');

plt.savefig("binary_classification.png")
eu.send_image("binary_classification.png")

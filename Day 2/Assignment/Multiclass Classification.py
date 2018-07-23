import numpy as np
import matplotlib.pyplot as plt

# 숫자 인식을 위한 데이터를 불러옵니다. 
from sklearn.datasets import load_digits
# 학습용, 테스트용으로 분류하기 위한 함수를 불러옵니다.
from sklearn.model_selection import train_test_split
# Logistic 회귀분석을 위한 클래스를 불러옵니다. 
from sklearn.linear_model import LogisticRegression

# 결과를 출력하기 위한 엘리스 유틸리티 툴을 불러옵니다.
import elice_utils
eu = elice_utils.EliceUtils()


# sklearn의 내장 숫자 데이터를 load_digits 함수를 이용해 저장합니다.
digits = load_digits()
# 아래 주석을 해제하고 데이터를 확인하세요.
print(digits.data.shape, digits.target.shape)

# 데이터를 이미지로 출력합니다.
plt.figure(figsize=(20,5))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title("Training: {}\n".format(label), fontsize=15)
    
plt.savefig("dataset.png") 
eu.send_image("dataset.png")


# 데이터를 학습용, 테스트용으로 분리합니다.
x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state = 0)
                                   
                                   
# 아래 주석을 해제하고 학습용, 테스트용 데이터를 확인하세요.
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Logistic 회귀분석 모델을 학습합니다.
model = LogisticRegression()
model.fit(x_train,y_train)

# 모델의 예측값과 정확도를 저장합니다.
predictions = model.predict(x_test)
score = model.score(x_test,y_test)

# 예측값, 실제값, 정확도를 출력합니다.
print("Predicted:", predictions[:10])
print("Actual:", y_test[:10])
print("Score:", score)


# 잘못 분류된 데이터를 이미지로 출력합니다.
index = 0

misclassifiedIndexes = []
for label, predict in zip(y_test, predictions):
    if label != predict: 
        misclassifiedIndexes.append(index)
    index +=1

plt.figure(figsize=(20,5))
for plotIndex, wrong in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}".format(predictions[wrong], y_test[wrong]), fontsize = 15)
    
plt.savefig("misclassified.png") 
eu.send_image("misclassified.png")

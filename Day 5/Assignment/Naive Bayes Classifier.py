# 실습에 필요한 라이브러리를 불러옵니다.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 결과를 출력하기 위한 엘리스 유틸리티 툴을 불러옵니다.
import elice_utils
eu = elice_utils.EliceUtils()

np.random.seed(42)

# 데이터를 불러옵니다.
datafile = open('data/spambase.data', 'r')
data = []
for line in datafile:
    line = [float(element) for element in line.rstrip('\n').split(',')]
    data.append(np.asarray(line))

num_features = 48

# feature와 label을 구분해줍니다.
X = [data[i][:num_features] for i in range(len(data))]
y = [int(data[i][-1]) for i in range(len(data))]

# train, test 데이터셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

class Naive_Bayes_Classifier(object):
    # 48개의 feature를 이용합니다
    def __init__(self, num_features=48):
        self.num_features = num_features

    # (2) log(P(feature_vector | Class))를 계산합니다.
    # likeihood = P(x|c)
    def log_likelihood_naivebayes(self, feature_vector, Class):
        assert len(feature_vector) == self.num_features
        log_likelihood = 0.0 #log-likelihood를 사용해 underflow 회피
        
        if Class == 0: # ham
            for feature in range(len(feature_vector)):
                if feature_vector[feature]==1:
                    log_likelihood = log_likelihood + np.log10(self.likelihoods_ham[feature])
                else:
                    log_likelihood = log_likelihood + np.log10(1-self.likelihoods_ham[feature])
        elif Class == 1: # spam
            for feature in range(len(feature_vector)):
                if feature_vector[feature]==1:
                    log_likelihood = log_likelihood + np.log10(self.likelihoods_spam[feature])
                else:
                    log_likelihood = log_likelihood + np.log10(1-self.likelihoods_spam[feature])
        else:
            raise ValueError("Class takes integer values 0 or 1")
        
        return log_likelihood
    
    # 3. 각 클래스의 Posterior probability를 구합니다.
    #bayes rule's : (likelihood P(x|C) * P(C) prior) / P(x) evidence
    def class_posteriors(self, feature_vector):
        log_likelihood_ham = self.log_likelihood_naivebayes(feature_vector, Class = 0) # ham 
        log_likelihood_spam = self.log_likelihood_naivebayes(feature_vector, Class = 1) # spam
        
        log_posterior_ham = log_likelihood_ham + self.log_prior_ham
        log_posterior_spam = log_likelihood_spam + self.log_prior_spam
        
        return log_posterior_ham, log_posterior_spam
    
    # Maximum A Priori(MAP) inference를 이용해 사후확률이 가장 큰 클래스를 선택합니다.
    # C = 1 이면 if (P(C=1|x)>P(C=0|x))

    def spam_classify(self, document):
        feature_vector = [int(element>0.0) for element in document]
        log_posterior_ham, log_posterior_spam = self.class_posteriors(feature_vector)
        
        if log_posterior_ham > log_posterior_spam:
            return 0
        else:
            return 1


    # (1)모델을 학습하는 train 함수를 작성하세요.
    def train(self, X_train, y_train):
        # Likelihood estimator 만들기
        # 스팸 클래스와 햄 클래스 나누기
        X_train_spam = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1]
        X_train_ham = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
    
        # 각 클래스의 feature 각각에 대한 likelihood 구하세요.
        # P(s|w)=P(s,w)P(w) = P(w|s)P(s)P(w) : 스팸일 확률 
        # likelihood -> p(X|c) C가 주어졌을때 x의 확률 (log)
        self.likelihoods_ham =  np.mean(X_train_ham, axis=0) / 100
        self.likelihoods_spam = np.mean(X_train_spam, axis=0) / 100

        # 각 class의 log-prior를 계산하세요
        num_ham = float(len(X_train_ham))
        num_spam = float(len(X_train_spam))
        
        #확률 
	    #Non-Spam  2788  (60.6%)
        prior_probability_ham = num_ham / (num_ham + num_spam)
	    #Spam	  1813  (39.4%)
        prior_probability_spam = num_spam / (num_ham + num_spam)

        self.log_prior_ham = np.log10(prior_probability_ham)
        self.log_prior_spam = np.log10(prior_probability_spam)
    
        return self.log_prior_ham, self.log_prior_spam
    
    def predict(self, X_test):
        predictions = []
        for case in X_test:
            predictions.append(self.spam_classify(case))
        
        return predictions


NB = Naive_Bayes_Classifier()
NB.train(X_train, y_train)
pred = NB.predict(X_test)

def evaluate_performance(predictions, ground_truth_labels):
    correct_count = 0.0
    for item_index in range(len(predictions)):
        if predictions[item_index] == ground_truth_labels[item_index]:
            correct_count += 1.0
    accuracy = correct_count/len(predictions)
    return accuracy

accuracy_naivebayes = evaluate_performance(pred, y_test)
print(accuracy_naivebayes)

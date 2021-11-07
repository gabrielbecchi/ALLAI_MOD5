import pandas as p
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# LOADING DATA
data = load_breast_cancer()

# PRE-PROCESSING
scaler = preprocessing.MinMaxScaler().fit(data.data)
data.data = scaler.transform(data.data)

# CLASSIFICATIONS 
scoring = ['accuracy', 'f1','precision','recall']
scores = {}

# LOGISTIC REGRESSION
engine = LogisticRegression()
scores['LR'] = cross_validate(engine, data.data, data.target, scoring=scoring)

# NAIVE BAYES
engine = GaussianNB()
scores['NB'] = cross_validate(engine, data.data, data.target, scoring=scoring)

# KNN
engine = KNeighborsClassifier()
scores['KNN'] = cross_validate(engine, data.data, data.target, scoring=scoring)

# DECISION TREE
engine = DecisionTreeClassifier()
scores['DT'] = cross_validate(engine, data.data, data.target, scoring=scoring)

# SVD
engine = SVC()
scores['SVD'] = cross_validate(engine, data.data, data.target, scoring=scoring)

for method, score in scores.items():
	train = np.mean(score['fit_time'])
	test = np.mean(score['score_time'])
	accuracy = np.mean(score['test_accuracy'])
	precision = np.mean(score['test_precision'])
	recall = np.mean(score['test_recall'])
	f1 = np.mean(score['test_f1'])
	print(method+' '*(7-len(method))+"""TRAIN: {:.5f}s, TESTE: {:.5f}s, ACCURACY: {:.5f}, PRECISION: {:.5f}, RECALL: {:.5f}, F1: {:.5f}""".format(train, test, accuracy, precision, recall, f1))



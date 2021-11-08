import pandas as p
import numpy as np

from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier

# LOADING DATA
data = p.read_csv('drug200.csv')

# PRE-PROCESSING
target = data['Drug']
del data['Drug']
encoder = preprocessing.LabelEncoder()
encoder.fit(target.unique())
target = encoder.transform(target)

for column_name in data.columns:
	if(data[column_name].dtype == object):
		aux = p.get_dummies(data[column_name], prefix=column_name)
		del data[column_name]
		data = p.concat([data,aux], axis=1)

scaler = preprocessing.MinMaxScaler().fit(data)
data = scaler.transform(data)

# CLASSIFICATIONS 
scoring = ['accuracy', 'f1_micro','precision_micro','recall_micro']
scores = {}

# LOGISTIC REGRESSION
engine = OneVsRestClassifier(LogisticRegression())
scores['LR_OvR'] = cross_validate(engine, data, target, scoring=scoring)
engine = OneVsOneClassifier(LogisticRegression())
scores['LR_OvO'] = cross_validate(engine, data, target, scoring=scoring)
engine = OutputCodeClassifier(LogisticRegression())
scores['LR_OC'] = cross_validate(engine, data, target, scoring=scoring)

# NAIVE BAYES
engine = OneVsRestClassifier(GaussianNB())
scores['NB_OvR'] = cross_validate(engine, data, target, scoring=scoring)
engine = OneVsOneClassifier(GaussianNB())
scores['NB_OvO'] = cross_validate(engine, data, target, scoring=scoring)
engine = OutputCodeClassifier(GaussianNB())
scores['NB_OC'] = cross_validate(engine, data, target, scoring=scoring)

# KNN
engine = OneVsRestClassifier(KNeighborsClassifier())
scores['KNN_OvR'] = cross_validate(engine, data, target, scoring=scoring)
engine = OneVsOneClassifier(KNeighborsClassifier())
scores['KNN_OvO'] = cross_validate(engine, data, target, scoring=scoring)
engine = OutputCodeClassifier(KNeighborsClassifier())
scores['KNN_OC'] = cross_validate(engine, data, target, scoring=scoring)

# DECISION TREE
engine = OneVsRestClassifier(DecisionTreeClassifier())
scores['DT_OvR'] = cross_validate(engine, data, target, scoring=scoring)
engine = OneVsOneClassifier(DecisionTreeClassifier())
scores['DT_OvO'] = cross_validate(engine, data, target, scoring=scoring)
engine = OutputCodeClassifier(DecisionTreeClassifier())
scores['DT_OC'] = cross_validate(engine, data, target, scoring=scoring)

# SVD
engine = OneVsRestClassifier(SVC())
scores['SVD_OvR'] = cross_validate(engine, data, target, scoring=scoring)
engine = OneVsOneClassifier(SVC())
scores['SVD_OvO'] = cross_validate(engine, data, target, scoring=scoring)
engine = OutputCodeClassifier(SVC())
scores['SVD_OC'] = cross_validate(engine, data, target, scoring=scoring)

for method, score in scores.items():
	train = np.mean(score['fit_time'])
	test = np.mean(score['score_time'])
	accuracy = np.mean(score['test_accuracy'])
	precision = np.mean(score['test_precision_micro'])
	recall = np.mean(score['test_recall_micro'])
	f1 = np.mean(score['test_f1_micro'])
	print(method+' '*(10-len(method))+"""TRAIN: {:.5f}s, TESTE: {:.5f}s, ACCURACY: {:.5f}, PRECISION: {:.5f}, RECALL: {:.5f}, F1: {:.5f}""".format(train, test, accuracy, precision, recall, f1))


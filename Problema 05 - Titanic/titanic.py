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
# SELECIONE AS COLUNAS
'''
colunas = ['PassengerId','Survived','Pclass','Name','Sex','Age',
	'SibSp','Parch','Ticket','Fare','Cabin','Embarked']
'''

colunas = ['PassengerId','Survived','Pclass','Sex','Age',
	'SibSp','Parch','Fare','Cabin','Embarked']

data = p.read_csv('train.csv',usecols=colunas)

# PRE-PROCESSING

# TRATANDO NaN
#data = data.dropna(axis=0, how='any')
data = data.fillna(data.mean())

# TRATANDO COLUNAS
aux = p.get_dummies(data['Sex'], prefix='Sex')
del data['Sex']
data = p.concat([data,aux], axis=1)

aux = p.get_dummies(data['Cabin'], prefix='Cabin')
del data['Cabin']
data = p.concat([data,aux], axis=1)

aux = p.get_dummies(data['Embarked'], prefix='Embarked')
del data['Embarked']
data = p.concat([data,aux], axis=1)

# LABEL
encoder = preprocessing.LabelEncoder()
encoder.fit(data['Survived'].unique())
label = encoder.transform(data['Survived'])
del data['Survived']
print(len(label))

# ESCALARES
scaler = preprocessing.MinMaxScaler().fit(data)
data = scaler.transform(data)

# CLASSIFICATIONS 
scoring = ['accuracy', 'f1','precision','recall']
scores = {}

# LOGISTIC REGRESSION
engine = LogisticRegression()
scores['LR'] = cross_validate(engine, data, label, scoring=scoring)

# NAIVE BAYES
engine = GaussianNB()
scores['NB'] = cross_validate(engine, data, label, scoring=scoring)

# KNN
engine = KNeighborsClassifier()
scores['KNN'] = cross_validate(engine, data, label, scoring=scoring)

# DECISION TREE
engine = DecisionTreeClassifier()
scores['DT'] = cross_validate(engine, data, label, scoring=scoring)

# SVD
engine = SVC()
scores['SVD'] = cross_validate(engine, data, label, scoring=scoring)

for method, score in scores.items():
	train = np.mean(score['fit_time'])
	test = np.mean(score['score_time'])
	accuracy = np.mean(score['test_accuracy'])
	precision = np.mean(score['test_precision'])
	recall = np.mean(score['test_recall'])
	f1 = np.mean(score['test_f1'])
	print(method+' '*(7-len(method))+"""TRAIN: {:.5f}s, TESTE: {:.5f}s, ACCURACY: {:.5f}, PRECISION: {:.5f}, RECALL: {:.5f}, F1: {:.5f}""".format(train, test, accuracy, precision, recall, f1))



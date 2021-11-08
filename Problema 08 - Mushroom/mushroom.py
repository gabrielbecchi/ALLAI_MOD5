import pandas as p
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier

from single_model import test_single_model

# LOADING DATA
data = p.read_csv('mushrooms.csv')

# PRE-PROCESSING
target = data['class']
del data['class']
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

scores = {}
scoring = ['accuracy', 'f1_micro','precision_micro','recall_micro']
scores = test_single_model(data,target)

# ENSEMBLES

# HARD VOTING
engine_lr = LogisticRegression()
engine_nb = GaussianNB()
engine_knn = KNeighborsClassifier()
engine_dt = DecisionTreeClassifier()
engine_svm = SVC()
ensemble = VotingClassifier(estimators=[('LR',engine_lr),('NB',engine_nb),
	('KNN',engine_knn),('DT',engine_dt),('SVM',engine_svm)],voting='hard')
scores['HARD_VOTING'] = cross_validate(ensemble, data, target, scoring=scoring)

# SOFT VOTING
engine_lr = LogisticRegression()
engine_nb = GaussianNB()
engine_knn = KNeighborsClassifier()
engine_dt = DecisionTreeClassifier()
engine_svm = SVC(probability=True)
ensemble = VotingClassifier(estimators=[('LR',engine_lr),('NB',engine_nb),
	('KNN',engine_knn),('DT',engine_dt),('SVM',engine_svm)],voting='soft')
scores['SOFT_VOTING'] = cross_validate(ensemble, data, target, scoring=scoring)

# BAGGING
engine_lr = LogisticRegression()
ensemble = BaggingClassifier(engine_lr)
scores['BAGGING_LR'] = cross_validate(ensemble, data, target, scoring=scoring)

engine_nb = GaussianNB()
ensemble = BaggingClassifier(engine_nb)
scores['BAGGING_NB'] = cross_validate(ensemble, data, target, scoring=scoring)

engine_knn = KNeighborsClassifier()
ensemble = BaggingClassifier(engine_knn)
scores['BAGGING_KNN'] = cross_validate(ensemble, data, target, scoring=scoring)

engine_dt = DecisionTreeClassifier()
ensemble = BaggingClassifier(engine_dt)
scores['BAGGING_DT'] = cross_validate(ensemble, data, target, scoring=scoring)

engine_svm = SVC()
ensemble = BaggingClassifier(engine_svm)
scores['BAGGING_SVM'] = cross_validate(ensemble, data, target, scoring=scoring)

# RANDOM FOREST
ensemble = RandomForestClassifier()
scores['RANDOM_FOREST'] = cross_validate(ensemble, data, target, scoring=scoring)

# GRADIENT BOOSTING < IMPLEMENTADO NO SKLEARN COM DECISION TREE
ensemble = GradientBoostingClassifier()
scores['GRAD_BOOSTING'] = cross_validate(ensemble, data, target, scoring=scoring)

# ADABOOST < SUPORTA ALGUNS LEARNER NO SKLEARN
engine_lr = LogisticRegression()
ensemble = AdaBoostClassifier(engine_lr)
scores['ADABOOST_LR'] = cross_validate(ensemble, data, target, scoring=scoring)

engine_nb = LogisticRegression()
ensemble = AdaBoostClassifier(engine_nb)
scores['ADABOOST_NB'] = cross_validate(ensemble, data, target, scoring=scoring)

engine_dt = DecisionTreeClassifier()
ensemble = AdaBoostClassifier(engine_dt)
scores['ADABOOST_DT'] = cross_validate(ensemble, data, target, scoring=scoring)


# STACKING
engine_lr = OneVsRestClassifier(LogisticRegression())
engine_nb = GaussianNB()
engine_knn = KNeighborsClassifier()
engine_dt = DecisionTreeClassifier()
engine_svm = SVC()
ensemble = StackingClassifier(estimators=[('NB',engine_nb),
	('KNN',engine_knn),('DT',engine_dt),('SVM',engine_svm)],
	final_estimator=engine_lr)
scores['STCK_LR_OvR'] = cross_validate(ensemble, data, target, scoring=scoring)

# STACKING 2
engine_lr = LogisticRegression()
engine_nb = GaussianNB()
engine_knn = OneVsOneClassifier(KNeighborsClassifier())
engine_dt = DecisionTreeClassifier()
engine_svm = SVC()
ensemble = StackingClassifier(estimators=[('NB',engine_nb),
	('LR',engine_lr),('DT',engine_dt),('SVM',engine_svm)],
	final_estimator=engine_knn)
scores['STCK_KNN_OvO'] = cross_validate(ensemble, data, target, scoring=scoring)

for method, score in scores.items():
	train = np.mean(score['fit_time'])
	test = np.mean(score['score_time'])
	accuracy = np.mean(score['test_accuracy'])
	precision = np.mean(score['test_precision_micro'])
	recall = np.mean(score['test_recall_micro'])
	f1 = np.mean(score['test_f1_micro'])
	print(method+' '*(15-len(method))+"""TRAIN: {:.5f}s, TESTE: {:.5f}s, ACCURACY: {:.5f}, PRECISION: {:.5f}, RECALL: {:.5f}, F1: {:.5f}""".format(train, test, accuracy, precision, recall, f1))

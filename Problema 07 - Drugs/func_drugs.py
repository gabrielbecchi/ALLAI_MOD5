import pandas as p
import numpy as np

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
scoring = ['accuracy', 'f1_weighted','precision_weighted','recall_weighted']



def drug_classifier(k):
	engine = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=k))
	res = cross_validate(engine, data, target, scoring=scoring, cv=2)
	return np.mean(res['test_f1_weighted'])
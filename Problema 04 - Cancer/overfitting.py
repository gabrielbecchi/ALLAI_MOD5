import pandas as p
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import f1_score

# LOADING DATA
data = load_breast_cancer()

# PRE-PROCESSING
scaler = preprocessing.MinMaxScaler().fit(data.data)
data.data = scaler.transform(data.data)

X_train, X_test, y_train, y_test = train_test_split(data.data,data.target,test_size=0.80)


engine = DecisionTreeClassifier(criterion='entropy',max_depth=100,
	min_samples_split=2,min_samples_leaf=1,max_features=1)

engine.fit(X_train,y_train)

y_predicted = engine.predict(X_train)
f1_test = f1_score(y_train, y_predicted)
print(f1_test)

y_predicted = engine.predict(X_test)
f1_test = f1_score(y_test, y_predicted)
print(f1_test)
import pandas as p
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_curve, auc

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

X_train, X_test, y_train, y_test = train_test_split(data.data,data.target,test_size=0.95)

# LOGISTIC REGRESSION
engine = LogisticRegression()
engine.fit(X_train,y_train)
y_predicted = engine.predict(X_test)

cm = confusion_matrix(y_test,y_predicted)
print("Logistic Regression")
print(cm)
svc_disp = plot_roc_curve(engine, X_test, y_test)

# NAIVE BAYES
engine = GaussianNB()
engine.fit(X_train,y_train)
y_predicted = engine.predict(X_test)

cm = confusion_matrix(y_test,y_predicted)
print("Naive Bayes")
print(cm)
plot_roc_curve(engine, X_test, y_test, ax=svc_disp.ax_) 

# KNN
engine = KNeighborsClassifier()
engine.fit(X_train,y_train)
y_predicted = engine.predict(X_test)

cm = confusion_matrix(y_test,y_predicted)
print("KNN")
print(cm)
plot_roc_curve(engine, X_test, y_test, ax=svc_disp.ax_) 

# DECISION TREE
engine = DecisionTreeClassifier()
engine.fit(X_train,y_train)
y_predicted = engine.predict(X_test)

cm = confusion_matrix(y_test,y_predicted)
print("DECISION TREE")
print(cm)
plot_roc_curve(engine, X_test, y_test, ax=svc_disp.ax_) 

# SVD
engine = SVC()
engine.fit(X_train,y_train)
y_predicted = engine.predict(X_test)

cm = confusion_matrix(y_test,y_predicted)
print("SVD")
print(cm)
plot_roc_curve(engine, X_test, y_test, ax=svc_disp.ax_) 

plt.show()


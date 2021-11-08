import pandas as p
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from pyswarm import pso

# LOADING DATA
data = load_breast_cancer()
# PRE-PROCESSING
scaler = preprocessing.MinMaxScaler().fit(data.data)
data.data = scaler.transform(data.data)
scoring = ['f1']

# HYPERPARAMETERS OTIMIZADOR
n_neighbors = [1,100]
weights = ['uniform', 'distance']
distance = ['euclidean','manhattan','chebyshev']

lower_bound = [1,0,0]
upper_bound = [100, len(weights)-1e-8, len(distance)-1e-8]

def funcao_objetivo(x):
	p_n_neighbors = int(x[0])
	p_weights = weights[int(x[1])]
	p_distance = distance[int(x[2])]
	engine = KNeighborsClassifier(n_neighbors=p_n_neighbors,
		weights=p_weights, metric=p_distance)
	result = cross_validate(engine, data.data, data.target, scoring=scoring)
	result = np.mean(result['test_f1'])
	return -1*result

xopt, fopt = pso(funcao_objetivo, lower_bound, upper_bound, 
	swarmsize=10, maxiter=100,minstep=1e-4)

print("Neighbors: %s" % int(xopt[0]))
print("Weights: %s" % weights[int(xopt[1])])
print("Distancia: %s" % distance[int(xopt[2])])
print("F1 = %0.2f" % (fopt*-1))

print(xopt)
print(fopt)

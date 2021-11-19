import pandas as p
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from pyswarm import pso

data = p.read_csv('housing.csv')
data = data.dropna(axis=0, how='any')
target = data['median_house_value']
del data['median_house_value']
aux = p.get_dummies(data['ocean_proximity'], prefix='ocean_proximity')
del data['ocean_proximity']
data = p.concat([data,aux], axis=1)

# SCORE
scoring = ['r2']

# HYPERPARAMETERS OTIMIZADOR
lower_bound = [0,0]
upper_bound = [10,0.999]

def funcao_objetivo(x):
	engine = ElasticNet(alpha=x[0], l1_ratio=x[1])
	result = cross_validate(engine, data, target, scoring=scoring,cv=3)
	result = np.mean(result['test_r2'])
	print("Alpha: %s, Ratio: %s -> R2: %s" % (x[0],x[1],result))
	return -1*result

xopt, fopt = pso(funcao_objetivo, lower_bound, upper_bound, 
	swarmsize=10, maxiter=100,minstep=1e-8)

print(xopt)
print(fopt)

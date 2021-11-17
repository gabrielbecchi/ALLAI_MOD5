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

data = p.read_csv('car.csv')

# TRATANDO NaN
#data = data.dropna(axis=0, how='any')
data = data.fillna(data.mean())

del data['name']

# PRE-PROCESSING
target = data['selling_price']
del data['selling_price']

# DUMMIES
aux = p.get_dummies(data['fuel'], prefix='fuel')
del data['fuel']
data = p.concat([data,aux], axis=1)

aux = p.get_dummies(data['seller_type'], prefix='seller_type')
del data['seller_type']
data = p.concat([data,aux], axis=1)

aux = p.get_dummies(data['transmission'], prefix='transmission')
del data['transmission']
data = p.concat([data,aux], axis=1)

aux = p.get_dummies(data['owner'], prefix='owner')
del data['owner']
data = p.concat([data,aux], axis=1)

del data['mileage']
del data['engine']
del data['max_power']
del data['torque']

scaler = preprocessing.MinMaxScaler().fit(data)
data = scaler.transform(data)

scores = {}
scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error',
		'neg_mean_squared_error','r2','explained_variance']

engine = LinearRegression(normalize=True)
scores['LINEAR_REG'] = cross_validate(engine, data, target, scoring=scoring)

engine = Lasso(max_iter=500)
scores['LASSO'] = cross_validate(engine, data, target, scoring=scoring)

engine = Ridge()
scores['RIDGE'] = cross_validate(engine, data, target, scoring=scoring)

engine = ElasticNet()
scores['ELASTICNET'] = cross_validate(engine, data, target, scoring=scoring)

engine = KNeighborsRegressor()
scores['KNN'] = cross_validate(engine, data, target, scoring=scoring)

engine = DecisionTreeRegressor()
scores['DT'] = cross_validate(engine, data, target, scoring=scoring)

#engine = SVR()
#scores['SVM'] = cross_validate(engine, data, target, scoring=scoring)

engine = AdaBoostRegressor()
scores['ADABOOST'] = cross_validate(engine, data, target, scoring=scoring)

#engine = Lasso()
#engine = BaggingRegressor(engine)
#scores['BAGGING'] = cross_validate(engine, data, target, scoring=scoring)

engine = GradientBoostingRegressor()
scores['BOOSTING'] = cross_validate(engine, data, target, scoring=scoring)

engine = RandomForestRegressor()
scores['RANDOM_FOREST'] = cross_validate(engine, data, target, scoring=scoring)

for method, score in scores.items():
	train = np.mean(score['fit_time'])
	test = np.mean(score['score_time'])
	rmse = -1*np.mean(score['test_neg_root_mean_squared_error'])
	mae = -1*np.mean(score['test_neg_mean_absolute_error'])
	mse = -1*np.mean(score['test_neg_mean_squared_error'])
	r2 = np.mean(score['test_r2'])
	ev = np.mean(score['test_explained_variance'])
	#print(method+' '*(15-len(method))+"""TRAIN: {:.5f}s, TESTE: {:.5f}s, RMSE: {:.5f}, MAE: {:.5f}, MSE: {:.5f}, R2: {:.5f}, EV: {:.5f}""".format(train, test, rmse, mae, mse, r2, ev))
	print(method+' '*(20-len(method))+"""TRAIN: {:.5f}s, TESTE: {:.5f}s, RMSE: {:.5f}, R2: {:.5f}, EV: {:.5f}""".format(train, test, rmse, r2, ev))

import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

data = p.read_csv('housing.csv')

# TRATANDO NaN
data = data.dropna(axis=0, how='any')
#data = data.fillna(data.mean())


# PRE-PROCESSING
target = data['median_house_value']
del data['median_house_value']

aux = p.get_dummies(data['ocean_proximity'], prefix='ocean_proximity')
del data['ocean_proximity']
data = p.concat([data,aux], axis=1)

#del data['longitude']
#del data['latitude']

# CALCULO DE DISTÂNCIA
data['dist_LA'] = ((data['longitude']-(-118.14))**2+(data['latitude']-(-34.3))**2)**0.5
data['dist_SF'] = ((data['longitude']-(-122.24))**2+(data['latitude']-(-37.46))**2)**0.5
data['dist_SD'] = ((data['longitude']-(-117.09))**2+(data['latitude']-(-32.42))**2)**0.5

# LOG SQUARED
'''
sns.set_theme()
sns.distplot(data['median_income'])
plt.show()
data['median_income'] = np.log(data['median_income']+1)
sns.set_theme()
sns.distplot(data['median_income'])
plt.show()
'''

# BINNING
n_bins = 3
bins = np.linspace(np.min(data['median_income']),np.max(data['median_income']),n_bins)
labels = list(range(n_bins-1))
data['classes'] = p.cut(data['median_income'], bins=bins, labels=labels, include_lowest=True)

aux = p.get_dummies(data['classes'], prefix='classes')
del data['classes']
data = p.concat([data,aux], axis=1)


#scaler = preprocessing.MinMaxScaler().fit(data)
#data = scaler.transform(data)

scores = {}
scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error',
		'neg_mean_squared_error','r2','explained_variance']

'''
engine = LinearRegression(normalize=True)
scores['LINEAR_REG'] = cross_validate(engine, data, target, scoring=scoring)


engine = Lasso(max_iter=500)
scores['LASSO'] = cross_validate(engine, data, target, scoring=scoring)

engine = Ridge()
scores['RIDGE'] = cross_validate(engine, data, target, scoring=scoring)
'''

engine = ElasticNet(alpha=2.30, l1_ratio=0.9545)
scores['ELASTICNET'] = cross_validate(engine, data, target, scoring=scoring)

'''
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
'''

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

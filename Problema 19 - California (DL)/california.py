import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical 

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
scaler = preprocessing.MinMaxScaler().fit(data)
data = scaler.transform(data)

# PREPARE OUTPUT
target = target.values.reshape(-1, 1)
scaler = preprocessing.MinMaxScaler().fit(target)
target  = scaler.transform(target)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.50)

# DESCREVENDO O MODELO
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=13))
#model.add(Dropout(.1))
#model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(X_train, y_train, epochs=100)

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('MSE on training data: {}'.format(scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('MSE on test data: {}'.format(scores2[1]))  

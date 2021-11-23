import pandas as p
import numpy as np

from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical 

# LOADING DATA
data = load_digits()

# PRE-PROCESSING
scaler = preprocessing.MinMaxScaler().fit(data.data)
data.data = scaler.transform(data.data)

# PREPARE OUTPUT
encoder = preprocessing.OneHotEncoder()
data.target = encoder.fit_transform(data.target.reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.50)

# DESCREVENDO O MODELO
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=64))
model.add(Dropout(.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))  

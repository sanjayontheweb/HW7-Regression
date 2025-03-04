"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import pandas as pd
from regression.logreg import LogisticRegressor
from sklearn.model_selection import train_test_split
# (you will probably need to import more things here)

def train_test_generate():
	#Load data from data/nsclc.csv
	data = pd.read_csv('data/nsclc.csv')

	#Split data into X and y, choose numerical features
	X = data.drop('NSCLC', axis=1)[['Penicillin V Potassium 500 MG','Computed tomography of chest and abdomen','Plain chest X-ray (procedure)','Diastolic Blood Pressure','Body Mass Index','Body Weight','Body Height']]
	y = data['NSCLC']

	#Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	return X_train, X_test, y_train, y_test

def test_prediction():
	X_train, X_test, y_train, y_test =  train_test_generate()
	model = LogisticRegressor(num_feats=X_train.shape[1])
	pred = model.make_prediction(X_train)
	assert pred.shape[0] == y_train.shape[0]
	#assert all of pred is between 0 and 1
	assert np.all(pred >= 0) and np.all(pred <= 1)


def test_loss_function():
	X_train, X_test, y_train, y_test =  train_test_generate()
	model = LogisticRegressor(num_feats=X_train.shape[1])
	pred = model.make_prediction(X_train)
	loss = model.loss_function(y_train, pred)
	#assert loss is a scalar
	assert isinstance(loss, float)
	#assert loss is not negative
	assert loss >= 0
	#True loss
	assert np.isclose(loss, -np.mean(y_train * np.log(pred) + (1 - y_train) * np.log(1 - pred)))

	

def test_gradient():
	X_train, X_test, y_train, y_test =  train_test_generate()
	model = LogisticRegressor(num_feats=X_train.shape[1])
	pred = model.make_prediction(X_train)
	grad = model.calculate_gradient(y_train, X_train)
	#assert grad is a numpy array
	assert isinstance(grad, np.ndarray)
	#assert grad has the same shape as the weights minus 1 (bias term)
	assert grad.shape[0] == model.W.shape[0] - 1
	#True gradient
	assert np.allclose(grad, np.dot(X_train.T, pred - y_train) / y_train.shape[0])

def test_training():
	X_train, X_test, y_train, y_test =  train_test_generate()
	model = LogisticRegressor(num_feats=X_train.shape[1])
	init_weights = model.W.copy()
	model.train_model(X_train, y_train, X_train, y_train)
	#assert weights have changed
	assert not np.allclose(init_weights, model.W)
	#assert loss history is being recorded
	assert len(model.loss_hist_train) > 0
	#assert loss is decreasing
	assert model.loss_hist_train[0] > model.loss_hist_train[-1]
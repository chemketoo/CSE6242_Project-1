#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:50:39 2019

@author: swuser
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeCV,LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("2016_properties_60000v3_train.csv").dropna()
test = pd.read_csv("2016_properties_60000v3_test.csv").dropna()

train_y = train.iloc[:,7]
test_y = test.iloc[:,7]
train_xs = train.iloc[:,1:7]
test_xs = test.iloc[:,1:7]

train_xl = train.drop(["parcelid","taxvaluedollarcnt"],axis=1)
test_xl = test.drop(["parcelid","taxvaluedollarcnt"],axis=1)

score = []
error = []
# linear regression
lin = LinearRegression(n_jobs=-1).fit(train_xs,train_y)
test_ys_pred = lin.predict(test_xs)
score.append(lin.score(test_xs,test_y))
error.append(mean_squared_error(test_y,test_ys_pred))
lin = LinearRegression(n_jobs=-1).fit(train_xl,train_y)
test_yl_pred = lin.predict(test_xl)
score.append(lin.score(test_xl,test_y))
error.append(mean_squared_error(test_y,test_yl_pred))

#  Ridge
ridge = RidgeCV(cv=5).fit(train_xs,train_y)
score.append(ridge.score(test_xs,test_y))
test_ys_pred = ridge.predict(test_xs)
error.append(mean_squared_error(test_y,test_ys_pred))

ridge = RidgeCV(cv=5).fit(train_xl,train_y)
score.append(ridge.score(test_xl,test_y))
test_yl_pred = ridge.predict(test_xl)
error.append(mean_squared_error(test_y,test_yl_pred))

# Lasso
lasso = LassoCV(cv=5).fit(train_xs,train_y)
score.append(lasso.score(test_xs,test_y))
test_ys_pred = lasso.predict(test_xs)
error.append(mean_squared_error(test_y,test_ys_pred))


lasso = LassoCV(cv=5).fit(train_xl,train_y)
score.append(lasso.score(test_xl,test_y))
test_yl_pred = lasso.predict(test_xl)
error.append(mean_squared_error(test_y,test_yl_pred))

# SVR
scaler = StandardScaler()
scaler.fit(train_xs)
train_xs_std = scaler.transform(train_xs)
test_xs_std = scaler.transform(test_xs)
scaler.fit(train_xl)
train_xl_std = scaler.transform(train_xl)
test_xl_std = scaler.transform(test_xl)

svrrbf = GridSearchCV(SVR(kernel='rbf'),cv=5,param_grid={"C":[0.1,1,10,100],
                      "gamma":[0.1,0.5,1,5,10]},n_jobs=-1)
svrrbf.fit(train_xs,train_y)
score.append(svrrbf.score(test_xs,test_y))
test_ys_pred = svrrbf.predict(test_xs)
error.append(mean_squared_error(test_y,test_ys_pred))
#SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1,
  #kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
svrrbf = GridSearchCV(SVR(kernel='rbf'),cv=5,param_grid={"C":[0.1,1,10,100],
                      "gamma":[0.1,0.5,1,5,10]},n_jobs=-1)
svrrbf.fit(train_xs_std,train_y)
score.append(svrrbf.score(test_xs_std,test_y))
test_ys_pred = svrrbf.predict(test_xs_std)
error.append(mean_squared_error(test_y,test_ys_pred))


svrrbf = GridSearchCV(SVR(kernel='rbf'),cv=5,param_grid={"C":[0.1,1,10,100],
                      "gamma":[0.1,0.5,1,5,10]},n_jobs=-1)
svrrbf.fit(train_xl_std,train_y)
score.append(svrrbf.score(test_xl_std,test_y))
test_yl_pred = svrrbf.predict(test_xl_std)
error.append(mean_squared_error(test_y,test_yl_pred))
#SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1,
 # kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# SVR linear
svrrbf = GridSearchCV(SVR(kernel='linear'),cv=5,param_grid={"C":[0.1,1,10,100]},n_jobs=-1)
svrrbf.fit(train_xs_std,train_y)
score.append(svrrbf.score(test_xs_std,test_y))
test_ys_pred = svrrbf.predict(test_xs_std)
error.append(mean_squared_error(test_y,test_ys_pred))
#C=100

svrrbf = GridSearchCV(SVR(kernel='linear'),cv=5,param_grid={"C":[0.1,1,10,100]},n_jobs=-1)
svrrbf.fit(train_xl_std,train_y)
score.append(svrrbf.score(test_xl_std,test_y))
test_yl_pred = svrrbf.predict(test_xl_std)
error.append(mean_squared_error(test_y,test_yl_pred))
#C=100

# SVR ori
svrrbf = GridSearchCV(SVR(kernel='linear'),cv=5,param_grid={"C":[0.1,1,10,100]},n_jobs=-1)
svrrbf.fit(train_xs,train_y)
score.append(svrrbf.score(test_xs,test_y))
test_ys_pred = svrrbf.predict(test_xs)
error.append(mean_squared_error(test_y,test_ys_pred))
#C=100

svrrbf = GridSearchCV(SVR(kernel='linear'),cv=5,param_grid={"C":[0.1,1,10,100]},n_jobs=-1)
svrrbf.fit(train_xl,train_y)
score.append(svrrbf.score(test_xl,test_y))
test_yl_pred = svrrbf.predict(test_xl)
error.append(mean_squared_error(test_y,test_yl_pred))
#C=100
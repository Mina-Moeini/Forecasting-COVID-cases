# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:23:54 2022

@author: Asus
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
#---------------------(Read Dataset)-------------------------------------------
df=pd.read_csv("C:/Users/Asus/Desktop/Covid-dataset/dataset-fr.csv")
df['date']=pd.to_datetime(df['date']).dt.date
df.set_index('date',inplace=True)
#------------------(Split Dataset)---------------------------------------------
TRAIN_SPLIT=int(len(df)*0.8)
lag=21 #for active cases (conf_j1)
end_train= (TRAIN_SPLIT - lag)
train_size_lc=[22,100,200,300,400,end_train]
#cross validation for time series
tscv = TimeSeriesSplit(n_splits = 3 )

# define lists to collect scores
train_scores_lr, test_scores_lr, train_scores_svr, test_scores_svr,train_scores_RF, test_scores_RF= list(), list(),list(), list(),list(), list()

#------------------(convert to supervised)-------------------------------------
#------------------(Multivariate_data function)--------------------------------
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)
#------------------------(set variable and convert to supervised)--------------
#the function just accept the array type
df=np.array(df)
#----------------------------(Scale Data)--------------------------------------
scaler =MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(df)
past_history =lag
#number of step for prediction
future_target = 1
STEP = 1



x_test_multi, y_test_multi = multivariate_data(df, df[:,12],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=False)

x_test_2d=x_test_multi.reshape(x_test_multi.shape[0],-1)
y_test_1d=y_test_multi.ravel()

#------------------(define models and fit)-------------------------------------
#---------------------(LR)-----------------------------------------------------

for i in train_size_lc:
    x_train_multi, y_train_multi= multivariate_data(df, df[:,12], 0,
                                                   i, past_history,
                                                   future_target, STEP,
                                                   single_step=False)
    
    x_train_2d=x_train_multi.reshape(x_train_multi.shape[0],(x_train_multi.shape[1]*x_train_multi.shape[2]))
    y_train_1d=y_train_multi.ravel()
    
    LR_model=LinearRegression()
    LR_model.fit(x_train_2d,y_train_1d)
    y_predict_train=LR_model.predict(x_train_2d)
    y_predict_test=LR_model.predict(x_test_2d)
    mse_LR_train= mean_squared_error(y_train_1d, y_predict_train)
    train_scores_lr.append(mse_LR_train)
    mse_LR_test= mean_squared_error(y_test_1d, y_predict_test)
    test_scores_lr.append(mse_LR_test)
    
# plot of train and test scores
plt.plot(train_size_lc, train_scores_lr, '-o', label='Train')
plt.plot(train_size_lc, test_scores_lr, '-o', label='Test')
plt.legend()
plt.show()    

#---------------------(SVR)----------------------------------------------------
#Building the support vector regression model
#we will choose the best parameter for SVR model
kernel = ['poly','sigmoid','rbf']
C = [0.01,0.1,1,10]
gamma = [0.01,0.1,1]
epsilon = [0.01,0.1,1]
shrinking = [True,False]
svr_grid = {'kernel':kernel,'C':C,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}
SVR = SVR()
svr_search=GridSearchCV(SVR,svr_grid,cv=tscv)
#svr_search = RandomizedSearchCV(SVR,svr_grid,cv=tscv)
svr_search.fit(x_train_2d,y_train_1d)
#svr_search.best_params_
svr_confirmed=svr_search.best_estimator_
print(svr_confirmed)

for i in train_size_lc:
    x_train_multi_svr, y_train_multi_svr= multivariate_data(df, df[:,12], 0,
                                                   i, past_history,
                                                   future_target, STEP,
                                                   single_step=False)
    
    x_train_2d_svr=x_train_multi_svr.reshape(x_train_multi_svr.shape[0],(x_train_multi_svr.shape[1]*x_train_multi_svr.shape[2]))
    y_train_1d_svr=y_train_multi_svr.ravel()
    
    svr_confirmed.fit(x_train_2d_svr,y_train_1d_svr)
    y_predict_train_svr=svr_confirmed.predict(x_train_2d_svr)
    y_predict_test_svr=svr_confirmed.predict(x_test_2d)
    mse_svr_train= mean_squared_error(y_train_1d_svr, y_predict_train_svr)
    train_scores_svr.append(mse_svr_train)
    mse_svr_test= mean_squared_error(y_test_1d, y_predict_test_svr)
    test_scores_svr.append(mse_svr_test)
  
# plot of train and test scores
plt.plot(train_size_lc, train_scores_svr, '-o', label='Train')
plt.plot(train_size_lc, test_scores_svr, '-o', label='Test')
plt.legend()
plt.show()   

#------------------------------------------------------------------------------   
"""
RF_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [20,50,100,200,1000]
}

RF = RandomForestRegressor()
#cv=the number of folds to use for cross validation
RF_search = RandomizedSearchCV(RF,RF_grid,cv=tscv)
RF_search.fit(x_train_2d,y_train_1d)
RF_confirmed = RF_search.best_estimator_    
print(RF_confirmed)

for i in train_size_lc:
    x_train_multi_rf, y_train_multi_rf= multivariate_data(df, df[:,12], 0,
                                                   i, past_history,
                                                   future_target, STEP,
                                                   single_step=False)
    x_train_2d_rf=x_train_multi_rf.reshape(x_train_multi_rf.shape[0],(x_train_multi_rf.shape[1]*x_train_multi_rf.shape[2]))
    y_train_1d_rf=y_train_multi_rf.ravel()
   
    RF_confirmed.fit(x_train_2d_rf,y_train_1d_rf)
    y_predict_train_RF=RF_confirmed.predict(x_train_2d_rf)
    y_predict_test_RF=RF_confirmed.predict(x_test_2d)
    mse_RF_train= mean_squared_error(y_train_1d_rf, y_predict_train_RF)
    train_scores_RF.append(mse_RF_train)
    mse_RF_test= mean_squared_error(y_test_1d, y_predict_test_RF)
    test_scores_RF.append(mse_RF_test)
  
# plot of train and test scores
plt.plot(train_size_lc, train_scores_RF, '-o', label='Train')
plt.plot(train_size_lc, test_scores_RF, '-o', label='Test')
plt.legend()
plt.show()  
"""
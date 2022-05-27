from sklearn.neural_network import MLPRegressor
import math
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.feature_selection import r_regression
import matplotlib.pyplot as plt
from functools import partial
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
from skopt.plots import plot_convergence
import time

# import data
data1 = pd.read_csv("Combined_Data1.csv")
data2 = pd.read_csv("Combined_Data2.csv")

X1 = data1.iloc[:,3:]
X2 = data2.iloc[:,7:]

y_list1 = [data1.iloc[:,0],data1.iloc[:,1]]

y_list2 = []
for i in range(6):
    y_list2.append(data2.iloc[:,i+1])
    
#Data Preprocessing
def data_preprocess(dataset,target): 
    # Scaled data
    min_max_scaler = MinMaxScaler(feature_range = (0,1))
    np_scaled = min_max_scaler.fit_transform(dataset)
    X = pd.DataFrame(np_scaled)
    
    target_edit = pd.Series(target).values
    target_edit = target_edit.reshape(-1,1)
    np_scaled = min_max_scaler.fit_transform(target_edit)
    Y = pd.DataFrame(np_scaled)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
      
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
        
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    return X_train, y_train, X_test, y_test
  
# Parameter space including hidden layer size
param_space = [
        Real(0.0001,10000,prior="log-uniform",name="alpha"),
        Categorical(['tanh', 'relu', 'identity', 'logistic'],name="activation"),
        Categorical(['adam','lbfgs'],name="solver"),
        Categorical(['constant','adaptive'],name="learning_rate"),
        Integer(5,500,name='hidden_layer_sizes')]
param_names = ["alpha","activation","solver","learning_rate","hidden_layer_sizes"]

from statistics import stdev

# Function to optimise
def optimise(params, param_names, x, y):
    params = dict(zip(param_names,params))
    
    # build model
    mlp = MLPRegressor(max_iter=500,early_stopping=True,**params)
    kf = KFold(n_splits=5,shuffle=True)
    fold_scores = []
    
    for train_index, test_index in kf.split(x):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        mlp.fit(X_train.values,y_train.values)
        pred = mlp.predict(X_test.values)
        
        metric1 = -1*r2_score(y_test,pred)
        metric2 = mean_squared_error(y_test,pred)
        metric3 = -1*(np.corrcoef(y_test,pred))[0,1]
        fold_scores.append(metric3)
        
    # Check for and remove any NaN values
    safe_scores = [x for x in fold_scores if math.isnan(x) == False]
    # return average score
    av_score = np.mean(safe_scores)
    stdv = stdev(safe_scores)
    print("%0.3f (+/-%0.03f) for %r" % (av_score, stdv, params))

    return av_score, stdv
  
# Optimised K-fold score
def kf_optimised_mlp(X,y):
    start = time.time()
    print("MLP: ", y.name)
    
    optimisation_function = partial(optimise,param_names=param_names,x=X,y=y)
    
    result = gp_minimize(
        optimisation_function,
        dimensions=param_space,
        n_calls=40,
        n_initial_points=15,
        verbose=2,
        n_jobs = -1,
        acq_optimizer = "lbfgs")
        
    params = dict(zip(param_names,result.x))
    
    score = -1*result.fun
    print("Best score: ", score)
    
    plot_convergence(result)
    
    end = time.time()
    time_elapsed = end - start
    print("Time elapsed: ", time_elapsed)
    
    return score, params, time_elapsed, result

# Run hyperparameter optimisation for 5 repetitions for all phenotypes
# Contstruct list of (x,y) pairs
data_list = []
for y in y_list1:
    data_list.append((X1,y))
for y in y_list2:
    data_list.append((X2,y))

full_results = []
results = []
times = []

for (X,y) in data_list:
    trait = y.name
    print(trait)
    # run 5 repetitions for each trait
    for i in range(5):
        score, params, t, full_result = kf_optimised_mlp(X,y)
        print("Score for",trait,":",score)
        results.append((trait,score,params,t))
        times.append(t)
        full_results.append(full_result)
    plt.show()
        
print(results)

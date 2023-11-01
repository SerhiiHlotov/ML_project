'''Module with functions that generate 7 Machine Learning models'''
# importing handmade represantation functions and metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# importing  classes of ML algorithms
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel



def Random_Forest_function (x_train, y_train, x_valid, y_valid):
    '''input: x_train, y_train, x_valid, y_valid
    X - numpyarray
    y - value
    output: mean squared_error and  Model
    '''
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42,  ) 
    rf_regressor.fit(x_train, y_train) 
    predictions = rf_regressor.predict(x_valid)

    mse = mean_squared_error(predictions, y_valid)
    mae = mean_absolute_error(predictions, y_valid)
    return (mae, rf_regressor)

def MLP_Regressor_function (x_train, y_train, x_valid, y_valid):
    '''input: x_train, y_train, x_valid, y_valid
    X - numpyarray
    y - value
    output: mean squared_error and Model
    '''
    mlp_reg =  MLPRegressor(hidden_layer_sizes=(100),  activation='identity', solver='adam', max_iter=1000, random_state=42,)
    mlp_reg.fit(x_train, y_train)
    predictions = mlp_reg.predict(x_valid)

    mse = mean_squared_error(predictions, y_valid)
    mae = mean_absolute_error(predictions, y_valid)
    return (mae, mlp_reg)

def Linear_Regressor_function(x_train, y_train, x_valid, y_valid):
    '''input: x_train, y_train, x_valid, y_valid
    X - numpyarray
    y - value
    output: mean squared_error 
    and  Model
    '''
    lr_reg = LinearRegression()
    lr_reg.fit(x_train, y_train)
    predictions = lr_reg.predict(x_valid)

    mse = mean_squared_error(predictions, y_valid)
    mae = mean_absolute_error(predictions, y_valid)
    return (mae, lr_reg)

def KNeighbors_Regressor_fucntion(x_train, y_train, x_valid, y_valid):
    '''input: x_train, y_train, x_valid, y_valid
    X - numpyarray
    y - value
    output: mean squared_error 
    and Model
    '''
    KN_reg = KNeighborsRegressor(n_neighbors=10)
    KN_reg.fit(x_train, y_train)
    predictions = KN_reg.predict(x_valid)

    mse = mean_squared_error(predictions, y_valid)
    mae = mean_absolute_error(predictions, y_valid)
    return (mae, KN_reg)

def SVR_Regressor_fucntion(x_train, y_train, x_valid, y_valid):
    '''input: x_train, y_train, x_valid, y_valid
    X - numpyarray
    y - value
    output: mean squared_error 
    and Model
    '''
    svr_reg = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_reg.fit(x_train, y_train)
    predictions = svr_reg.predict(x_valid)

    mse = mean_squared_error(predictions, y_valid)
    mae = mean_absolute_error(predictions, y_valid)
    return (mae, svr_reg)

def GaussianProcess_Regressor_function (x_train, y_train, x_valid, y_valid):
    '''input: x_train, y_train, x_valid, y_valid
    X - numpyarray
    y - value
    output: mean squared_error 
    and Model
    '''
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr.fit(x_train, y_train)
    
    pred = gpr.predict(x_valid)
    mse = mean_squared_error(pred, y_valid)
    mae = mean_absolute_error(pred, y_valid)
    return (mae, gpr)


# for implementation of tanimoto class

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import positive, print_summary
from gpflow.utilities.ops import broadcasting_elementwise
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

#importing usefull utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict


# importing handmade functions
from functions.fps_gen_and_dim_red_functions import *
from functions.MLmodel_functions import *


def find_min_in_dict(my_dict):
    min_key = min(my_dict, key=lambda k: my_dict[k])
    min_value = my_dict[min_key]
    return (min_key, min_value)


def generate_the_best_model(x_train, y_train, x_test, y_test ):
    
    y_train = np.array(y_train.to_list())
    y_test = np.array(y_test.to_list())
    #list for iteration
    '''TSNE_dim_red, need to be inserted on larger dataset, '''
    fps_repr_func_list = [ RDKIT_fps_repr, Morgan_ECFP_fps_repr, E3FP_fps_repr, PLEC_E3FP_fps_repr, PLEC_E3FP_fps_repr ]
    dim_red_func_list = [ no_dim_red, PCA_dim_red,  MDS_dim_red]
    dim_sizes_list = [2,10]
    MLmodel_func_list = [MLP_Regressor_function,Random_Forest_function,  Linear_Regressor_function, KNeighbors_Regressor_fucntion, SVR_Regressor_fucntion, GaussianProcess_Regressor_function]
    cycles_count = 0
    mae_dict ={}
    model_dict = {}
    for fps_repr_func in fps_repr_func_list:
        #iterating through list of fps respresenttions fucntions 
        fps_x_train = fps_repr_func(x_train)
        fps_x_valid = fps_repr_func(x_test)
        
        for dim_red_func in dim_red_func_list:
            for dim_size in dim_sizes_list:
                red_fps_x_train = dim_red_func(fps_x_train, dim_size)
                red_fps_x_valid = dim_red_func(fps_x_valid, dim_size)
                


                for MLmodel_func  in  MLmodel_func_list:
                    
                    x = MLmodel_func(red_fps_x_train, y_train, red_fps_x_valid, y_test)
                    model_information_tuple = fps_repr_func.__name__ , dim_red_func.__name__, dim_size, MLmodel_func.__name__
                    
                    mae_dict[model_information_tuple] = x[0]
                    model_dict[model_information_tuple]= x[1]


                    cycles_count +=1
                    print('cycle_count=', cycles_count)
                    
    return mae_dict, model_dict
    # return {'mae_dict':mae_dict,  'model_dict': model_dict }


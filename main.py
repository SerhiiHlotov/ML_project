
# importing core functions and built in python modules
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import time 
import os
from rdkit import Chem
from rdkit.Chem import AllChem

start_time = time.time()
#importing usefull utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# importing handmade functions
from functions.fps_gen_and_dim_red_functions import *
from functions.MLmodel_functions import *
from functions.best_model_function import find_min_in_dict


# hiding warnings  or we can delete this two lines and run the line (python -Wignore your_script.py)
import warnings
warnings.filterwarnings("ignore")



# creating the way to save logs about each program execution
log_folder = "info_logs"  
log_filename = "log"  
max_log_files = 10 
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

existing_log_files = [file for file in os.listdir(log_folder) if file.startswith(log_filename)]

if len(existing_log_files) >= max_log_files:
    oldest_log_file = os.path.join(log_folder, min(existing_log_files))
    os.remove(oldest_log_file)

log_file_path = os.path.join(log_folder, f"{log_filename}_{len(existing_log_files) + 1}.log")


# creating logger object
logger_obj = logging.getLogger('my_logger_obj')
handler = logging.FileHandler(filename=log_file_path, mode='w', )
formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s   - %(message)s - [%(filename)s]')
handler.setLevel(level=logging.DEBUG)
handler.setFormatter(fmt=formater)
logger_obj.addHandler(handler)

# creating parser for getting input from command line
parser = argparse.ArgumentParser(description='ML prediction based on smiles. ', prog='MLPredict')
parser.add_argument( 'input_file', metavar='CSV file', type=str, help='Path to the full table with values and SMILES ')
parser.add_argument('-n','--number_of_top_molecules', type=int, default=150,  help='Provide the number of rows that will be a sample of the full table')
parser.add_argument('-o', '--output_file', type=str, default='output.csv', help='Path to the output file where the predicted values will be stored  in form of table (default: output.txt)')
parser.add_argument('-p', '--protein_file_name', type=str, default='6nm0.pdb', help='Path to the protein.pdb file for PLEC function default="6nm0.pdb")')
parser.add_argument('-m' , '--model_file_name', type=str, default='chosen_trained_model.pkl',  help='the name of the file where the chosen model will be stored in a model.pkl format ')
parser.add_argument('-c' , '--columns_name', nargs='+',  type=str, default=['smiles', 'measured log solubility in mols per litre'], help='Names of two columns(first with smiles, second with values) to process as a list of strings, default=["smiles", "measured log solubility in mols per litre"]')
args = parser.parse_args()

input_file = args.input_file
number_of_top_molecules = args.number_of_top_molecules
output_file = args.output_file
columns_name = args.columns_name
model_filename = args.model_file_name

logger_obj.info('PROGRAM BEGINS')
logger_obj.info(f'argument from the command line: {args}')
logger_obj.info('Preprocess begins')
#processing the input data


full_table = pd.read_csv(input_file)
full_table = full_table[columns_name]

#renaming the columns to use them easily
full_table.columns.values[1] = 'values'
full_table.columns.values[0] = 'SMILES'
small_table  = full_table.sample(n=number_of_top_molecules)

# creating the table without chosen rows
full_table_without_small = full_table.drop(small_table.index).reset_index(drop=True)
small_table.reset_index(drop=True)

# preprocessing and dividing the set
X= small_table['SMILES']
y = small_table['values']
X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logger_obj.info('Preprocess finished')
# the core function 
def generate_the_best_model(x_train, y_train, x_test, y_test ):
    
    y_train = np.array(y_train.to_list())
    y_test = np.array(y_test.to_list())
    
    #lists for iteration in
    '''TSNE_dim_red, 100 need to be inserted on larger dataset, E3FP_fps_repr, PLEC_E3FP_fps_repr '''
    fps_repr_func_list = [  RDKIT_fps_repr, Morgan_ECFP_fps_repr,  PLEC_fps_repr, E3FP_fps_repr]
    dim_red_func_list = [ no_dim_red, PCA_dim_red,  MDS_dim_red]
    dim_sizes_list = [2,10]
    MLmodel_func_list = [MLP_Regressor_function,Random_Forest_function,  Linear_Regressor_function, KNeighbors_Regressor_fucntion, SVR_Regressor_fucntion, GaussianProcess_Regressor_function]
    
    cycles_count = 0
    model_dict = {}

    for fps_repr_func in fps_repr_func_list:
        #iterating through list of fps respresenttions fucntions 
        logger_obj.info(f'{fps_repr_func.__name__}  generate the fingerprints ')
        try:
            fps_x_train = fps_repr_func(x_train)
            fps_x_valid = fps_repr_func(x_test)
        except Exception as e: 
            logging.error(f'This is the error message {e} for {fps_repr_func.__name__}')
            logging.info('skipping this representation')
            continue
        
        for dim_red_func in dim_red_func_list:
            for dim_size in dim_sizes_list:
            
                red_fps_x_train = dim_red_func(fps_x_train, dim_size)
                red_fps_x_valid = dim_red_func(fps_x_valid, dim_size)
                


                for MLmodel_func  in  MLmodel_func_list:

                    cycles_count +=1
                    x = MLmodel_func(red_fps_x_train, y_train, red_fps_x_valid, y_test)

                    model_functions_tuple = x[1], dim_red_func, fps_repr_func,  dim_size
                    
                    model_dict[model_functions_tuple]= x[0]

                    # it's made for controling the fucntion
                    model_information_tuple  = MLmodel_func.__name__, dim_red_func.__name__,  fps_repr_func.__name__ ,  dim_size
                    logger_obj.debug(f'Status of the longest function: {model_information_tuple}')
                    
    return  model_dict


# searching for the best architecture combination 
model_dict = generate_the_best_model(X_train, y_train, X_test, y_test)
best_model_tuple= find_min_in_dict(model_dict)
    
logger_obj.info(f'dict :{model_dict}')
logger_obj.info(f'The total amount of generated models:{len(model_dict)}')
logger_obj.info('The end of the function')

chosen_model = best_model_tuple[0][0]
chosen_dim_red = best_model_tuple[0][1]
chosen_repr_func = best_model_tuple[0][2]
dim_chosen_size = best_model_tuple[0][3]

logger_obj.info(f'Best model combinations:{ chosen_model, chosen_dim_red.__name__, chosen_repr_func.__name__, dim_chosen_size}')

#Saving the model to a file 

with open(model_filename, 'wb') as file:
    pickle.dump(chosen_model, file)

logger_obj.info(f'Make prediction on the full data')
if full_table_without_small.shape[0]>10000:
    full_table_without_small = full_table_without_small.sample(n=int(full_table_without_small.shape[0]/2))
# apllying the chosen architecture that consist of model, reprepresanta_function, dimensionality_reduction and dimension size to full data
X_2= full_table_without_small['SMILES']
y_2 = full_table_without_small['values']
y_2 = np.array(y_2.to_list())

predictions_full_table_without_small = chosen_model.predict(chosen_dim_red (chosen_repr_func(X_2), dim_chosen_size))

#chosing the molecules with highest score
top_new_small_table = pd.DataFrame({'SMILES':full_table_without_small['SMILES'], "values": predictions_full_table_without_small})
top_new_small_table = top_new_small_table.sort_values(by='values', ascending=False)
top_new_small_table = top_new_small_table.head(number_of_top_molecules)
top_new_small_table.reset_index(drop=True)
top_new_small_table.to_csv(output_file, index= False)

logger_obj.info('Program finished the job')



end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time/ 60


# Log the elapsed time
logger_obj.info(f"Program execution time: {elapsed_time} seconds")
logger_obj.info(f'Program execution time: {elapsed_time_minutes} minutes ')
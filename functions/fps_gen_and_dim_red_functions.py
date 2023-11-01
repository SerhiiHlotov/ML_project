'''Module with funtctions for generating  different fingerprints (5 represantations)
and functions for dimensionality reduction (3 method and 1 no reduction) '''
# importing core modules

import numpy as np
from sklearn.model_selection import train_test_split

# imporing RDKIT
from rdkit import Chem
from rdkit.Chem import AllChem

# importing modules for generating fingerprints

from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
import e3fp 
from e3fp.pipeline import fprints_from_mol,  fprints_from_smiles, confs_from_smiles
from e3fp.conformer.generate import generate_conformers
import oddt
from oddt.fingerprints import PLEC

#importing dimensionality reduction functions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS


import logging
logger_obj = logging.getLogger('my_logger_obj')

# fingerptins represantation functions


def Morgan_ECFP_fps_repr(smiles_list):
    """input : list of rdkit molecules
    output : numpy array of fingerprints"""
    fps = [AllChem.GetMorganFingerprintAsBitVect( Chem.MolFromSmiles(smile), radius=3 , nBits=2048)for smile in smiles_list]
    fps_list = [list(fp.ToBitString()) for fp in fps ]
    fps_array = np.array(fps_list, dtype=int)
    return fps_array

def RDKIT_fps_repr(smiles_list):
    """input : list of rdkit molecules
    output : numpy array of fingerprints"""
    rdkit_fingerprints_list= []
    for smile in smiles_list:
        fps = AllChem.RDKFingerprint(Chem.MolFromSmiles(smile), fpSize=2048, maxPath=7, minPath =1)
        fps_bit_string = list(fps.ToBitString())
        rdkit_fingerprints_list.append(fps_bit_string)
        
      
    return np.array(rdkit_fingerprints_list, dtype=int)
    
def E3FP_fps_repr(smiles_list):
    """input: list of SMILES 
    output: numpy array of  fingerprints"""

    

    fps_list = []
    prob = 0
    for smile in smiles_list:
        
        try : 
            thr_dim_fp = fprints_from_smiles(smiles=smile, name=smile, confgen_params={'num_conf': 1, 'first' :True}, fprint_params={'bits': 2048},)
        # extracting fingerpritnt object from the list
            fps_object = thr_dim_fp[0]
        # function returns FINGEPRING object, so we need to extract it as bit string and transfomr it into bit vector
            fps_bit_string = list(fps_object.to_bitstring())
        
            fps_list.append(fps_bit_string)
        except Exception as e:
            prob +=1
            logger_obj.info(f'For this smile {smile} 3D fingerprint wasn\'t generated problem raised : {e}')
            fps_bit_string = [0] * 2048
            fps_list.append(fps_bit_string)
            logger_obj.info(f'failed 3D fingerprint was replaced with empty-zeroes bitstring. {prob} problems were enountered ')

        
    fps_array = np.array(fps_list, dtype=int)

    return fps_array 

def PLEC_fps_repr(smiles_list, pdb_file_name='6nm0.pdb'):
    """input : list of mol objects, string name of file
    output : numpy array"""
    protein = next(oddt.toolkit.readfile('pdb', pdb_file_name))
    protein.protein =True
    protein.addh(only_polar=True)
    array_list = []
    for smile in smiles_list:
        ligand = oddt.toolkit.Molecule(Chem.MolFromSmiles(smile))
        ligand.addh(only_polar=True)
        plec_result = PLEC(ligand=ligand, protein=protein, sparse=False, size=2048)
        plec_result[plec_result != 1] = 0
        array_list.append(plec_result)
        
    array_list = np.array(np.vstack(array_list), dtype=int)
    return array_list   

def PLEC_E3FP_fps_repr(smiles_list, pdb_file_name='6nm0.pdb'):

    protein = next(oddt.toolkit.readfile('pdb', pdb_file_name))
    protein.protein =True
    protein.addh(only_polar=True)
    array_list = []
    for smile in smiles_list:
        #PLEC part
        ligand = oddt.toolkit.Molecule(Chem.MolFromSmiles(smile))
        ligand.addh(only_polar=True)
        plec_result = PLEC(ligand=ligand, protein=protein, sparse=False, size=2048)
        plec_result[plec_result != 1] = 0
        plec_part = plec_result

        #E3FP part
        thr_dim_fp = fprints_from_smiles(smiles=smile, name=smile, confgen_params={'num_conf': 1, 'first': True}, fprint_params={'bits': 2048} )
        # extracting fingerpritnt object from the list
        fps_object = thr_dim_fp[0]
        # function returns FINGEPRING object, so we need to extract it as bit string and transfomr it into bit vector
        fps_bit_string = list(fps_object.to_bitstring())
        e3fp_part = np.array(fps_bit_string)

        #Concateting string
        concat_string = np.hstack((plec_part, e3fp_part))
        array_list.append(concat_string)
    
    plec_e3fp_array = np.array(np.vstack(array_list), dtype=int)
    return plec_e3fp_array


# dimensionality reduction funstions 

def no_dim_red(fingerprint_list, dim_number):
    return fingerprint_list

def PCA_dim_red(fingerprint_list, dim_number):
    pca = PCA(n_components=dim_number)
    reduced_data = pca.fit_transform(fingerprint_list)
    return reduced_data

def TSNE_dim_red(fingerprint_list, dim_number):
    if dim_number >3:
        dim_number=2
    tsne = TSNE(n_components=dim_number)
    reduced_data = tsne.fit_transform(fingerprint_list)
    return reduced_data

def MDS_dim_red(fingerprint_list, dim_number):

    mds = MDS(n_components=dim_number)
    reduced_data = mds.fit_transform(fingerprint_list)
    return reduced_data 
        

# other functions

def need_to_be_checked(X, y):
    #X= small_table['SMILES']
    #y = small_table['R']
    X_train, X_temp,  y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=0)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, train_size=0.5, random_state=0)
    y_train = np.array(y_train.to_list())
    y_valid = np.array(y_valid.to_list())

#train_X_1 =  PCA_dim_red(Morgan_ECFP_fps_repr(smiles_list=X_train), 2)
#test_X_1=  PCA_dim_red(Morgan_ECFP_fps_repr(smiles_list=X_test), 2)
def transform_data(X_train, y_train, X_test, y_test):
    """
    Apply feature scaling to the data. Return the standardised train and
    test sets together with the scaler object for the target values.
    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

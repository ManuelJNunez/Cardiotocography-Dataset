# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import numpy as np
import pandas as pd
import pathlib as pl

# Función lectira de 
def read_data(file_name):
    data = pd.read_excel(pl.Path(__file__).parent / f"datos/{file_name}", skipfooter= 3, sheet_name= 2, header = 0)
    data.dropna(axis=0, thresh=10, inplace=True) 
    data.drop(columns=['FileName', 'Date', 'SegFile', 'b', 'e', 'LBE', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'DR'], inplace=True)
    data = np.asarray(data)
    X_data = data[:,:-2]
    y_fhr = data[:,-2]
    y_nsp = data[:,-1]
    return X_data, y_fhr, y_nsp

data_file = 'CTG.xls'

X_data, y_fhr, y_nsp = read_data(data_file)
print(X_data)
print(y_fhr)
print(y_nsp)
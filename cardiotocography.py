# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")
    
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

def plot_class_distribution(y_data, class_names):
    classes, count = np.unique(y_data, return_counts=True)
    plt.bar(np.arange(classes.size), count)
    plt.xticks(np.arange(classes.size), class_names)
    plt.xlabel("Clase a la que pertenece")
    plt.ylabel("Ocurrencias de la clase")
    plt.title(f"Balance de clases en el conjunto de datos con {classes.size} clases")
    plt.show()
    stop()

X_data, y_fhr, y_nsp = read_data(data_file)
plot_class_distribution(y_fhr, ('A', 'B', 'C', 'D', 'SH', 'AD', 'DE', 'LD', 'FS', 'SUSP'))
plot_class_distribution(y_nsp, ('Normal', 'Suspect', 'Pathologic'))
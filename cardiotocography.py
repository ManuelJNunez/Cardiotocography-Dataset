# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import pathlib as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")
    
# Función lectira de 
def read_data(file_name):
    data = pd.read_excel(pl.Path(__file__).parent / f"datos/{file_name}", skipfooter= 3, sheet_name= 2, header = 0)
    data.dropna(axis=0, thresh=10, inplace=True) 
    data.drop(columns=['FileName', 'Date', 'SegFile', 'b', 'e', 'LBE', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'DR'], inplace=True)
    feature_names = data.columns
    data = np.asarray(data)
    X_data = data[:,:-2]
    y_fhr = data[:,-2]
    y_nsp = data[:,-1]

    return X_data, y_fhr, y_nsp, feature_names

def plot_histogram(data, feature_names):
    nsubplots = 25
    fig, ax = plt.subplots(np.sqrt(nsubplots).astype(int), np.sqrt(nsubplots).astype(int), figsize=(15, 15), constrained_layout=True)
    ax = ax.reshape(nsubplots)

    for i in range(data.shape[1]):
        ax[i].hist(data[:,i], bins='doane')
        ax[i].set_title(feature_names[i])

    for i in range(data.shape[1], nsubplots):
        fig.delaxes(ax[i])

    plt.show()
    stop()

def plot_class_distribution(y_data, class_names):
    classes, count = np.unique(y_data, return_counts=True)
    plt.bar(np.arange(classes.size), count)
    plt.xticks(np.arange(classes.size), class_names)
    plt.xlabel("Clase a la que pertenece")
    plt.ylabel("Ocurrencias de la clase")
    plt.title(f"Balance de clases en el conjunto de datos con {classes.size} clases")
    plt.show()
    stop()

def plot_features_correlation(data, feature_names):
    corr = np.corrcoef(data, rowvar= False)

    fig, ax = plt.subplots(figsize=(11,11))

    # Representación la matriz de correlación y asignandole la colorbar
    cs = ax.imshow(corr)
    fig.colorbar(cs)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(feature_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, "%.2f" % corr[i, j], ha="center", va="center", color="w")
    
    ax.set_title('Heatmap de la correlación entre característiscas')
    fig.tight_layout()
    plt.show()
    stop()

data_file = 'CTG.xls'

X_data, y_fhr, y_nsp, feature_names = read_data(data_file)
plot_class_distribution(y_fhr, ('A', 'B', 'C', 'D', 'SH', 'AD', 'DE', 'LD', 'FS', 'SUSP'))
plot_class_distribution(y_nsp, ('Normal', 'Suspect', 'Pathologic'))
#plot_histogram(X_data, feature_names)
plot_features_correlation(X_data, feature_names[:-2])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_fhr, train_size=0.7, random_state=1)

# Ajuste y selección de parámetros SGDClassifier
pipe_linear_standarized = Pipeline(steps=[('scaler' ,StandardScaler()), ('sgd', SGDClassifier())])
param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'sgd__loss': ['log'],
    'sgd__penalty': ['l1', 'l2'],
    'sgd__n_jobs': [-1],
    'sgd__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
}

clf_linear_standarized = GridSearchCV(pipe_linear_standarized, param_grid, scoring='f1_weighted', n_jobs=-1)
clf_linear_standarized.fit(X_train, y_train)

print(f"Parámetros usados para ajustar el modelo de SGDClassifier: {clf_linear_standarized.best_params_}")

print("\nBondad del modelo de SGDClassifier con características estandarizadas para el modelo de 10 clases")
y_pred = clf_linear_standarized.predict(X_train)
print(f"Ein = {f1_score(y_train, y_pred, average='weighted')}")
y_pred = clf_linear_standarized.predict(X_test)
print(f"Etest = {f1_score(y_test, y_pred, average='weighted')}")
print(f"Ecv = {clf_linear_standarized.best_score_}")

# -*- coding: utf-8 -*-
# Universidad de Granada
# Grado en Ingeniería Informática
# Aprendizaje Automático
# Curso 2019/2020

# Proyecto Final: Ajuste del mejor modelo 
# Dataset: Cardiotocography (https://archive.ics.uci.edu/ml/datasets/cardiotocography)

# Manuel Jesús Núñez Ruiz
# Javier Rodríguez Rodríguez 

#############################
#####     LIBRERIAS     #####
#############################
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  SGDClassifier)
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             f1_score, plot_confusion_matrix, plot_roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

np.random.seed(1)

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")
    #pass
    
# Función para lectura de datos en formato .xls
# Los datos son separados adecuadamente en variables predictoras y clase
# Como los datos están clasificados de dos formas distintas, se generan dos vectores de clasificación "real", uno para cada forma
def read_data(file_name):
    data = pd.read_excel(pl.Path(__file__).parent / f"datos/{file_name}", skipfooter= 3, sheet_name= 2, header = 0)
    data.dropna(axis=0, thresh=10, inplace=True) 
    data.drop(columns=['FileName', 'Date', 'SegFile', 'b', 'e', 'LBE', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'DR'], inplace=True)
    feature_names = data.columns
    data = np.asarray(data)
    X = data[:,:-2]
    y = data[:,[-2, -1]]

    return X, y, feature_names

# Función auxiliar para la graficación de histogramas
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

# Función auxiliar para la graficación de diagramas de barra para comparar la proporción de clases
def plot_class_distribution(y_data, class_names):
    classes, count = np.unique(y_data, return_counts=True)
    plt.bar(np.arange(classes.size), count)
    plt.xticks(np.arange(classes.size), class_names)
    plt.xlabel("Clase a la que pertenece")
    plt.ylabel("Ocurrencias de la clase")
    plt.title(f"Balance de clases en el conjunto de datos con {classes.size} clases")
    plt.show()
    stop()

# Función auxiliar para la graficación de un mapa de calor que exprese la correlación entre las variables predictoras 
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

# Archivo de entrada
data_file = 'CTG.xls'

# Leemos los archivos
# y[0] = Clasificación FHR (10 clases)
# y[1] = Clasificación NSP (3 clases)
X, y, feature_names = read_data(data_file)

# Graficamos el balance de clase 
labels = [('A', 'B', 'C', 'D', 'SH', 'AD', 'DE', 'LD', 'FS', 'SUSP'), ('Normal', 'Suspect', 'Pathologic')]
plot_class_distribution(y[:, 0], labels[0])
plot_class_distribution(y[:, 1], labels[1])
# la distribución de valores de cada atributo
plot_histogram(X, feature_names)
# y la correlación entre atributos
plot_features_correlation(X, feature_names[:-2])

# Separamos el conjunto de datos en conjunto de train y test
# Como el número de instancias (2126) es suficientemente alto, separamos en 70% training y 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

# Evaluamos los distintos modelos para la clasificación FHR y NSP individualmente
for i, nombre in zip(np.arange(2), ('FHR (10 clases)', 'NSP (3 clases)')):

    # Clasificación utilizada
    print(f"\033[92;1;4mCLASIFICACIÓN {nombre}\033[0m")

    # Modelo 
    print(f"\033[94;1;1mAjuste utilizando Regresión Logística (modelo lineal)\033[0m")

    # Usamos un pipeline para preprocesar y ajustar los datos de forma automatizada
    pipe_rl = Pipeline(steps=[('scaler', 'passthrough'), ('poly', PolynomialFeatures()), ('solver', 'passthrough')])
    
    # Para la elección de hiper-parámetros, utilizamos Cross Validation entre distintos valores para cada hiperparámetro
    # Creamos dos grids, uno para utilizar la técnica de Gradiente Descendente Estocástico (SGD), y la otra para el solucionador
    # saga incorporado en la función LogisticRegression de sklearn.
    param_grid = [{
        # Estandarización de los datos a través de dos mecanismos distintos
        'scaler': [StandardScaler(), MinMaxScaler()],
        # Evaluación de clases de funciones lineales y cuadráticas
        'poly__degree': [1, 2],
        'solver': [SGDClassifier()],
        # La función de pérdida ha de ser logarítmica para que sgd ajuste un modelo de regresión logística
        'solver__loss': ['log'],
        # Probamos ambas funciones de regularización,
        'solver__penalty': ['l1', 'l2'],
        # si se aplica balanceo a los pesos,
        'solver__class_weight': ['balanced'],
        # se utilizan todos los procesadores,
        'solver__n_jobs': [-1],
        # se consideran múltiples valores del parámetro alfa,
        'solver__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        # se fija la semilla para reproducir consistentemente la aleatoriedad
        'solver__random_state': [1],
        # y el máximo de iteraciones para permitir la convergencia
        'solver__max_iter' :[1000000]
        },
        # Para el uso de LogisticRegression el proceso es similar
        {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'poly__degree': [1, 2],
        'solver': [LogisticRegression()],
        # Probamos valores de C, la intensidad de la regularización
        'solver__C' : [0.1, 1, 10],
        # la tolerancia a 3 decimales
        'solver__tol' : [0.001],
        # y el solucionador
        'solver__solver': ['saga'],
        'solver__penalty': ['l1', 'l2'],
        'solver__random_state' : [1],
        'solver__max_iter' : [100000]
        }
    ]

    # Ajustamos el mejor modelo de entre los probados con sus distintos hiperparámetro, conservando aquel que proporcione mejor f1-score con peso
    clf_rl = GridSearchCV(pipe_rl, param_grid, scoring='balanced_accuracy', n_jobs=-1).fit(X_train, y_train[:, i])

    # Imprimimos los resultados 
    print(f"Parámetros usados para ajustar el modelo de Regresión logística: {clf_rl.best_params_}")
    print(f"\nBondad del modelo de Regresión Logística con características estandarizadas para el modelo {nombre}")
    y_pred = clf_rl.predict(X_train)
    print(f"Ein = {1-balanced_accuracy_score(y_train[:, i], y_pred)}")
    y_pred = clf_rl.predict(X_test)
    print(f"Etest = {1-balanced_accuracy_score(y_test[:, i], y_pred)}")
    print(f"Ecv = {1-clf_rl.best_score_}")

    best_estimator = clf_rl
    best_name = "Regresión Logística"
    stop()

    plot_confusion_matrix(clf_rl.best_estimator_, X_test, y_test[:, i], display_labels=labels[i], values_format='d')
    plt.title(f"Matriz de confusión para el caso de {nombre}\n usando Regresión Logística")
    plt.ylabel(f"Clase verdadera")
    plt.xlabel(f"Clase predicha")
    plt.show()

    stop()

    # Ajuste mediante SVM

    print(f"\033[94;1;1mAjuste utilizando Support Vector Machine\033[0m")

    pipe_svm = Pipeline(steps=[('scaler', 'passthrough'), ('pca', 'passthrough'), ('svm', 'passthrough')])
    param_grid = [
        {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'pca': [PCA(), 'passthrough'],
        'svm' : [SVC()],
        'svm__kernel': ['poly'],
        'svm__degree': [1, 2, 3],
        'svm__gamma': ['scale', 'auto'],
        'svm__class_weight': [None, 'balanced'],
        'svm__C': [1, 10, 100, 1000],
        'svm__random_state': [1]
        },
        {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'pca': [PCA(), 'passthrough'],
        'svm' : [SVC()],
        'svm__kernel': ['rbf'],
        'svm__gamma': ['scale', 'auto'],
        'svm__class_weight': [None, 'balanced'],
        'svm__C': [1, 10, 100, 150, 200],
        'svm__random_state': [1]
        }
    ]

    clf_svm = GridSearchCV(pipe_svm, param_grid, scoring='balanced_accuracy', n_jobs=-1)
    clf_svm.fit(X_train, y_train[:, i])

    y_pred = clf_svm.predict(X_train)
    print(f"Ein = {1-balanced_accuracy_score(y_train[:, i], y_pred)}")
    y_pred = clf_svm.predict(X_test)
    print(f"Etest = {1-balanced_accuracy_score(y_test[:, i], y_pred)}")
    print(f"Ecv = {1-clf_svm.best_score_}")

    print(f"\nMejores hiperparámetros para este modelo: {clf_svm.best_params_}")

    stop()
    plot_confusion_matrix(clf_svm.best_estimator_, X_test, y_test[:, i], display_labels=labels[i], values_format='d')
    plt.title(f"Matriz de confusión para el caso de {nombre}\n usando Support Vector Machine")
    plt.ylabel(f"Clase verdadera")
    plt.xlabel(f"Clase predicha")
    plt.show()

    if (clf_svm.best_score_ > best_estimator.best_score_):
        best_estimator = clf_svm
        best_name = "Support Vector Machine"
    stop()

    # Ajuste a través de Random Forest
    print(f"\033[94;1;1mAjuste utilizando RandomForestClassifier\033[0m")

    pipe_rf = Pipeline(steps=[('scaler', 'passthrough'), ('randomforest', RandomForestClassifier())])
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'randomforest__n_estimators': [100, 200, 250, 300],
        'randomforest__max_depth': [10, 11, 12],
        'randomforest__criterion': ['gini', 'entropy'],
        'randomforest__max_features': ['sqrt'],
        'randomforest__class_weight': ['balanced', 'balanced_subsample'],
        'randomforest__n_jobs': [-1],
        'randomforest__random_state': [1]
    }

    clf_rf = GridSearchCV(pipe_rf, param_grid, scoring='balanced_accuracy', n_jobs=-1)
    clf_rf.fit(X_train, y_train[:, i])

    y_pred = clf_rf.predict(X_train)
    print(f"Ein = {1-balanced_accuracy_score(y_train[:, i], y_pred)}")
    y_pred = clf_rf.predict(X_test)
    print(f"Etest = {1-balanced_accuracy_score(y_test[:, i], y_pred)}")
    print(f"Ecv = {1-clf_rf.best_score_}")

    print(f"\nMejores hiperparámetros para este modelo: {clf_rf.best_params_}")

    stop()
    plot_confusion_matrix(clf_rf.best_estimator_, X_test, y_test[:, i], display_labels=labels[i], values_format='d')
    plt.title(f"Matriz de confusión para el caso de {nombre}\n usando Random Forest")
    plt.ylabel(f"Clase verdadera")
    plt.xlabel(f"Clase predicha")
    plt.show()

    if (clf_rf.best_score_ > best_estimator.best_score_):
        best_estimator = clf_rf
        best_name = "Random Forest"

    stop()

    # Ajuste a través de Boosting
    print(f"\033[94;1;1mAjuste utilizando AdaBoostClassifier\033[0m")

    pipe_ab = Pipeline(steps=[('scaler', 'passthrough'), ('adaboost', AdaBoostClassifier())])
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'adaboost__n_estimators': [50, 100, 200, 250, 300],
        'adaboost__base_estimator': [DecisionTreeClassifier(max_depth=1, class_weight='balanced'), DecisionTreeClassifier(max_depth=6, class_weight='balanced')],
        'adaboost__random_state': [1],
        'adaboost__learning_rate': [1, 1.1, 1.2]
    }

    clf_ab = GridSearchCV(pipe_ab, param_grid, scoring='balanced_accuracy', n_jobs=-1)
    clf_ab.fit(X_train, y_train[:, i])

    y_pred = clf_ab.predict(X_train)
    print(f"Ein = {1-balanced_accuracy_score(y_train[:, i], y_pred)}")
    y_pred = clf_ab.predict(X_test)
    print(f"Etest = {1-balanced_accuracy_score(y_test[:, i], y_pred)}")
    print(f"Ecv = {1-clf_ab.best_score_}")

    print(f"\nMejores hiperparámetros para este modelo: {clf_ab.best_params_}")

    stop()

    plot_confusion_matrix(clf_ab.best_estimator_, X_test, y_test[:, i], display_labels=labels[i], values_format='d')
    plt.title(f"Matriz de confusión para el caso de {nombre}\n usando AdaBoost")
    plt.ylabel(f"Clase verdadera")
    plt.xlabel(f"Clase predicha")
    plt.show()

    if (clf_ab.best_score_ > best_estimator.best_score_):
        best_estimator = clf_ab
        best_name = "AdaBoost"

    stop()

    # Ajuste a través de Perceptron Multicapa
    print(f"\033[94;1;1mAjuste utilizando Perceptrón Multicapa\033[0m")

    pipe_mlp = Pipeline(steps=[('scaler', 'passthrough'), ('mlp', MLPClassifier())])
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'mlp__hidden_layer_sizes' : [(50, 50, 50), (100, 100, 100)],
        'mlp__alpha' : [1e-4, 1e-3, 1e-2, 1e-1],
        'mlp__random_state' : [1],
        'mlp__solver' : ['lbfgs', 'sgd', 'adam'],
        'mlp__activation' : ['tanh', 'logistic'],
        'mlp__tol' : [0.001],
        'mlp__max_iter' : [10000]
    }

    clf_mlp = GridSearchCV(pipe_mlp, param_grid, scoring='balanced_accuracy', n_jobs=-1)
    clf_mlp.fit(X_train, y_train[:, i])

    y_pred = clf_mlp.predict(X_train)
    print(f"Ein = {1-balanced_accuracy_score(y_train[:, i], y_pred)}")
    y_pred = clf_mlp.predict(X_test)
    print(f"Etest = {1-balanced_accuracy_score(y_test[:, i], y_pred)}")
    print(f"Ecv = {1-clf_mlp.best_score_}")

    print(f"\nMejores hiperparámetros para este modelo: {clf_mlp.best_params_}")

    stop()

    plot_confusion_matrix(clf_mlp.best_estimator_, X_test, y_test[:, i], display_labels=labels[i], values_format='d')
    plt.title(f"Matriz de confusión para el caso de {nombre}\n usando Perceptrón Multicapa")
    plt.ylabel(f"Clase verdadera")
    plt.xlabel(f"Clase predicha")
    plt.show()

    if (clf_mlp.best_score_ > best_estimator.best_score_):
        best_estimator = clf_mlp
        best_name = "Perceptrón Multicapa"
    stop()


    print(f"El mejor modelo para la clasificación {nombre} es {best_name}")
    y_pred = best_estimator.predict(X_test)
    print(f"El reporte de clasificación de este modelo es:")
    print(classification_report(y_test[:, i], y_pred, target_names=labels[i]))

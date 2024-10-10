
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import itertools
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_target_balance(df, target_column='fraud'):
    """
    Grafica la distribución de la variable objetivo (target).
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        target_column (str): Columna que contiene la variable objetivo.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[target_column])
    plt.title(f'Distribución de la columna "{target_column}"')
    plt.xlabel(target_column)
    plt.ylabel('Conteo')
    plt.show()


def plot_outliers(df, numerical_columns):
    """
    Grafica un boxplot para las columnas numéricas para visualizar valores atípicos.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        numerical_columns (list): Lista de las columnas numéricas a analizar.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[numerical_columns], orient='h')
    plt.title('Boxplot de variables numéricas')
    plt.show()


def plot_numerical_distribution(df, numerical_columns, bins=5):
    """
    Grafica la distribución de las variables numéricas en forma de histogramas.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        numerical_columns (list): Lista de las columnas numéricas a analizar.
        bins (int): Número de bins para los histogramas.
    """
    plt.figure(figsize=(8, 4))
    for i, column in enumerate(numerical_columns):
        plt.subplot(1, len(numerical_columns), i + 1)
        sns.histplot(df[column], bins=bins, kde=True)
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()


def report_fraud_in_outliers(df, column, target_column):
    """
    Identifica valores atípicos en una columna numérica y calcula el porcentaje de fraudes entre ellos.

    Usa el rango intercuartílico (IQR) para detectar outliers y determina cuántos de ellos están asociados 
    con casos de fraude, asumiendo que 1 en la columna objetivo indica fraude.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        column (str): Columna numérica para detectar outliers.
        target_column (str): Columna que indica fraudes (1 = fraude).

    Returns:
        num_fraudes (int): Número de fraudes entre los outliers.
        fraud_percentage (float): Porcentaje de fraudes entre los outliers.
    """
    # Calcular cuartiles y rango intercuartílico (IQR)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir límites para los valores atípicos
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtrar outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    total_outliers = len(outliers)

    # Contar los fraudes entre los outliers
    num_fraudes = outliers[target_column].sum()  # Asume que 1 es fraude
    
    # Calcular el porcentaje de fraudes respecto al total de outliers
    fraud_percentage = (num_fraudes / total_outliers * 100) if total_outliers > 0 else 0
    
    # Imprimir resultados
    print(f"Número de fraudes entre los outliers de {column}: {num_fraudes}")
    print(f"Porcentaje de fraudes entre los outliers de {column}: {fraud_percentage:.2f}%")
    
    


def count_negative_values(df, numerical_columns):
    """
    Cuenta y muestra el número de valores negativos en las columnas numéricas.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        numerical_columns (list): Lista de las columnas numéricas a analizar.
    """
    for column in numerical_columns:
        negative_count = (df[column] < 0).sum()
        print(f"Número de valores menores a cero en {column}: {negative_count}")

def plot_correlation_matrix(df):
    """
    Grafica la matriz de correlación entre las variables numéricas del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
    """
    correlation_mat = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_mat, annot=True, cmap="Blues", linewidths=0.8)
    plt.title('Matriz de correlación')
    plt.show()


def train_evaluate_and_plot(model, X_train, y_train, X_test, y_test, model_name="Modelo"):
    """
    Entrena un modelo clasificatorio, evalúa su rendimiento y grafica la importancia de las variables.
    
    Args:
        model: El modelo a entrenar (ej: LogisticRegression, DecisionTreeClassifier, etc.).
        X_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Datos de prueba.
        y_test (pd.Series): Variable objetivo de prueba.
        model_name (str): Nombre del modelo para mostrar en los resultados y gráficos.
    """
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular las métricas de rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Imprimir las métricas
    print(f"Resultados para {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"AUC: {auc:.4f}")


def plot_feature_importance(model, feature_names, model_name="Modelo"):
    """
    Grafica la importancia de las variables en un modelo.
    
    Args:
        model: El modelo entrenado (con coeficientes o atributos de importancia).
        feature_names (list): Lista de nombres de las características.
        model_name (str): Nombre del modelo para el título.
    """
    if hasattr(model, 'coef_'):  # Regresión logística
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    elif hasattr(model, 'feature_importances_'):  # Árboles o ensembles
        importances = pd.Series(model.feature_importances_, index=feature_names)
    
    importances.sort_values(ascending=True, inplace=True)

    plt.figure(figsize=(8, 4))
    importances.plot(kind='barh', color='skyblue')
    plt.title(f"Importancia de Variables en {model_name}")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.tight_layout()
    plt.show()
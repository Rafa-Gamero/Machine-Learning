
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
from sklearn.model_selection import GridSearchCV


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


def evaluate_model_with_confusion(model, X_test, y_test, model_name="Modelo"):
    """Función para evaluar el modelo y mostrar métricas junto a la matriz de confusión."""
    # Predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # Crear una figura
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Dos columnas

    # Cuadro de métricas
    metrics_text = (f"Resultados del {model_name}:\n"
                    f"Accuracy: {accuracy:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    f"F1 Score: {f1:.4f}\n"
                    f"AUC: {auc:.4f}")
    
    # Añadir el cuadro de texto
    ax[0].text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center')
    ax[0].axis('off')  # Ocultar ejes

    # Graficar la matriz de confusión
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
    ax[1].set_title(f"Matriz de Confusión - {model_name}")
    ax[1].set_ylabel('Clase verdadera')
    ax[1].set_xlabel('Clase predicha')

    plt.tight_layout()
    plt.show()
    


def plot_feature_importance(model, X_train, model_name="Modelo"):
    """Función para graficar la importancia de las variables."""
    # Verificar si el modelo tiene el atributo `feature_importances_`
    if hasattr(model, 'feature_importances_'):
        # Obtener la importancia de las características
        importance = pd.Series(model.feature_importances_, index=X_train.columns)
        importance.sort_values(ascending=True, inplace=True)

        # Graficar
        plt.figure(figsize=(8, 4))
        importance.plot(kind='barh', color='skyblue')
        plt.title(f"Importancia de Variables en {model_name}")
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.tight_layout()
        plt.show()
    else:
        print(f"El modelo {model_name} no soporta la importancia de variables.")


def plot_logistic_regression_importance(model, X_train):
    """
    Función para graficar la importancia de las variables en un modelo de regresión logística.

    :param model: Modelo de regresión logística ajustado (LogisticRegression).
    :param X_train: DataFrame de pandas que contiene las características utilizadas para ajustar el modelo.
    """
    # Obtener los coeficientes del modelo de regresión logística
    coef_importance = pd.Series(np.abs(model.coef_[0]), index=X_train.columns)

    coef_importance.sort_values(ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef_importance.values, y=coef_importance.index)

    plt.title("Importancia de Variables en Regresión Logística", fontsize=16)
    plt.xlabel("Importancia (Valor Absoluto del Coeficiente)", fontsize=12)
    plt.ylabel("Características", fontsize=12)

    plt.tight_layout()

    plt.show()     
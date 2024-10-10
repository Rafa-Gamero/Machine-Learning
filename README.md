# Análisis de Transacciones Fraudulentas con Ingeniería de Características y Balanceo de Clases

## Descripción del Proyecto
Este proyecto analiza un conjunto de datos de transacciones financieras con el objetivo de detectar transacciones fraudulentas. Se implementan técnicas de ingeniería de características, exploración y limpieza de datos, y se aplican estrategias de balanceo de clases para mejorar la capacidad predictiva de los modelos de detección de fraude.

## Contenido del Archivo de Datos
El archivo `card_transdata.csv` contiene las siguientes columnas:

- `distance_from_home`: Distancia desde el hogar del titular de la tarjeta.
- `distance_from_last_transaction`: Distancia desde la última transacción realizada.
- `ratio_to_median_purchase_price`: Relación del precio de la compra en comparación con el precio mediano de compra.
- `repeat_retailer`: Indica si la transacción fue realizada en un comercio previamente visitado.
- `used_chip`: Indica si se utilizó un chip en la tarjeta para la transacción.
- `used_pin_number`: Indica si se utilizó un PIN durante la transacción.
- `online_order`: Indica si la transacción se realizó en línea.
- `fraud`: Variable objetivo que indica si la transacción fue fraudulenta (1) o no (0).

## Exploración de Datos

1. **Ausencia de Valores Nulos y Duplicados**:
   - El conjunto de datos no contiene valores nulos ni duplicados, lo que elimina la necesidad de tratamiento adicional para estos aspectos.

2. **Desbalance en la Variable Objetivo (Fraud)**:
   - El conjunto de datos está desbalanceado, con una mayor cantidad de transacciones no fraudulentas. Esto implica la necesidad de aplicar técnicas de balanceo de clases (como undersampling o oversampling).

3. **Valores Atípicos (Outliers)**:
   - Se detectaron outliers en varias variables numéricas. Estos valores atípicos se asocian a transacciones fraudulentas, lo que es consistente con la naturaleza anómala de estas transacciones.

4. **Correlación entre Variables**:
   - No hay correlaciones significativas entre las variables explicativas, pero se observa una fuerte correlación entre la variable `ratio_to_median_purchase_price` y la variable objetivo `fraud`, lo que la convierte en un predictor clave.

## Limpieza de Datos
- Las variables numéricas se transformaron en categóricas, y las categóricas fueron codificadas utilizando One-Hot Encoding.

## Ingeniería de Características
- Las variables categóricas fueron transformadas en variables dummy utilizando One-Hot Encoding.
- Las características fueron escaladas utilizando `RobustScaler` para mitigar el impacto de los valores atípicos sin perder su valor predictivo.

## Modelado

### Regresión Logística
- **Accuracy**: 0.9350
- **Precision**: 0.5762
- **Recall**: 0.9522
- **F1 Score**: 0.7180
- **AUC**: 0.9427
- **Confusion Matrix**:
![Matriz de Confusion Regresión Logística](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20164926.png)

- **Gráfico de Importancia de Variables**:
![Importancia Variables Regresión Logística](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20164943.png)

### Árbol de Decisión
- **Accuracy**: 0.9998
- **Precision**: 0.9981
- **Recall**: 1.0000
- **F1 Score**: 0.9991
- **AUC**: 0.9999
- **Confusion Matrix**:
- ![Matriz de Confusion Árbol de Decisión](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20165000.png)

- **Gráfico de Importancia de Variables**:
![Importancia Variables Árbol de Decisión](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20165015.png)

### Random Forest
- **Accuracy**: 0.9999
- **Precision**: 0.9984
- **Recall**: 1.0000
- **F1 Score**: 0.9992
- **AUC**: 0.9999
- **Confusion Matrix**:
- ![Matriz de Confusion Random Forest](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20165029.png)

- **Gráfico de Importancia de Variables**:
![Importancia Variables Random Forest](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20165047.png)

### Gradient Boosting
- **Accuracy**: 0.9979
- **Precision**: 0.9770
- **Recall**: 0.9998
- **F1 Score**: 0.9883
- **AUC**: 0.9988
- **Confusion Matrix**:

- **Gráfico de Importancia de Variables**:
![Importancia Variables Gradient Boosting](ruta_imagen_gb.png)

### Cuadro Comparativo de Métricas

| Modelo               | Accuracy | Precision | Recall | F1 Score | AUC   |
|----------------------|----------|-----------|--------|----------|-------|
| Regresión Logística   | 0.9350   | 0.5762    | 0.9522 | 0.7180   | 0.9427|
| Árbol de Decisión     | 0.9998   | 0.9981    | 1.0000 | 0.9991   | 0.9999|
| Random Forest         | 0.9999   | 0.9984    | 1.0000 | 0.9992   | 0.9999|
| Gradient Boosting     | 0.9979   | 0.9770    | 0.9998 | 0.9883   | 0.9988|

## Enlaces 

- [Datos](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- [Presentación]()

## Puedes contactar con nosotros:

- [Ana Nofuentes Solano](https://www.linkedin.com/in/ana-nofuentes-solano-654026a3/)
- [Óscar Sánchez Riveros](https://www.linkedin.com/in/oscar-sanchez-riveros/)
- [Rafael Gamero Arrabal](https://www.linkedin.com/in/rafael-gamero-arrabal-619200186/)




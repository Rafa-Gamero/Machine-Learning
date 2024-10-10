# Análisis de Transacciones Fraudulentas con Ingeniería de Características y Balanceo de Clases

Descripción del Proyecto
Este proyecto analiza un conjunto de datos de transacciones financieras con el objetivo de detectar transacciones fraudulentas. Se implementan técnicas de ingeniería de características, exploración y limpieza de datos, y se aplican estrategias de balanceo de clases para mejorar la capacidad predictiva de los modelos de detección de fraude.

Contenido del Archivo de Datos
El archivo card_transdata.csv contiene las siguientes columnas:

distance_from_home: Distancia desde el hogar del titular de la tarjeta.
distance_from_last_transaction: Distancia desde la última transacción realizada.
ratio_to_median_purchase_price: Relación del precio de la compra en comparación con el precio mediano de compra.
repeat_retailer: Indica si la transacción fue realizada en un comercio previamente visitado.
used_chip: Indica si se utilizó un chip en la tarjeta para la transacción.
used_pin_number: Indica si se utilizó un PIN durante la transacción.
online_order: Indica si la transacción se realizó en línea.
fraud: Variable objetivo que indica si la transacción fue fraudulenta (1) o no (0).

Exploración de Datos
Ausencia de Valores Nulos y Duplicados:
El conjunto de datos no contiene valores nulos ni duplicados, lo que elimina la necesidad de tratamiento adicional para estos aspectos.
Desbalance en la Variable Objetivo (Fraud):
El conjunto de datos está desbalanceado, con una mayor cantidad de transacciones no fraudulentas. Esto implica la necesidad de aplicar técnicas de balanceo de clases (como undersampling o oversampling).
Valores Atípicos (Outliers):
Se detectaron outliers en varias variables numéricas. Estos valores atípicos se asocian a transacciones fraudulentas, lo que es consistente con la naturaleza anómala de estas transacciones.
Correlación entre Variables:
No hay correlaciones significativas entre las variables explicativas, pero se observa una fuerte correlación entre la variable ratio_to_median_purchase_price y la variable objetivo fraud, lo que la convierte en un predictor clave.

Limpieza de Datos
Las variables numéricas se transformaron en categóricas, y las categóricas fueron codificadas utilizando One-Hot Encoding.

Ingeniería de Características
Las variables categóricas fueron transformadas en variables dummy utilizando One-Hot Encoding.
Las características fueron escaladas utilizando RobustScaler para mitigar el impacto de los valores atípicos sin perder su valor predictivo.

Enlaces 
[Datos](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
[Presentación]()

Puedes contactar con nosotros 
[Linkedin](https://www.linkedin.com/in/ana-nofuentes-solano-654026a3/)
[Linkedin](https://www.linkedin.com/in/oscar-sanchez-riveros/)
[Linkedin](https://www.linkedin.com/in/rafael-gamero-arrabal-619200186/)

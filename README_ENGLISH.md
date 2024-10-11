# Analysis of Fraudulent Transactions with Feature Engineering and Class Balancing

## Project Description
This project analyzes a dataset of financial transactions with the aim of detecting fraudulent transactions. It implements feature engineering techniques, data exploration and cleaning, and applies class balancing strategies to enhance the predictive power of fraud detection models.

## Dataset Contents
The file `card_transdata.csv` contains the following columns:

- `distance_from_home`: Distance from the cardholder's home.
- `distance_from_last_transaction`: Distance from the last transaction made.
- `ratio_to_median_purchase_price`: Ratio of the purchase price compared to the median purchase price.
- `repeat_retailer`: Indicates whether the transaction was made at a previously visited retailer.
- `used_chip`: Indicates whether a chip was used in the card for the transaction.
- `used_pin_number`: Indicates whether a PIN was used during the transaction.
- `online_order`: Indicates whether the transaction was made online.
- `fraud`: Target variable indicating whether the transaction was fraudulent (1) or not (0).

## Data Exploration

1. **Absence of Null Values and Duplicates**:
   - The dataset does not contain null or duplicate values, which eliminates the need for further treatment regarding these aspects.

2. **Imbalance in the Target Variable (Fraud)**:
   - The dataset is imbalanced, with a greater number of non-fraudulent transactions. This necessitates the application of class balancing techniques (such as undersampling or oversampling).

3. **Outliers**:
   - Outliers were detected in several numerical variables. These outliers are associated with fraudulent transactions, which is consistent with the anomalous nature of these transactions.

4. **Correlation Between Variables**:
   - There are no significant correlations among the explanatory variables, but a strong correlation is observed between the variable `ratio_to_median_purchase_price` and the target variable `fraud`, making it a key predictor.

## Data Cleaning
- Numerical variables were transformed into categorical ones, and categorical variables were encoded using One-Hot Encoding.

## Feature Engineering
- Categorical variables were transformed into dummy variables using One-Hot Encoding.
- Features were scaled using `RobustScaler` to mitigate the impact of outliers without losing their predictive value.

## Modeling

### Logistic Regression
- **Accuracy**: 0.9350
- **Precision**: 0.5762
- **Recall**: 0.9522
- **F1 Score**: 0.7180
- **AUC**: 0.9427
- **Confusion Matrix**:
![Confusion Matrix Logistic Regression](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20164926.png)

- **Feature Importance Chart**:
![Feature Importance Logistic Regression](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20164943.png)

### Decision Tree
- **Accuracy**: 0.9998
- **Precision**: 0.9981
- **Recall**: 1.0000
- **F1 Score**: 0.9991
- **AUC**: 0.9999
- **Confusion Matrix**:
![Confusion Matrix Decision Tree](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20165000.png)

- **Feature Importance Chart**:
![Feature Importance Decision Tree](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20165015.png)

### Random Forest
- **Accuracy**: 0.9999
- **Precision**: 0.9984
- **Recall**: 1.0000
- **F1 Score**: 0.9992
- **AUC**: 0.9999
- **Confusion Matrix**:
![Confusion Matrix Random Forest](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20165029.png)

- **Feature Importance Chart**:
![Feature Importance Random Forest](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20165047.png)

### Gradient Boosting
- **Accuracy**: 0.9979
- **Precision**: 0.9770
- **Recall**: 0.9998
- **F1 Score**: 0.9883
- **AUC**: 0.9988
- **Confusion Matrix**:
![Confusion Matrix Gradient Boosting](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20165059.png)

- **Feature Importance Chart**:
![Feature Importance Gradient Boosting](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/images/Captura%20de%20pantalla%202024-10-10%20165110.png)

### Comparative Metrics Table

| Model                | Accuracy | Precision | Recall | F1 Score | AUC   |
|----------------------|----------|-----------|--------|----------|-------|
| Logistic Regression   | 0.9344   | 0.5741    | 0.9522 | 0.7169   | 0.9434|
| Decision Tree         | 0.9998   | 0.9983    | 1.0000 | 0.9991   | 0.9999|
| Random Forest         | 0.9999   | 0.9989    | 1.0000 | 0.9994   | 0.9999|
| Gradient Boosting     | 0.9998   | 0.9987    | 0.9999 | 0.9993   | 0.9999|

## Links 

- [Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- [Presentation](https://www.canva.com/design/DAGTLP_zEBo/ufT_Eso4muMz71UY7lXamQ/view?utm_content=DAGTLP_zEBo&utm_campaign=designshare&utm_medium=link&utm_source=editor)

## You can contact us:

- [Ana Nofuentes Solano](https://www.linkedin.com/in/ana-nofuentes-solano-654026a3/)
- [Óscar Sánchez Riveros](https://www.linkedin.com/in/oscar-sanchez-riveros/)
- [Rafael Gamero Arrabal](https://www.linkedin.com/in/rafael-gamero-arrabal-619200186/)



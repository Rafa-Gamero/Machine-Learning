# Fraud Transaction Analysis with Feature Engineering and Class Balancing

## Project Description
This project analyzes a financial transaction dataset to detect fraudulent transactions. Feature engineering techniques, data exploration, cleaning, and class balancing strategies are applied to improve the predictive performance of fraud detection models.

## Dataset Contents
The file `card_transdata.csv` contains the following columns:

- `distance_from_home`: Distance from the cardholder's home.
- `distance_from_last_transaction`: Distance from the last transaction.
- `ratio_to_median_purchase_price`: Ratio of the purchase price compared to the median purchase price.
- `repeat_retailer`: Indicates whether the transaction was made at a previously visited retailer.
- `used_chip`: Indicates if a chip was used for the transaction.
- `used_pin_number`: Indicates if a PIN was used during the transaction.
- `online_order`: Indicates if the transaction was made online.
- `fraud`: Target variable that indicates if the transaction was fraudulent (1) or not (0).

## Data Exploration

1. **Missing Values and Duplicates**:
   - The dataset contains no missing values or duplicates, so no further processing is needed in this regard.

2. **Class Imbalance in the Target Variable (Fraud)**:
   - The dataset is imbalanced, with more non-fraudulent transactions. This requires applying class balancing techniques (such as undersampling or oversampling).

3. **Outliers**:
   - Outliers were detected in several numerical variables. These outliers are associated with fraudulent transactions, consistent with the anomalous nature of such transactions.

4. **Correlation Between Variables**:
   - No significant correlations were found between explanatory variables. However, a strong correlation was observed between `ratio_to_median_purchase_price` and the target variable `fraud`, making it a key predictor.

## Data Cleaning
- Numerical variables were transformed into categorical ones, and categorical variables were encoded using One-Hot Encoding.

## Feature Engineering
- Categorical variables were transformed into dummy variables using One-Hot Encoding.
- Features were scaled using `RobustScaler` to mitigate the impact of outliers while retaining predictive power.

## Modeling

### Logistic Regression
- **Accuracy**: 0.9350
- **Precision**: 0.5762
- **Recall**: 0.9522
- **F1 Score**: 0.7180
- **AUC**: 0.9427
- **Confusion Matrix**:
 ![Matriz de Confusion Regresión Logística](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20164926.png)
- **Variable Importance Chart:**:
  ![Importancia Variables Regresión Logística](https://github.com/Rafa-Gamero/Machine-Learning/blob/main/Captura%20de%20pantalla%202024-10-10%20164943.png)
### Decision Tree
- **Accuracy**: 0.9998
- **Precision**: 0.9981
- **Recall**: 1.0000
- **F1 Score**: 0.9991
- **AUC**: 0.9999
- **Confusion Matrix**: (Confusion Matrix for Decision Tree)
- **Variable Importance Chart:**:
### Random Forest
- **Accuracy**: 0.9999
- **Precision**: 0.9984
- **Recall**: 1.0000
- **F1 Score**: 0.9992
- **AUC**: 0.9999
- **Confusion Matrix**: (Confusion Matrix for Random Forest)
- **Variable Importance Chart:**:
### Gradient Boosting
- **Accuracy**: 0.9979
- **Precision**: 0.9770
- **Recall**: 0.9998
- **F1 Score**: 0.9883
- **AUC**: 0.9988
- **Confusion Matrix**: (Confusion Matrix for Gradient Boosting)
- **Variable Importance Chart:**:
## Comparative Metrics Table
| Model               | Accuracy | Precision | Recall  | F1 Score | AUC    |
|---------------------|----------|-----------|---------|----------|--------|
| Logistic Regression | 0.9350   | 0.5762    | 0.9522  | 0.7180   | 0.9427 |
| Decision Tree       | 0.9998   | 0.9981    | 1.0000  | 0.9991   | 0.9999 |
| Random Forest       | 0.9999   | 0.9984    | 1.0000  | 0.9992   | 0.9999 |
| Gradient Boosting   | 0.9979   | 0.9770    | 0.9998  | 0.9883   | 0.9988 |

## Links
- [Data](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- [Presentation]()

You can contact us:
- [Ana Nofuentes Solano](https://www.linkedin.com/in/ana-nofuentes-solano-654026a3/)
- [Óscar Sánchez Riveros](https://www.linkedin.com/in/oscar-sanchez-riveros/)
- [Rafael Gamero Arrabal](https://www.linkedin.com/in/rafael-gamero-arrabal-619200186/)


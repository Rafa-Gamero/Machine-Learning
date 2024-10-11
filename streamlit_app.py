import streamlit as st
import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('model_decision_tree.pkl')

# App title
st.title("Fraud Detection App")

# App description
st.write("This application predicts whether a transaction is fraudulent or not.")

# Function to collect user input using Streamlit widgets
def get_user_input():
    # Collect inputs from the user
    distance_from_home = st.number_input("Enter distance from home (e.g., 1.5):", min_value=0.0, format="%.2f")
    distance_from_last_transaction = st.number_input("Enter distance from last transaction (e.g., 2.0):", min_value=0.0, format="%.2f")
    ratio_to_median_purchase_price = st.number_input("Enter ratio to median purchase price (e.g., 0.75):", min_value=0.0, format="%.2f")

    # Collect 'Yes' or 'No' inputs for the categorical fields
    repeat_retailer = st.radio("Is this a repeat retailer?", ('Yes', 'No'))
    used_chip = st.radio("Was a chip used?", ('Yes', 'No'))
    used_pin_number = st.radio("Was a PIN used?", ('Yes', 'No'))
    online_order = st.radio("Is this an online order?", ('Yes', 'No'))

    # Convert the Yes/No responses into 0 and 1 encoded values
    repeat_retailer_0 = 1 if repeat_retailer == 'No' else 0
    repeat_retailer_1 = 1 if repeat_retailer == 'Yes' else 0

    used_chip_0 = 1 if used_chip == 'No' else 0
    used_chip_1 = 1 if used_chip == 'Yes' else 0

    used_pin_number_0 = 1 if used_pin_number == 'No' else 0
    used_pin_number_1 = 1 if used_pin_number == 'Yes' else 0

    online_order_0 = 1 if online_order == 'No' else 0
    online_order_1 = 1 if online_order == 'Yes' else 0

    # Create a DataFrame with the necessary columns for the model
    data = {
        'distance_from_home': [distance_from_home],
        'distance_from_last_transaction': [distance_from_last_transaction],
        'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
        'repeat_retailer_0': [repeat_retailer_0],
        'repeat_retailer_1': [repeat_retailer_1],
        'used_chip_0': [used_chip_0],
        'used_chip_1': [used_chip_1],
        'used_pin_number_0': [used_pin_number_0],
        'used_pin_number_1': [used_pin_number_1],
        'online_order_0': [online_order_0],
        'online_order_1': [online_order_1]
    }
    features = pd.DataFrame(data)
    return features

# Call the function to get user input
X_test = get_user_input()

# Display the collected data as a table, adjust the width to fit screen
st.write("### Transaction Details:", X_test.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector="th", props=[('text-align', 'center')])]))

# Button to trigger the prediction
if st.button("Predict"):
    prediction = loaded_model.predict(X_test)
    
    if prediction == 0:
        st.success("The transaction is NOT fraudulent.")
    else:
        st.error("Fraud detected! This transaction is likely fraudulent.")


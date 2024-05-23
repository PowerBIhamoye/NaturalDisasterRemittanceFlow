import numpy as np
import pickle
import streamlit as st

# Load the remittance prediction model
model_path = 'Models/model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))

# Load the One-Hot Encoder
encoder_path = 'Models/one_hot_encoder_country.pkl'
encoder = pickle.load(open(encoder_path, 'rb'))

# Load the Standard Scaler for the year
scaler_path = 'Models/scaler.pkl'
scaler = pickle.load(open(scaler_path, 'rb'))

# Creating a function for remittance prediction
def remittance_prediction(country, year, total_deaths, total_affected, gdp):
    # Encode the country using the pre-trained OneHotEncoder
    country_encoded = encoder.transform([[country]])
    numerical_input = [year, total_deaths, total_affected, gdp]  # Adjust based on your features
    numerical_input_as_float = [float(x) for x in numerical_input]
    numerical_input_as_numpy_array = np.asarray(numerical_input_as_float).reshape(1, -1)
    
    # Scale the numerical input data using the pre-trained StandardScaler
    numerical_input_scaled = scaler.transform(numerical_input_as_numpy_array)

    # Combine with the numerical_input
    #input_data = np.concatenate((country_encoded, np.array([[float(year)]])), axis=1)
    input_data = np.concatenate((country_encoded,  numerical_input_scaled), axis=1)
    prediction = loaded_model.predict(input_data)
    return prediction[0]


# Function to format remittance amounts
def format_remittance(amount):
    if amount >= 1e9:
        return f"{amount / 1e9:,.2f} billion"
    elif amount >= 1e6:
        return f"{amount / 1e6:,.2f} million"
    elif amount >= 1e3:
        return f"{amount / 1e3:,.2f} thousand"
    else:
        return f"{amount:,.2f}"


def main():
    # Giving a title
    st.title('Remittance Prediction Application')

    # Getting the input data from the user
    country = st.text_input('Country', 'Nigeria')
    year = st.number_input("Year", min_value=None, max_value=None, value=2025, step=1, format="%d")
    total_deaths = st.number_input("Total Deaths", min_value=None, max_value=None, value=200, step=1, format="%d")
    total_affected = st.number_input("Total Affected", min_value=None, max_value=None, value=3000, step=1, format="%d")
    gdp = st.number_input("GDP", min_value=None, max_value=None, value=472000000000, step=1, format="%d")
    #numerical_cols = ['Total Deaths', 'Total Affected', 'Year', 'GDP']

    # Code for Prediction
    if st.button('Predict Remittance'):
        try:
            prediction = remittance_prediction(country, year, total_deaths, total_affected, gdp)
            formatted_prediction = format_remittance(prediction)
            st.success(f'Predicted remittance for {country} in {year} is: {formatted_prediction}')
        except ValueError as e:
            st.error(f'Error: {e}. Please ensure the country name is correct and matches the training data.')

if __name__ == '__main__':
    main()

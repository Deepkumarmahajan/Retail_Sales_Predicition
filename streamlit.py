import streamlit as st
import pandas as pd
import pickle
import os

# Load the preprocessor
if os.path.exists('artifacts/proprocessor.pkl'):
    with open('artifacts/proprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
else:
    st.error("Preprocessor file not found.")

# Load the model
if os.path.exists('artifacts/model.pkl'):
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    st.error("Model file not found.")

# Create the Streamlit web app
def main():
    st.title("Retail Sales Prediction")
    
    # Upload and preprocess data
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            preprocessed_data = preprocess_data(data)
            if preprocessed_data is not None:  # Only proceed if preprocessing succeeds
                predictions = make_predictions(preprocessed_data)
                display_predictions(predictions, data)
        except FileNotFoundError:
            st.write("The required files could not be found. Please check the paths.")
        except ValueError as e:
            st.write(f"An error occurred: {e}. Please ensure the CSV is correctly formatted.")
        except Exception as e:
            st.write(f"An unexpected error occurred: {e}")

def preprocess_data(data):
    # Select the required columns
    columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
               'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
               'CompetitionDistance', 'CompetitionOpenSinceMonth',
               'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
               'Promo2SinceYear', 'PromoInterval']
    # Check if all required columns are present
    if not all(col in data.columns for col in columns):
        st.error("The uploaded CSV is missing some required columns.")
        return None
    
    # Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # Preprocess the data using the preprocessor
    preprocessed_data = preprocessor.transform(data)

    return preprocessed_data

def make_predictions(data):
    # Make predictions using the model
    predictions = model.predict(data)
    return predictions

def display_predictions(predictions, data):
    # Display the expected sales values along with store number and date
    result_df = pd.DataFrame({
        'Store': data['Store'],
        'Date': pd.to_datetime(data['Date']),
        'Expected Sales': predictions
    })
    st.write(result_df)

if __name__ == '__main__':
    main()

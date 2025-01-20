import streamlit as st
import pandas as pd
import pickle
import os

# Load the preprocessor
if os.path.exists('artifacts/proprocessor.pkl'):
    with open('artifacts/proprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
else:
    st.error("⚠️ Preprocessor file not found.")

# Load the model
if os.path.exists('artifacts/model.pkl'):
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    st.error("⚠️ Model file not found.")

# Create the Streamlit web app
def main():
    # Page config to make it clean and with emojis
    st.set_page_config(page_title="Retail Sales Prediction", page_icon="📊", layout="wide")
    
    # Sidebar with upload option and header
    st.sidebar.header("🔧 Upload Data & Settings")
    uploaded_file = st.sidebar.file_uploader("Upload your sales data (CSV)", type="csv")
    
    if uploaded_file is not None:
        with st.spinner('⏳ Processing your file...'):
            try:
                # Reading the CSV file
                data = pd.read_csv(uploaded_file)
                preprocessed_data = preprocess_data(data)
                if preprocessed_data is not None:
                    predictions = make_predictions(preprocessed_data)
                    display_predictions(predictions, data)
            except FileNotFoundError:
                st.error("❌ The required files could not be found. Please check the paths.")
            except ValueError as e:
                st.error(f"❌ Error: {e}. Please ensure the CSV file is properly formatted.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

def preprocess_data(data):
    # Select required columns
    columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
               'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
               'CompetitionDistance', 'CompetitionOpenSinceMonth',
               'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
               'Promo2SinceYear', 'PromoInterval']
    
    # Check if the required columns are present
    if not all(col in data.columns for col in columns):
        st.error("⚠️ Missing some required columns in the CSV.")
        return None
    
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # Preprocess using the preprocessor
    preprocessed_data = preprocessor.transform(data)
    
    return preprocessed_data

def make_predictions(data):
    # Make predictions using the trained model
    predictions = model.predict(data)
    return predictions

def display_predictions(predictions, data):
    # Display predictions along with store number and date
    result_df = pd.DataFrame({
        'Store': data['Store'],
        'Date': pd.to_datetime(data['Date']),
        'Predicted Sales': predictions
    })
    
    st.subheader("💡 Predictions Overview")
    st.write(result_df)
    
    # Display message on predictions
    st.markdown("""
    ### 🧮 How the Prediction Works

    The model uses historical data including store details, promotions, competition distance, and more to forecast the sales for each store. Here's how it works:
    
    1. **Preprocessing**: The data is cleaned and transformed to handle any missing values and categorical data.
    2. **Prediction**: After preprocessing, the model makes predictions based on the transformed data.
    3. **Output**: You get a forecast of expected sales for each store.

    ### 🎯 Purpose of the Prediction

    The primary goal of these predictions is to help businesses:
    - Optimize inventory management 📦.
    - Plan promotions more effectively 🎉.
    - Make data-driven decisions to improve profitability 💰.

    ### 📈 Ultimate Goal of This Report

    This report aims to empower business owners, analysts, and decision-makers to forecast future sales accurately. It helps them make smarter decisions around stock levels, staffing, and marketing strategies.
    """)
    
    # Section for further explanation on the purpose of the model
    st.markdown("""
    #### ✨ Key Benefits:
    - Predict future sales to optimize stock levels and avoid overstocking or understocking.
    - Leverage predictions to plan promotions effectively and understand demand patterns.
    - Use sales forecasts to align staffing and operational costs.
    
    #### ⚙️ How We Calculate:
    - **Model Training**: The model is trained on historical sales data, which includes various factors affecting sales.
    - **Sales Forecast**: The model then predicts the sales based on new, unseen data (uploaded by you) after preprocessing.
    - **Accuracy**: We ensure that the model is robust and can handle different types of data variations.
    """)
    
    # Emojis for the final touch
    st.markdown("""
    ### 🌟 Thank you for using the Retail Sales Prediction tool!
    We're here to help you forecast, plan, and make better business decisions. If you have any questions, feel free to reach out! 📧
    """)

if __name__ == '__main__':
    main()

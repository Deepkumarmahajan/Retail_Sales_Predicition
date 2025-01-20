import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.set_page_config(page_title="Retail Sales Prediction", page_icon="📊", layout="wide")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Upload Data & Settings")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        with st.spinner('Processing your file...'):
            try:
                data = pd.read_csv(uploaded_file)
                preprocessed_data = preprocess_data(data)
                if preprocessed_data is not None:  # Only proceed if preprocessing succeeds
                    predictions = make_predictions(preprocessed_data)
                    display_predictions(predictions, data)
            except FileNotFoundError:
                st.error("❌ The required files could not be found. Please check the paths.")
            except ValueError as e:
                st.error(f"❌ An error occurred: {e}. Please ensure the CSV is correctly formatted.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")

def preprocess_data(data):
    # Select the required columns
    columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
               'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
               'CompetitionDistance', 'CompetitionOpenSinceMonth',
               'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
               'Promo2SinceYear', 'PromoInterval']
    # Check if all required columns are present
    if not all(col in data.columns for col in columns):
        st.error("⚠️ The uploaded CSV is missing some required columns.")
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
    
    st.subheader("📈 Predictions Overview")
    st.write(result_df)
    
    # Visualize predictions in a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result_df['Date'], result_df['Expected Sales'], label="Predicted Sales", color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Expected Sales')
    ax.set_title('Retail Sales Predictions Over Time')
    ax.legend()
    st.pyplot(fig)
    
    # Bar chart to compare actual vs predicted sales
    st.subheader("📊 Actual vs Predicted Sales")
    data['Sales'] = data['Sales'].fillna(0)  # Fill missing sales with 0 for comparison
    bar_chart_data = pd.DataFrame({
        'Store': data['Store'],
        'Date': data['Date'],
        'Actual Sales': data['Sales'],
        'Predicted Sales': predictions
    })
    
    bar_fig, bar_ax = plt.subplots(figsize=(10, 6))
    bar_chart_data.groupby('Store')[['Actual Sales', 'Predicted Sales']].mean().plot(kind='bar', ax=bar_ax)
    bar_ax.set_ylabel('Sales')
    bar_ax.set_title('Actual vs Predicted Sales by Store')
    st.pyplot(bar_fig)
    
    # Distribution of predicted sales
    st.subheader("📉 Distribution of Predicted Sales")
    hist_fig, hist_ax = plt.subplots(figsize=(10, 6))
    hist_ax.hist(predictions, bins=30, color='skyblue', edgecolor='black')
    hist_ax.set_xlabel('Predicted Sales')
    hist_ax.set_ylabel('Frequency')
    hist_ax.set_title('Distribution of Predicted Sales')
    st.pyplot(hist_fig)
    
    # Correlation matrix for input features
    st.subheader("🔍 Feature Correlation")
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Add explanations about the purpose of the model and report
    st.markdown("""
    ### Purpose of the Report

    The goal of this report is to provide an accurate forecast of retail sales, helping businesses:
    - Optimize inventory management.
    - Plan promotional strategies.
    - Improve decision-making around staffing, promotions, and stock levels.
    
    ### How the Prediction Works

    The model uses historical sales data and additional features (such as store type, competition, and promotions) to predict future sales for each store.
    
    **Key Calculations:**
    1. **Data Preprocessing**: We clean the data, handle missing values, and encode categorical variables to make them suitable for the model.
    2. **Prediction**: The trained model predicts future sales based on the preprocessed data.
    3. **Visualization**: We visualize actual vs predicted sales, the distribution of predictions, and feature correlations to provide deeper insights.
    
    ### Ultimate Purpose of the Report
    
    This sales prediction report aims to assist business owners and analysts in making informed decisions about future sales, enabling them to better prepare for demand fluctuations and market trends.
    """)

if __name__ == '__main__':
    main()

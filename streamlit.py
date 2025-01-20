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

# Page configuration: Clean & modern
st.set_page_config(page_title="🔮 Retail Sales Prediction", page_icon="📊", layout="wide")

# Apply custom CSS to make the UI futuristic and clean
st.markdown("""
    <style>
        .reportview-container {
            background-color: #1e1e1e;
        }
        .sidebar .sidebar-content {
            background-color: #212121;
            color: white;
        }
        .sidebar .sidebar-header {
            color: #76c7c0;
        }
        h1, h2, h3, h4 {
            color: #f5f5f5;
        }
        .stButton>button {
            background-color: #76c7c0;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 12px 24px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #4fa3a1;
        }
        .stAlert {
            background-color: #d32f2f;
            color: white;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Project Owner Details
def display_project_owner_details():
    st.markdown("""
        ### 👤 Project Owner: NIT,NGP - Group no. 17 (Admin Details are given below )
        - **Email**: [thedeepkumarmahajan@gmail.com](mailto:thedeepkumarmahajan@gmail.com)
        - **LinkedIn**: [Deepkumar Mahajan](https://www.linkedin.com/in/deepkumarmahajan/)
        
        For any queries, support, or further assistance, feel free to reach out to me! 😊
    """)

# Create the Streamlit web app
def main():
    # Sidebar with upload option and header
    st.sidebar.header("🔧 Upload Data & Settings")
    uploaded_file = st.sidebar.file_uploader("Upload your sales data (CSV)", type="csv")

    # File Preview and Description
    if uploaded_file is not None:
        st.sidebar.markdown("### 📄 Uploaded Data Preview")
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.write(data.head())  # Preview the first few rows

            # Display Data Summary (e.g., number of rows, stores, date range)
            data_summary = f"""
            - **Total Records**: {len(data)}
            - **Stores Included**: {data['Store'].nunique()}
            - **Date Range**: {data['Date'].min()} to {data['Date'].max()}
            """
            st.sidebar.markdown("### 📊 Data Summary")
            st.sidebar.markdown(data_summary)

            with st.spinner('⏳ Processing your file...'):
                preprocessed_data = preprocess_data(data)
                if preprocessed_data is not None:
                    predictions = make_predictions(preprocessed_data)
                    display_predictions(predictions, data)
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading file: {str(e)}")
    
    # Provide additional guidance on usage
    st.markdown("""
        ### ✨ Welcome to the Retail Sales Prediction Tool
        This tool predicts sales for different stores based on historical data. Upload your CSV and let the AI handle the rest!

        ### 🧮 How to Use This Tool:
        1. **Upload your sales data** in CSV format.
        2. The model will process the data and predict sales.
        3. Download the predictions and start optimizing your business!

        ### 📈 The Ultimate Goal:
        This tool helps businesses optimize inventory, plan promotions, and make smarter decisions based on accurate sales predictions.
    """)

    # Project owner section at the bottom of the page
    display_project_owner_details()

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

    st.subheader("💡 Predicted Sales Overview")
    st.write(result_df)

    # Add some details and download functionality
    st.markdown("""
        ### 🔍 How the Prediction Works:
        The prediction model uses historical data, including store types, promotions, customer numbers, etc., to forecast future sales.

        ### 🧮 Purpose of the Prediction:
        - **Optimize Inventory**: Avoid overstocking or understocking.
        - **Plan Promotions**: Determine the best time to run promotions.
        - **Business Insights**: Drive smarter decision-making across various departments.

        ### 📈 The Ultimate Goal:
        This tool helps businesses improve their bottom line by predicting sales, making informed decisions, and aligning stock, staffing, and marketing strategies.

    """)

    # Add a download button to export the result
    st.download_button(
        label="Download Predicted Sales (CSV)",
        data=result_df.to_csv(index=False),
        file_name="predicted_sales.csv",
        mime="text/csv",
    )

    # Add a thank you message
    st.markdown("""
        ### 🎉 Thank you for using the Retail Sales Prediction Tool! 🌟
        We hope this helps you optimize your business and achieve better results. 🚀
    """)

if __name__ == '__main__':
    main()

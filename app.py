from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the preprocessor
with open('artifacts/proprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load the model
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']

    # Read the uploaded file
    data = pd.read_csv(file)

    # Preprocess the data using the preprocessor
    preprocessed_data = preprocess_data(data)

    # Make predictions using the model
    predictions = make_predictions(preprocessed_data)

    # Combine the predictions with the original data
    result_df = pd.concat([data[['Store', 'Date']], pd.DataFrame(predictions, columns=['Expected Sales'])], axis=1)
    result_data = result_df.values.tolist()

    return render_template('result.html', result=result_data)

def preprocess_data(data):
    # Select the required columns
    columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
               'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
               'CompetitionDistance', 'CompetitionOpenSinceMonth',
               'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
               'Promo2SinceYear', 'PromoInterval']
    data = data[columns]

    # Preprocess the data using the preprocessor
    preprocessed_data = preprocessor.transform(data)

    return preprocessed_data

def make_predictions(data):
    # Make predictions using the model
    predictions = model.predict(data)

    return predictions

if __name__ == '__main__':
    app.run(debug=True)

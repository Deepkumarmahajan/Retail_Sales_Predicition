
# Retail Sales Prediction - End-to-End Regression Project

This repository contains code and resources for an end-to-end regression project on retail sales prediction. The goal of this project is to develop a regression model that can accurately predict retail sales based on various features.

![Screenshot 2025-01-20 142419](https://github.com/user-attachments/assets/df09de4e-8be0-42d3-a5b7-6f1acaa776cf)


**Steps involved in building a ML Model:**

> Step 1: Data gathering and Understanding

> Step 2: Data preparation

> Step 3: Data Cleaning

> Step 4: Exploratory data analysis

> Step 5: Feature engineering and selection

> Step 6: ML Model assumption and checks

> Step 7: Data preparation for modelling

> Step 8: Model Building

> Step 9: Model Validation & Evaluation

> Step 10: Predictions & Saving model using pickel library.


**Libraries used in EDA & Machine Learning:**
1. Pandas
2. Numpy
3. Matplotib
4. Seaborn
5. Plotly
6. Sklearn
7. Scipy


**Graphs used for representation:**
1. Bar plot
2. Pie plot
3. Box Plot
4. Grouped bar plot
5. Donut plot
6. Heatmap
7. Pair plot

**Insights from EDA impacting business:**

* The most selling and crowded store type is A.
* More stores are opened during School holidays than State holidays.
* Mondays have most sales since most of the Sundays are closed.
* Promo 1 has given positive yields where as Promo 2 is a disaster.
* Store type b has higher sales and customers per store than other store types. 
* Assortment b is available only at store type b and it has more sales and customers than any other assortment.

**Suggestions provided to increase the Sales:**

* There are very few B type stores, few more can be opened as average sales are quite high as compared to other types.
* Assortment B is only available with store type B which can be extended to other types as well to cater the demands of customers.
* Promo 2 should be discontinued and Promo 1 can be extended futher as it shows better results.
* Very few stores are opened during State Holidays, so it suggested to open a subsequent amount of stores to serve in emergency purposes.

**Model Implementation:**

* Our aim was to predict sales for a particular store using various regression algorithms and time series analysis.
*  The following algorithms were used for regression analysis: Linear Regression, Lasso Regression, Ridge Regression, Decision Tree Regressor, Extra Tree Regressor, XG Boost Regressor, and Light GBM Regressor. The models were trained on a dataset containing historical sales data for the store, along with other relevant features such as promotional events, holidays, and weather conditions.

* After evaluating the performance of each regression model using appropriate metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score, it was found that the LightGBM performed the best among all the models with the lowest MSE and RMSE.

* Next, time series analysis was also performed using Facebook Prophet library to forecast future sales for the store. The historical sales data was preprocessed and used to train a Prophet model. Future sales predictions were made for the next year using the trained model, and the results were plotted to visualize the forecasted sales.

Overall, this project demonstrated the use of various regression algorithms and time series analysis to predict sales for a particular store, with the XG Boost Regressor performing the best among all models. Additionally, the project highlighted the importance of selecting appropriate evaluation metrics and preprocessing techniques for regression analysis.




**ML Model selected for deployment: Light GBM**
> Light GBM is a fast, distributed, high-performance gradient boosting framework that uses a tree-based learning algorithm. It also supports GPU learning and is thus widely used for data science application development.

**Advantages:**
* **Faster training speed and higher efficiency**: Light GBM uses a histogram-based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure.

* **Lower memory usage**: Replaces continuous values to discrete bins which results in lower memory usage.

*  **Better accuracy** : It produces much more complex trees by following **leaf wise split approach** rather than a **level-wise approach** which is the main factor in achieving higher accuracy.

* **Good Compatibility with Large Datasets**: It is capable of performing equally well with large datasets with a significantly less  training time as compared to XGBoost.


**Limitations:**

* **Complexity**: Light GBM split the tree leaf-wise which can lead to overfitting as it produces much complex trees.
* **Overfitting**: Light GBM is sensitive to overfitting and thus can easily overfit small dataset.


**Suggestion:**
* When we are dealing with huge dataset & time is a constraint use Light GBM Model else when dataset is small than XGBoost can provide better results.

## Dataset

The dataset used for this project contains the following columns:

- **Store**: The store ID.
- **DayOfWeek**: The day of the week (1-7, where 1 is Monday and 7 is Sunday).
- **Date**: The date of the sales record.
- **Sales**: The total sales for the given day and store.
- **Customers**: The number of customers on the given day and store.
- **Open**: Indicates whether the store was open (1) or closed (0) on the given day.
- **Promo**: Indicates whether a promotional offer was active (1) or not (0) on the given day.
- **StateHoliday**: Indicates whether the day was a state holiday (a, b, c) or not (0).
- **SchoolHoliday**: Indicates whether the day was a school holiday (1) or not (0).
- **StoreType**: The type of store (a, b, c, d).
- **Assortment**: The assortment level of the store (a = basic, b = extra, c = extended).
- **CompetitionDistance**: The distance to the nearest competitor store.
- **CompetitionOpenSinceMonth**: The month when the nearest competitor store opened.
- **CompetitionOpenSinceYear**: The year when the nearest competitor store opened.
- **Promo2**: Indicates whether a continuous promotional offer is active (1) or not (0).
- **Promo2SinceWeek**: The week of the year when the continuous promotional offer started.
- **Promo2SinceYear**: The year when the continuous promotional offer started.
- **PromoInterval**: The intervals at which the continuous promotional offer is repeated.

## Project Structure

The project structure is organized as follows:
```
├───artifacts
├───logs
├───notebook
│   └───data
├───src
│   ├───components
│   ├───pipeline
├───static
├───templates
```


- `notebook/`: Jupyter notebooks for data exploration, preprocessing, and model development.
- `notebook/data/`: contains the dataset file(s).
- `src/`: Source code for the project, including preprocessing functions and model training .
- `pipeline/`:Source code for different implemented pipelines.
- `artifacts/`: Directory to store model and evaluation results and perform predictions.
- `static/` and `templates/` contains basic frontend framework for deployment using flask.


## Getting Started

To get started with this project, you can follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/Deepkumarmahajan/Retail_Sales_Predicition.git
   ```
2. Create virtual environment
   ```
   conda create -p venv python==3.8
   conda activate venv/
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Explore the collab notebooks in the `notebooks/` directory to understand the data and the steps involved in preprocessing and training the regression model.

5. Run the data ingestion script to preprocess,tranform the dataset along with training the model:

   ```
   python src/data_ingestion.py
   ```



6. Tune the model by changing parameters in model_trainer.py:

   ```
   python src/model_trainer.py
   ```

7. Run flask app by using 
`python app.py`


8. Streamlit deployment link https://deepkumarmahajan-retailsalespredicition.streamlit.app/

9. Load the desired file for predicting the result.

Feel free to modify the code and experiment with different models and techniques to improve the prediction accuracy.
## Acknowledgments

- The dataset used in this project is obtained from https://www.kaggle.com/competitions/rossmann-store-sales/data.



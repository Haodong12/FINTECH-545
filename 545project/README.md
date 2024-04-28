Scripts Overview

data_prep.py
Fetches historical stock price data for AAPL and the NASDAQ Composite Index (as a market reference) using Yahoo Finance (yfinance library).
Prepares the dataset by calculating several technical indicators like moving averages (MA5, MA10, MA20, MA50) and percentage changes in stock prices.
Saves the prepared data with features and targets into a CSV file for model training.

training.py
Loads the prepared stock data and splits it into training and testing sets based on the date.
Defines a Random Forest Regressor and a Linear Regression model to predict the next day's closing price of AAPL stock.
Trains both models on historical data and evaluates them using metrics like RMSE, MAE, and R-squared.
Saves the trained models and predictions for further analysis.

plotting.py
Visualizes the performance of the trained models by generating various plots, including:
Kernel Density Estimation (KDE) plots to compare the distribution of actual returns with predicted returns.
Residual plots to evaluate the prediction errors of both models.
Boxplots to show the distribution of predictions for individual observations.
Line plots to compare actual vs. predicted stock price movements over time.
Data Sources
Historical stock prices for AAPL and the NASDAQ index are retrieved from Yahoo Finance.

Requirements
To run the scripts, ensure you have the following packages installed:

pandas
numpy
matplotlib
seaborn
scipy
yfinance
sklearn

Usage
To prepare the data, run:
python data_prep.py

To train the models and save the predictions, run:
python training.py

To generate and save the plots, run:
python plotting.py

Conclusion
These scripts serve as a foundation for predicting AAPL stock prices. The Linear Regression model has shown notable accuracy in our tests, but the final choice of model should consider the specific financial context and risk profile desired.

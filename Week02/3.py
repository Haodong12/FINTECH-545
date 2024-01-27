import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to read the CSV file
def parse_csv(file_name):
    data = pd.read_csv(file_name)
    return data

# Reading the dataset
df = parse_csv('problem3.csv')
time_series_values = df['x']  # Extracting the time series data

# Function to analyze model fit using ACF and PACF
def analyze_model_fit(model_order, series_data):
    try:
        model = ARIMA(series_data, order=model_order)
        fitted_model = model.fit()
        fig, (acf_ax, pacf_ax) = plt.subplots(1, 2, figsize=(16, 9))
        plot_acf(fitted_model.resid, lags=20, ax=acf_ax, title=f'Residuals ACF - Order {model_order}', zero=False)
        plot_pacf(fitted_model.resid, lags=20, ax=pacf_ax, title=f'Residuals PACF - Order {model_order}', zero=False)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Plotting the time series data
plt.figure(figsize=(10, 6))
plt.plot(time_series_values)
plt.title('Time Series Data')
plt.xlabel('Observation Number')
plt.ylabel('Series Value')
plt.show()

# Function to fit ARIMA model and return AIC and BIC statistics
def fit_arima_and_get_stats(model_order, series_data):
    model = ARIMA(series_data, order=model_order)
    model_fit_results = model.fit()
    aic_value = model_fit_results.aic
    bic_value = model_fit_results.bic
    return aic_value, bic_value

# Dictionary to store AIC and BIC values for each model
model_stats = {}

# Fit AR models (AR(1) to AR(3)) and store their AIC and BIC
for ar_order in range(1, 4):
    model_stats[f'AR({ar_order})'] = fit_arima_and_get_stats((ar_order, 0, 0), time_series_values)
    analyze_model_fit((ar_order, 0, 0), time_series_values)

# Fit MA models (MA(1) to MA(3)) and store their AIC and BIC
for ma_order in range(1, 4):
    model_stats[f'MA({ma_order})'] = fit_arima_and_get_stats((0, 0, ma_order), time_series_values)
    analyze_model_fit((0, 0, ma_order), time_series_values)

# Convert the stats dictionary to a DataFrame 
stats_dataframe = pd.DataFrame(model_stats, index=['AIC', 'BIC']).T
print(stats_dataframe)

import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.arima.model import ARIMA

# Function to load CSV data and calculate returns
def return_calculate(file_path, method="DISCRETE", date_column="Date"):
    prices = pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
    
    if method.upper() == "DISCRETE":
        returns = prices.pct_change().dropna()
    elif method.upper() == "LOG":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be either 'LOG' or 'DISCRETE'")
    
    return returns

# Function to calculate various VaR
def calculate_var(returns, confidence_level=0.05, lambda_=0.94):
    # Adjust 'META' returns to have a mean of 0
    adjusted_returns = returns - returns.mean()
    
    # 1. VaR using a normal distribution
    var_normal = norm.ppf(confidence_level, adjusted_returns.mean(), adjusted_returns.std())

    # 2. VaR using EWM
    ewm_var = adjusted_returns.ewm(alpha=(1 - lambda_)).var().iloc[-1]
    var_ewm = norm.ppf(confidence_level, adjusted_returns.mean(), np.sqrt(ewm_var))

    # 3. VaR using MLE fitted T distribution
    params = t.fit(adjusted_returns)
    var_t = t.ppf(confidence_level, *params[:-2], loc=params[-2], scale=params[-1])

    # 4. VaR using AR(1) model
    model = ARIMA(adjusted_returns, order=(1,0,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    var_ar1 = norm.ppf(confidence_level, forecast, adjusted_returns.std())

    # 5. VaR using Historical Simulation
    var_hs = np.percentile(adjusted_returns, confidence_level * 100)

    return var_normal, var_ewm, var_t, var_ar1, var_hs

data_file = "DailyPrices.csv"  
arithmetic_returns = return_calculate(data_file, "DISCRETE")

# Adjusting the META returns
if 'META' in arithmetic_returns.columns:
    mean_meta = arithmetic_returns['META'].mean()
    arithmetic_returns['META'] = arithmetic_returns['META'] - mean_meta
    print("\nAdjusted META Returns with Mean=0:")
    print(arithmetic_returns['META'])

    # Calculate VaR for the adjusted 'META' returns
    var_results = calculate_var(arithmetic_returns['META'], 0.05)
    print(f"\nVaR (Normal Distribution): {var_results[0]}")
    print(f"VaR (EWM): {var_results[1]}")
    print(f"VaR (MLE T Distribution): {var_results[2]}")
    print(f"VaR (AR(1) Model): {var_results[3]}")
    print(f"VaR (Historical Simulation): {var_results[4]}")

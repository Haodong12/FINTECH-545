import pandas as pd
import numpy as np
from scipy.stats import t, norm

# Function to calculate exponentially weighted variance
def exp_weighted_var(data, lam=0.97):
    weights = np.array([(1-lam) * (lam**i) for i in reversed(range(len(data)))])
    weighted_mean = np.sum(weights * data) / np.sum(weights)
    weighted_var = np.sum(weights * (data - weighted_mean) ** 2) / np.sum(weights)
    return weighted_var, weighted_mean

# Function to fit distributions and calculate VaR, ES in dollar value
def calculate_var_es_dollar(returns, portfolio_value, distribution='normal', alpha=0.05):
    if distribution == 'normal':
        variance, mean = exp_weighted_var(returns)
        std = np.sqrt(variance)
        VaR = norm.ppf(1 - alpha) * std
        ES = (1 / alpha) * norm.pdf(norm.ppf(1 - alpha)) * std - mean
    elif distribution == 't':
        params = t.fit(returns)
        VaR = t.ppf(1 - alpha, *params)
        ES = -t.mean(*params)  # This is a placeholder. Proper calculation for T distribution's ES might be different.
    return VaR * portfolio_value, ES * portfolio_value

# Assume file paths are corrected
daily_prices_df = pd.read_csv('/Users/Lenovo/Desktop/Week5/covariance/DailyPrices.csv').set_index('Date')
portfolio_df = pd.read_csv('/Users/Lenovo/Desktop/Week5/covariance/portfolio.csv')

# Calculate returns
returns_df = daily_prices_df.pct_change().dropna()

# Example total portfolio value in dollars
portfolio_value = 100000  

# Initialize results
results = {'Portfolio': [], 'VaR': [], 'ES': []}

# Iterate through portfolios
for portfolio in ['A', 'B', 'C']:
    stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]['Stock']
    portfolio_returns = returns_df[stocks].mean(axis=1)  # Simplified approach
    distribution = 't' if portfolio in ['A', 'B'] else 'normal'
    
    VaR, ES = calculate_var_es_dollar(portfolio_returns, portfolio_value, distribution=distribution)
    results['Portfolio'].append(portfolio)
    results['VaR'].append(VaR)
    results['ES'].append(ES)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate total VaR and ES
total_VaR = sum(results['VaR'])
total_ES = sum(results['ES'])

# Correct way to append a row for total results using pd.concat
total_df = pd.DataFrame({'Portfolio': ['Total'], 'VaR': [total_VaR], 'ES': [total_ES]})
results_df = pd.concat([results_df, total_df], ignore_index=True)

# Display or save results
print(results_df)
# results_df.to_csv('/path/to/save/results.csv', index=False)




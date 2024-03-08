import pandas as pd
import numpy as np
from scipy.stats import norm, t

# Pseudocode for advanced functions
def calculate_var(returns, alpha=0.05):
    # Calculate the VaR at the given alpha level
    return np.quantile(returns, alpha)

def calculate_es(returns, alpha=0.05):
    # Calculate the ES at the given alpha level
    # ES is the expected return on the portfolio in the worst alpha% of cases
    return returns[returns <= np.quantile(returns, alpha)].mean()

# Read the portfolio details
portfolio_df = pd.read_csv('/Users/Lenovo/Desktop/Week5/test_files/test9_1_portfolio.csv')

# Read the return series
returns_df = pd.read_csv('/Users/Lenovo/Desktop/Week5/test_files/test9_1_returns.csv')

# Calculate VaR and ES for each stock and the total portfolio
results = []
for stock in portfolio_df['Stock']:
    stock_returns = returns_df[stock]
    stock_starting_price = portfolio_df.loc[portfolio_df['Stock'] == stock, 'Starting Price'].iloc[0]
    
    var = calculate_var(stock_returns)
    es = calculate_es(stock_returns)
    
    results.append({
        'Stock': stock,
        'VaR95': var * stock_starting_price,
        'ES95': es * stock_starting_price,
        'VaR95_Pct': var,
        'ES95_Pct': es
    })

# Calculate for Total Portfolio
portfolio_returns = returns_df.sum(axis=1)
portfolio_starting_price = portfolio_df['Starting Price'].sum()

var_total = calculate_var(portfolio_returns)
es_total = calculate_es(portfolio_returns)

results.append({
    'Stock': 'Total',
    'VaR95': var_total * portfolio_starting_price,
    'ES95': es_total * portfolio_starting_price,
    'VaR95_Pct': var_total,
    'ES95_Pct': es_total
})

# Create DataFrame and save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv('testout_9.1.csv', index=False)




import pandas as pd
import numpy as np
from scipy.stats import norm

# Constants for VaR calculation
LAMBDA = 0.94
CONFIDENCE_LEVEL = 0.05

# Load portfolio holdings and price data
portfolio_df = pd.read_csv('portfolio.csv')
prices_df = pd.read_csv('DailyPrices.csv', index_col=0, parse_dates=True)

# Calculate daily returns
daily_returns = prices_df.pct_change().dropna()

# Define latest_prices here
latest_prices = prices_df.iloc[-1]

portfolio_df['Value'] = portfolio_df.apply(lambda row: row['Holding'] * latest_prices[row['Stock']], axis=1)
portfolio_total_values = {portfolio: portfolio_df[portfolio_df['Portfolio'] == portfolio]['Value'].sum() for portfolio in portfolio_df['Portfolio'].unique()}

def calculate_portfolio_returns(portfolio_df, daily_returns, latest_prices):
    portfolio_returns = pd.DataFrame(index=daily_returns.index)
    unique_portfolios = portfolio_df['Portfolio'].unique()
    for portfolio in unique_portfolios:
        # Filter stocks and their holdings for the current portfolio
        stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]['Stock']
        holdings = portfolio_df[portfolio_df['Portfolio'] == portfolio].set_index('Stock')['Holding']
        
        # Calculate weights
        values = holdings * latest_prices.reindex(stocks).fillna(0)
        weights = values / values.sum()
        
        # Ensure alignment by reindexing daily_returns to include only relevant stocks
        aligned_daily_returns = daily_returns.reindex(columns=stocks, fill_value=0)
        
        # Calculate weighted returns
        portfolio_returns[portfolio] = aligned_daily_returns.dot(weights)
    return portfolio_returns

# Now pass latest_prices as an argument to the function
portfolio_returns = calculate_portfolio_returns(portfolio_df, daily_returns, latest_prices)

# Calculate Historical VaR (Arithmetic) for each portfolio
historical_var_portfolios = portfolio_returns.quantile(CONFIDENCE_LEVEL)

# Calculate Lognormal VaR for each portfolio
log_returns = np.log1p(portfolio_returns)
mean_log_returns = log_returns.mean()
std_log_returns = log_returns.std()
lognormal_var_portfolios = np.exp(mean_log_returns + norm.ppf(CONFIDENCE_LEVEL) * std_log_returns) - 1

# Calculate the total portfolio returns and VaR
total_portfolio_returns = portfolio_returns.sum(axis=1) / len(portfolio_returns.columns)
historical_var_total = total_portfolio_returns.quantile(CONFIDENCE_LEVEL)
log_returns_total = np.log1p(total_portfolio_returns)
mean_log_returns_total = log_returns_total.mean()
std_log_returns_total = log_returns_total.std()
lognormal_var_total = np.exp(mean_log_returns_total + norm.ppf(CONFIDENCE_LEVEL) * std_log_returns_total) - 1

# Print the VaR results for each portfolio and the total portfolio
def calculate_var_in_dollars(var_fraction, portfolio_total_value):
    return var_fraction * portfolio_total_value

# Calculate total value for each portfolio
portfolio_total_values = {portfolio: portfolio_df[portfolio_df['Portfolio'] == portfolio]['Value'].sum() for portfolio in portfolio_df['Portfolio'].unique()}

# Calculate Historical VaR (Arithmetic) and Lognormal VaR in $ for each portfolio
historical_var_dollars = {portfolio: calculate_var_in_dollars(historical_var_portfolios[portfolio], portfolio_total_values[portfolio]) for portfolio in historical_var_portfolios.index}
lognormal_var_dollars = {portfolio: calculate_var_in_dollars(lognormal_var_portfolios[portfolio], portfolio_total_values[portfolio]) for portfolio in lognormal_var_portfolios.index}

# Calculate the total portfolio value
total_portfolio_value = sum(portfolio_total_values.values())

# Calculate total portfolio Historical VaR (Arithmetic) and Lognormal VaR in $
total_historical_var_dollars = calculate_var_in_dollars(historical_var_total, total_portfolio_value)
total_lognormal_var_dollars = calculate_var_in_dollars(lognormal_var_total, total_portfolio_value)

# Print the VaR results in $ for each portfolio and the total portfolio
print("Historical VaR (Arithmetic) in $ for each portfolio:")
for portfolio, var in historical_var_dollars.items():
    print(f"Portfolio {portfolio}: ${var:.2f}")

print("\nLognormal VaR in $ for each portfolio:")
for portfolio, var in lognormal_var_dollars.items():
    print(f"Portfolio {portfolio}: ${var:.2f}")

print(f"\nTotal Portfolio Historical VaR (Arithmetic) in $: ${total_historical_var_dollars:.2f}")
print(f"Total Portfolio Lognormal VaR in $: ${total_lognormal_var_dollars:.2f}")
import pandas as pd
import numpy as np
from scipy.stats import t, norm

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['x'].dropna()

# Exponentially weighted moving variance
def ewm_variance(data, lambda_factor=0.97):
    weights = np.array([(1 - lambda_factor) * (lambda_factor ** i) for i in range(len(data))][::-1])
    weights /= weights.sum()
    mean = np.average(data, weights=weights)
    variance = np.average((data - mean)**2, weights=weights)
    return mean, variance

# Calculate VaR and ES using a normal distribution with EW variance
def var_es_normal(data, alpha=0.05, lambda_factor=0.97):
    mean, variance = ewm_variance(data, lambda_factor)
    std = np.sqrt(variance)
    vaR = norm.ppf(alpha, mean, std)
    es = mean + std * norm.pdf(norm.ppf(alpha)) / alpha
    return vaR, es

# Fit T distribution using MLE and calculate VaR and ES
def var_es_t_distribution(data, alpha=0.05):
    params = t.fit(data)
    df, loc, scale = params
    vaR = t.ppf(alpha, df, loc=loc, scale=scale)
    es = loc + scale * (df + (t.ppf(alpha, df) ** 2)) / (df-1) * t.pdf(t.ppf(alpha, df), df) / alpha
    return vaR, es

# Calculate VaR and ES using historic simulation
def var_es_historic(data, alpha=0.05):
    sorted_returns = np.sort(data)
    vaR = np.percentile(sorted_returns, alpha*100)
    es = sorted_returns[sorted_returns <= vaR].mean()
    return vaR, es

# Example usage
file_path = '/Users/Lenovo/Desktop/Week5/covariance/problem1.csv'  # Replace with the path to your CSV file
data = load_data(file_path)

# Using a normal distribution with EW variance
vaR_normal, es_normal = var_es_normal(data)

# Using a MLE fitted T distribution
vaR_t, es_t = var_es_t_distribution(data)

# Using a Historic Simulation
vaR_hist, es_hist = var_es_historic(data)

# Display or save your results
print("Normal Distribution (EW variance): VaR = {}, ES = {}".format(vaR_normal, es_normal))
print("T Distribution (MLE fitted): VaR = {}, ES = {}".format(vaR_t, es_t))
print("Historic Simulation: VaR = {}, ES = {}".format(vaR_hist, es_hist))






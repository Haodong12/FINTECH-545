import numpy as np
import pandas as pd
from scipy.stats import norm, t

def calculate_var_from_distribution(data, distribution='normal', alpha=0.05):
    if distribution == 'normal':
        mu, std = norm.fit(data)
        var = -norm.ppf(alpha, loc=mu, scale=std)
        var_diff = var - mu
    elif distribution == 't':
        df, loc, scale = t.fit(data)
        var = -t.ppf(alpha, df, loc=loc, scale=scale)
        var_diff = var - loc
    else:
        raise ValueError("Unsupported distribution type")
    return var, var_diff

def save_var_to_csv(file_path, output_path, distribution):
    data = pd.read_csv(file_path)['x1']
    var, var_diff = calculate_var_from_distribution(data, distribution)
    results_df = pd.DataFrame({'VaR Absolute': [var], 'VaR Diff from Mean': [var_diff]})
    results_df.to_csv(output_path, index=False)

# Task 1: VaR from Normal Distribution
save_var_to_csv('test7_1.csv', 'testout_8.1.csv', 'normal')

# Task 2: VaR from T Distribution
save_var_to_csv('test7_2.csv', 'testout_8.2.csv', 't')

def simulate_and_save_var_t_distribution(input_path, output_path, alpha=0.05, simulations=10000):
    # Read the fitted T distribution parameters
    fitted_params = pd.read_csv(input_path)
    df, loc, scale = fitted_params['Value']

    # Simulate data based on the T distribution
    simulated_data = t.rvs(df, loc=loc, scale=scale, size=simulations)
    
    # Calculate the VaR from the simulated data
    var = -np.percentile(simulated_data, alpha * 100)
    var_diff = var - np.mean(simulated_data)

    # Save the results
    results_df = pd.DataFrame({'VaR Absolute': [var], 'VaR Diff from Mean': [var_diff]})
    results_df.to_csv(output_path, index=False)

# Task 3: VaR from Simulation -- compare to 8.2 values
simulate_and_save_var_t_distribution('test7_2.csv', 'testout_8.3.csv')

def calculate_es_from_distribution(data, distribution='normal', alpha=0.05):
    if distribution == 'normal':
        mu, std = norm.fit(data)
        var = -norm.ppf(alpha, loc=mu, scale=std)
        es = -norm.expect(lambda x: x, loc=mu, scale=std, lb=var)
        es_diff = es - mu
    elif distribution == 't':
        df, loc, scale = t.fit(data)
        var = -t.ppf(alpha, df, loc=loc, scale=scale)
        es = -t.expect(lambda x: x, df, loc=loc, scale=scale, lb=var)
        es_diff = es - loc
    else:
        raise ValueError("Unsupported distribution type")
    return es, es_diff

def save_es_to_csv(file_path, output_path, distribution):
    data = pd.read_csv(file_path)['x1']
    es, es_diff = calculate_es_from_distribution(data, distribution)
    results_df = pd.DataFrame({'ES Absolute': [es], 'ES Diff from Mean': [es_diff]})
    results_df.to



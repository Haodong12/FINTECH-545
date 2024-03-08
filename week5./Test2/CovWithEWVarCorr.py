import numpy as np
import pandas as pd

def exponentially_weighted_covariance(data, lambda_value):
    num_observations, num_variables = data.shape
    weights = np.empty(num_observations)
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean
    
    for i in range(num_observations):
        weights[i] = (1 - lambda_value) * lambda_value**(num_observations - i - 1)
    weights /= np.sum(weights)
    
    weighted_data = centered_data.T * weights
    covariance_matrix = weighted_data @ centered_data
    
    return covariance_matrix

# Function to load data from a CSV file into a numpy array
def load_data(file_path):
    return pd.read_csv(file_path).values

# Example usage:
file_path = '/Users/Lenovo/Desktop/Week5/test_files/test2.csv' # Replace with your actual data file path
data = load_data(file_path)

# Calculation process:
# First, calculate the exponentially weighted covariance with lambda = 0.97
ew_covariance_097 = exponentially_weighted_covariance(data, 0.97)
standard_deviations_097 = np.sqrt(np.diag(ew_covariance_097))

# Then, calculate the exponentially weighted covariance again with lambda = 0.94
ew_covariance_094 = exponentially_weighted_covariance(data, 0.94)
standard_deviations_094 = 1 / np.sqrt(np.diag(ew_covariance_094))

# Final calculation for the adjusted covariance matrix
adjusted_covariance_matrix = np.diag(standard_deviations_097) @ np.diag(standard_deviations_094) @ ew_covariance_094 @ np.diag(standard_deviations_094) @ np.diag(standard_deviations_097)

# Print the result
print("Covariance Matrix:")
print(adjusted_covariance_matrix)


'''
file_path = '/Users/Lenovo/Desktop/Week5/test_files/test2.csv' 
data_df = load_data_from_csv(file_path)
data_array = data_df.values # Convert the DataFrame to a NumPy array

lambda_var = 0.94
lambda_corr = 0.97
result = ew_cov_var_corr(data_array, lambda_var, lambda_corr)
print(result)'''
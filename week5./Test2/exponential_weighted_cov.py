import pandas as pd
import numpy as np

def populate_weights(n, lambda_value):
    weights = np.zeros(n)
    cumulative_weights = np.zeros(n)
    total_weight = 0.0

    # Calculate the weights in reverse order
    for i in range(n):
        weights[i] = (1 - lambda_value) * (lambda_value ** (n - i - 1))
        total_weight += weights[i]
        cumulative_weights[i] = total_weight

    # Normalize the weights and cumulative weights
    weights /= total_weight
    cumulative_weights /= total_weight
    #print(weights)
    #print(cumulative_weights)

    return weights, cumulative_weights

def ew_covariance(df, weights):
    # Demean the dataframe
    demeaned_df = df - df.mean()
    #print(demeaned_df.head())
    # Initialize the covariance matrix
    cov_matrix = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), index=df.columns, columns=df.columns)
    #print(cov_matrix)

    # Calculate the weighted covariance matrix
    for i in range(len(df)):
        row = demeaned_df.iloc[i, :] #ith row, all cols
        outer_product = np.outer(row, row)
        cov_matrix += weights[i] * outer_product
    return cov_matrix

# Load your data
df = pd.read_csv('/Users/Lenovo/Desktop/Week5/test_files/test2.csv')  

lambda_values = [0.97]

# Calculate and print the EW covariance matrices for each lambda value
for lambda_value in lambda_values:
    weights, _ = populate_weights(len(df), lambda_value) #",_" means a common way to skip returning the second output of populate_weights
    cov_matrix = ew_covariance(df, weights)

    # Save the covariance matrix to a CSV file
    #csv_filename = f"ew_covariance_matrix_lambda_{lambda_value}.csv"
    #cov_matrix.to_csv(csv_filename, index=False)

    print(f"EW Covariance Matrix for lambda={lambda_value}:")
    print(cov_matrix)
    print("\n")

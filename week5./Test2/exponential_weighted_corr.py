import pandas as pd
import numpy as np

def ew_correlation(df, lambda_value):
    # Calculate the decay factors for the weights
    n = len(df)
    decay_factors = np.flip((1 - lambda_value) * (lambda_value ** np.arange(n)))
    weights = decay_factors / decay_factors.sum()

    # Demean the data
    demeaned = df.sub(df.mean())
    
    # Apply the weights to the demeaned data
    weighted_demeaned = demeaned.mul(np.sqrt(weights), axis=0)
    
    # Calculate the covariance matrix of the weighted demeaned data
    cov_matrix = weighted_demeaned.T.dot(weighted_demeaned)
    
    # Calculate the standard deviations
    std = np.sqrt(np.diag(cov_matrix))
    
    # Create the correlation matrix
    corr_matrix = cov_matrix.div(std, axis='rows').div(std, axis='columns')
    
    return corr_matrix

# Load the data
data = pd.read_csv("/Users/Lenovo/Desktop/Week5/test_files/test1.csv")  # Replace with the correct path to your data file


# Calculate the exponentially weighted correlation matrix with lambda=0.94
lambda_value = 0.97
ew_corr_matrix = ew_correlation(data, lambda_value)

print("Exponentially Weighted Correlation Matrix with lambda=0.94:")
print(ew_corr_matrix)

csv_filename = f"ew_correlation_matrix_lambda_{lambda_value}.csv"
ew_corr_matrix.to_csv(csv_filename, index=False)

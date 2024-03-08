import numpy as np
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    return df.values

def chol_psd_with_adjustment(matrix, epsilon=1e-10):
    adjusted_matrix = matrix + epsilon * np.eye(matrix.shape[0])
    try:
        L = np.linalg.cholesky(adjusted_matrix)
        return L
    except np.linalg.LinAlgError:
        print("Adjusted matrix is still not positive definite.")
        return None

# Correctly define and load the matrix before using it
file_path = '/Users/Lenovo/Desktop/Week5/test_files/testout_3.1.csv'  # Adjust the file path as necessary
matrix = load_data(file_path)

# Attempt Cholesky decomposition with diagonal adjustment on the loaded matrix
L_adjusted = chol_psd_with_adjustment(matrix)

if L_adjusted is not None:
    print("Cholesky Decomposition with Diagonal Adjustment (L):")
    print(L_adjusted)
else:
    print("Failed to compute Cholesky Decomposition even with diagonal adjustment.")

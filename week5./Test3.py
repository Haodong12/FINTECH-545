import numpy as np
import pandas as pd

def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    if a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    invSD = None
    out = a.copy()

    if not np.allclose(np.diag(out), 1):
        std_devs = np.sqrt(np.diag(out))
        std_devs[std_devs <= 0] = epsilon  # Prevent division by zero or negative square roots
        invSD = np.diag(1.0 / std_devs)
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    out = vecs @ np.diag(vals) @ vecs.T

    if invSD is not None:
        invSD = np.diag(std_devs)
        out = invSD @ out @ invSD

    return out

def higham_nearest_psd(A, max_iter=100, tol=1e-8):
    n = A.shape[0]
    Y = A
    for k in range(max_iter):
        X = np.copy(Y)
        R = np.linalg.cholesky((X + X.T)/2)
        Y = np.linalg.inv(R).T @ np.linalg.inv(R)
        norm_diff = np.linalg.norm(Y - X, 'fro')
        if norm_diff < tol:
            break
    return (Y + Y.T) / 2

def load_data(file_path):
    # Load the CSV, correctly handling the first row as headers
    df = pd.read_csv(file_path)
    # Convert the DataFrame to a numpy array for processing
    return df.values

# Updated file paths
file_path_31 = '/Users/Lenovo/Desktop/Week5/test_files/testout_1.3.csv'  # For near_psd covariance
file_path_32 = '/Users/Lenovo/Desktop/Week5/test_files/testout_1.4.csv'  # For near_psd correlation

try:
    # Load and process matrices
    cin_31 = load_data(file_path_31)
    cout_31 = near_psd(cin_31)
    print("Near PSD Covariance Matrix:")
    print(cout_31)
    higham_covariance_31 = higham_nearest_psd(cin_31)
    print("Higham Near PSD Covariance Matrix:")
    print(higham_covariance_31)

    cin_32 = load_data(file_path_32)
    cout_32 = near_psd(cin_32)
    print("\nNear PSD Correlation Matrix:")
    print(cout_32)
    higham_correlation_32 = higham_nearest_psd(cin_32)
    print("\nHigham Near PSD Correlation Matrix:")
    print(higham_correlation_32)
    
except ValueError as e:
    print(e)







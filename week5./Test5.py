import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.linalg import eigh

def near_psd(a, epsilon=0.0):
    """
    Adjusts a square matrix to be near positive semi-definite by setting any negative eigenvalues to epsilon.
    """
    vals, vecs = eigh(a)
    # Thresholding negative eigenvalues to epsilon
    vals = np.maximum(vals, epsilon)
    # Reconstructing the matrix
    a_psd = vecs @ np.diag(vals) @ vecs.T
    return a_psd

# Function to read covariance matrix from a CSV file
def read_cov_matrix(file_path):
    df = pd.read_csv(file_path)
    return df.values

# Function to generate samples
def generate_samples(cov_matrix, n_samples=100000, seed=42):
    rng = default_rng(seed)
    mean = np.zeros(cov_matrix.shape[0])
    return rng.multivariate_normal(mean, cov_matrix, n_samples)

# Function to compare covariances (returns difference matrix)
def compare_covariances(input_cov, samples):
    output_cov = np.cov(samples, rowvar=False)
    difference = input_cov - output_cov
    return difference

# Subtask 1: Normal Simulation PD/PSD Input
file_path_1 = '/Users/Lenovo/Desktop/Week5/test_files/test5_1.csv'  # Update this path for your environment
input_cov_matrix_1 = read_cov_matrix(file_path_1)
samples_1 = generate_samples(input_cov_matrix_1)
difference_matrix_1 = compare_covariances(input_cov_matrix_1, samples_1)
print("Subtask 1 - Difference Matrix:")
print(difference_matrix_1)

# Subtask 2: Normal Simulation with nonPSD Input, near_psd fix
file_path_2 = '/Users/Lenovo/Desktop/Week5/test_files/test5_2.csv'  # Update this path for your environment
input_cov_matrix_2 = read_cov_matrix(file_path_2)
# Assuming the near_psd function as defined in previous solutions
input_cov_matrix_2_psd = near_psd(input_cov_matrix_2)  
samples_2 = generate_samples(input_cov_matrix_2_psd)
difference_matrix_2 = compare_covariances(input_cov_matrix_2_psd, samples_2)
print("\nSubtask 2 - Difference Matrix:")
print(difference_matrix_2)

# Higham's algorithm approximation for positive semi-definite fix
def higham_nearest_psd(A, epsilon=0.0):
    """
    A simplified approximation of Higham's algorithm to find the nearest PSD matrix to A.
    """
    vals, vecs = eigh(A)
    # Adjusting eigenvalues
    vals[vals < epsilon] = epsilon
    # Reconstructing the matrix
    return vecs @ np.diag(vals) @ vecs.T


# Example usage for Task 3 and 4
file_path_3 = '/Users/Lenovo/Desktop/Week5/test_files/test5_1.csv'  # Adjust this path
input_cov_matrix_3 = read_cov_matrix(file_path_3)

# Task 3: Normal Simulation nonPSD Input, near_psd fix
input_cov_matrix_3_near_psd = near_psd(input_cov_matrix_3)
samples_3 = generate_samples(input_cov_matrix_3_near_psd)
difference_matrix_3 = compare_covariances(input_cov_matrix_3_near_psd, samples_3)
print("Task 3 - Difference Matrix (near_psd):")
print(difference_matrix_3)

# Task 4: Normal Simulation PSD Input, higham fix
input_cov_matrix_3_higham = higham_nearest_psd(input_cov_matrix_3)
samples_4 = generate_samples(input_cov_matrix_3_higham)
difference_matrix_4 = compare_covariances(input_cov_matrix_3_higham, samples_4)
print("Task 4 - Difference Matrix (Higham):")
print(difference_matrix_4)
